"""Train the pairwise matcher and score the full decision layer offline.

Splits by ENTITY, never by pair: the same entity's candidates appearing in both
train and validation would leak. The corpus stays whole for both splits, so the
validation entities face the true distractor density.

The disjoint rows only see conflicts among the sampled validation entities, so a
small n_val understates them. n_val = 0 takes every entity not used for training,
which is the closest this gets to predict.py's whole-country competition.

Singleton head (singleton.py): P(n=0) per entity, replacing prod(1 - p) in
choose_k. Its training features must come from OUT-OF-FOLD pair scores, so the
training ENTITIES are split into OOF_K folds, a pair model is trained on the
other folds and scores the held-out one, and the head is fit on those scores.
It is calibrated on the validation calibration half and gated on the held-out
half: it is saved for predict.py --singleton on only if its precision on the
entities it makes abstain is at least SINGLETON_GATE and its macro F0.5 is no
lower than the product rule's, else the product rule stays.

Hard negatives (hard_negatives.py): HARD_NEG="blocker,sibling,same_name"
reweights the training negatives to that mix, using ground-truth record owners
to find which negatives are gold matches of a near-duplicate S1 entity. It also
trains the unchanged baseline on the same run and prints both on the held-out
half; the reweighted model is the one saved. Default off.

--profile prints wall clock, throughput and peak RSS per pipeline stage, per
country (blocking and features) and overall (profiling.py). Default off.

Usage: python train_eval.py [country] [n_train] [n_val] [--profile]
"""
import gc
import os
import pathlib
import pickle
import sys
import time

import lightgbm as lgb
import numpy as np
import pyarrow.parquet as pq
from sklearn.isotonic import IsotonicRegression

import blocking
import disjoint
import features
import hard_negatives
import singleton
import strfeatures
from metric import choose_k, f05
from profiling import PROF, pop_flag, stage
from textnorm import norm

ROOT = "data/parquet"
# Per-entity candidate cap. This, not the df cap, is what binds recall once the
# channels retrieve widely: raising DF_CAP 10x produced 60.0 candidates/entity
# against 59.4, because everything above this is trimmed. Env-overridable so the
# recall/cost frontier can actually be swept.
CAND_CAP = int(os.environ.get("CAND_CAP", 150))
OOF_K = 4
# break_even_precision(0.9) = 0.474 is the knife edge; 0.55 leaves margin for
# noise in both the precision estimate and f_alt (singleton.py).
SINGLETON_GATE = 0.55
PAIR_PARAMS = dict(objective="binary", learning_rate=0.06, num_leaves=63,
                   min_data_in_leaf=50, feature_fraction=0.9, bagging_fraction=0.8,
                   bagging_freq=1, verbose=-1, num_threads=8)
PAIR_ROUNDS = int(os.environ.get("PAIR_ROUNDS", 350))
# REL_FEATS=1 appends features.relative (string features relative to the
# entity's other candidates). predict.py infers it from the model's width.
REL = os.environ.get("REL_FEATS", "0") == "1"
PAIR_NAMES = (features.NAMES + strfeatures.NAMES
              + (features.REL_NAMES if REL else []))
CHAN_COLS = slice(0, len(blocking.CHANNELS))      # features.build puts chan first
IS_S3_COL = features.NAMES.index("is_s3")


def cap_candidates(q, c, chan, cap=CAND_CAP):
    """Keep the best `cap` candidates per entity by max channel score."""
    best = chan.max(1)
    starts = np.flatnonzero(np.r_[True, q[1:] != q[:-1]])
    ends = np.r_[starts[1:], len(q)]
    keep = []
    for s, e in zip(starts, ends):
        if e - s <= cap:
            keep.append(np.arange(s, e))
        else:
            sel = np.argpartition(best[s:e], -cap)[-cap:]
            keep.append(s + np.sort(sel))
    k = np.concatenate(keep)
    return q[k], c[k], chan[k]


def train_pair_model(X, y, w=None):
    return lgb.train(PAIR_PARAMS, lgb.Dataset(X, y, weight=w,
                                              feature_name=PAIR_NAMES),
                     num_boost_round=PAIR_ROUNDS)


def head_features(q, p, X, text):
    """Singleton-head features for entities 0..len(text)-1, from their pairs'
    calibrated scores p and pair feature matrix X (q sorted by entity)."""
    Xe, names = singleton.entity_features(q, p, X[:, CHAN_COLS], X[:, IS_S3_COL] > 0.5,
                                          len(text), chan_names=blocking.CHANNELS)
    return singleton.with_text(Xe, text), names


def zero_prob_product(p, starts, ends, n_ent, q):
    """The product rule prod(1 - p) per entity; 1.0 for an entity with no pairs."""
    pz = np.ones(n_ent)
    for s, e in zip(starts, ends):
        pz[q[s]] = float(np.prod(1.0 - p[s:e]))
    return pz


def run_block(names, addrs, corpus, tag):
    t0 = time.time()
    q, c, chan = blocking.generate(None, None, names, addrs, corpus, verbose=False)
    with stage("cap_candidates", n=len(q), unit="pairs"):
        order = np.argsort(q, kind="stable")
        q, c, chan = q[order], c[order], chan[order]
        q, c, chan = cap_candidates(q, c, chan)
    print(f"  {tag}: {len(q):,} pairs ({len(q)/max(len(names),1):.1f}/entity) "
          f"in {time.time()-t0:.0f}s")
    return q, c, chan


def prepare_country(country, n_tr, n_va, gtm, rng):
    """Blocking + features for one country. Frees the corpus before returning,
    so peak memory is one shard regardless of how many countries we train on."""
    PROF.begin(country)
    with stage("load_shard"):
        s1_tab, corpus_tab = blocking.load_shard(ROOT, "train", country)
    s1_ids = s1_tab.column("entity_id").to_pylist()
    c_ids = np.array(corpus_tab.column("entity_id").to_pylist(), dtype=object)
    print(f"\n{country}: {len(s1_ids):,} S1, {len(c_ids):,} corpus")

    t0 = time.time()
    corpus = blocking.build_corpus(corpus_tab, verbose=False,
                                   string_features=True)
    del corpus_tab
    print(f"  indexed in {time.time()-t0:.0f}s")

    n_va_arg = n_va
    if n_va <= 0:
        n_va = max(len(s1_ids) - n_tr, 1)
    pick = rng.choice(len(s1_ids), min(n_tr + n_va, len(s1_ids)), replace=False)
    split = int(len(pick) * n_tr / (n_tr + n_va))
    # Over the whole country's S1, not the sample: counted within a 15k sample
    # a chain's other branches are mostly missing, so train and validation
    # (and predict.py) would each see a different feature.
    with stage("name_dup", n=len(s1_ids), unit="records"):
        s1_norm = [norm(x) for x in s1_tab.column("business_name").to_pylist()]
        dup = features.name_dup_counts(s1_norm)
    with stage("idf lut"):
        idf_lut = {t: float(corpus["index"]["idf"][i]) for i, t in
                   enumerate(corpus["index"]["vocab"].to_pylist())
                   if corpus["index"]["idf"][i] > 0}
        s_idf = singleton.idf_from_index(corpus["index"])

    hn = hard_negatives.config()
    tr_idx = pick[:split]
    if hn:
        # Every candidate label is already certified by ground truth; what the
        # owners add is WHY a negative is one (hard_negatives.py).
        with stage("hard negatives: owners + neighbours", n=len(pick),
                   unit="entities"):
            owner = hard_negatives.record_owner(s1_ids, c_ids, gtm)
            nid = hard_negatives.name_ids(s1_norm)
            s1_index = blocking.build_index(s1_norm)
            nbr = np.full((len(s1_ids), hn["k"]), -1, np.int64)
            nbr[pick] = hard_negatives.neighbours(s1_index, s1_norm, pick, hn["k"])
            if hn["add"]:
                # Nearest neighbours of the training entities, never one already
                # sampled: a validation entity must not also be trained on.
                extra = np.setdiff1d(nbr[tr_idx, 0], np.r_[pick, -1])
                nbr[extra] = hard_negatives.neighbours(s1_index, s1_norm, extra,
                                                       hn["k"])
                tr_idx = np.r_[tr_idx, extra]
                print(f"  hard negatives: +{len(extra):,} nearest-neighbour "
                      f"training entities"
                      + (" (n_val = 0 leaves none unsampled; pass n_val > 0)"
                         if n_va_arg <= 0 else ""))
            del s1_index

    def prep(idx, tag, n_orig=None):
        with stage("normalise S1", n=len(idx), unit="records"):
            sub = s1_tab.take(idx)
            names, addrs = blocking.normalise(sub)
        ids = [s1_ids[i] for i in idx]
        q, c, chan = run_block(names, addrs, corpus, f"{country} {tag}")
        with stage("labels", n=len(q), unit="pairs"):
            truth = [set((gtm.get(i) or "").split(",")) - {""} for i in ids]
            cand_ids = c_ids[c]
            y = np.fromiter((cand_ids[j] in truth[q[j]] for j in range(len(q))),
                            np.int8, len(q))
        with stage("candidate ids", n=len(q), unit="pairs"):
            is_s3 = np.fromiter((s.startswith("S3-") for s in cand_ids), bool,
                                len(q))
        with stage("strfeatures.build", n=len(q), unit="pairs"):
            Xs = strfeatures.build(corpus["recs"], names, addrs, q, c, idf_lut)
            if REL:
                Xs = np.hstack([Xs, features.relative(q, Xs, strfeatures.NAMES)])
        with stage("features.build", n=len(q), unit="pairs"):
            X = np.hstack([features.build(q, chan, is_s3,
                                          dup[idx]), Xs])
        with stage("singleton text features", n=len(idx), unit="entities"):
            text = singleton.text_features(names, addrs, s_idf)
        hni = None
        if hn:
            cat = hard_negatives.categorise(y, idx[q], c, owner, nid, nbr)
            hit, tot = hard_negatives.coverage(idx, q, c, owner, nbr)
            hard_negatives.describe(y, cat, label=f"{tag} negatives")
            print(f"  {tag}: {hit:,} of {tot:,} nearest-neighbour gold records "
                  f"({hit/max(tot, 1):.1%}) are in the entity's candidates")
            # orig: pairs of the originally sampled entities, for the baseline
            hni = {"cat": cat, "orig": q < (len(idx) if n_orig is None else n_orig)}
        return X, y, q, cand_ids, truth, ids, text, hni

    out = (prep(tr_idx, "train", split), prep(pick[split:], "valid"))
    with stage("free corpus + gc"):
        del corpus, s1_tab, c_ids, idf_lut, s_idf
        gc.collect()
    PROF.end()
    return out


def main():
    if pop_flag(sys.argv):
        PROF.enable()
    countries = (sys.argv[1] if len(sys.argv) > 1 else "India").split(",")
    n_tr = int(sys.argv[2]) if len(sys.argv) > 2 else 15000
    n_va = int(sys.argv[3]) if len(sys.argv) > 3 else 8000
    rng = np.random.default_rng(0)
    grammar = strfeatures.load_grammar()
    print(f"corruption grammar: {strfeatures.GRAMMAR_PATH} "
          + (f"({len(grammar):,} entries, {grammar.sha256[:12]})"
             if grammar is not None else "absent, grammar features are 0"))

    with stage("load ground truth"):
        gt = pq.read_table(pathlib.Path(ROOT) / "train_ground_truth.parquet")
        gtm = dict(zip(gt.column("source1_entity_id").to_pylist(),
                       gt.column("matched_entity_ids").to_pylist()))
        del gt

    tr_parts, va_parts = [], []
    for ctry in countries:
        tr, va = prepare_country(ctry, n_tr, n_va, gtm, rng)
        tr_parts.append(tr)
        va_parts.append((ctry,) + va)

    hn = hard_negatives.config()
    Xtr = np.vstack([t[0] for t in tr_parts])
    ytr = np.concatenate([t[1] for t in tr_parts])
    # training entities renumbered across countries, as validation is below
    ent_off = np.cumsum([0] + [len(t[5]) for t in tr_parts])
    qtr = np.concatenate([t[2] + o for t, o in zip(tr_parts, ent_off)])
    text_tr = np.vstack([t[6] for t in tr_parts])
    ysing_tr = singleton.singleton_labels([s for t in tr_parts for s in t[4]])
    wtr = orig_tr = None
    if hn:
        cat_tr = np.concatenate([t[7]["cat"] for t in tr_parts])
        orig_tr = np.concatenate([t[7]["orig"] for t in tr_parts])
        wtr = hard_negatives.mix_weights(ytr, cat_tr, hn)
        mix = ("natural" if hn["mix"] is None
               else dict(zip(hard_negatives.CATS, hn["mix"].round(3))))
        print(f"\nhard negatives: mix {mix}"
              f", k = {hn['k']}, add neighbours = {hn['add']}")
        hard_negatives.describe(ytr, cat_tr, wtr, label="all training negatives")
        del cat_tr
    del tr_parts
    gc.collect()

    # Everything downstream treats validation entities [0, n_cal) as the
    # isotonic calibration half and [n_cal, N) as the held-out half. Entities
    # used to be numbered country by country, so with two countries the
    # calibration half was ALL of the first country and the scored half ALL of
    # the second: a "combined" India,US score was US-only, calibrated on India.
    # Split each country in half and order all first halves before all second
    # halves, so both halves carry every country. One country: unchanged order.
    first, second = [], []
    for ctry, X, y, q, cand, truth, ids, text, hni in va_parts:
        h = len(ids) // 2
        m = q < h
        cat = None if hni is None else np.asarray(hni["cat"])
        if cat is not None:
            assert len(cat) == len(y), "hni['cat'] is expected per pair"
        first.append((ctry, X[m], y[m], q[m], cand[m], truth[:h], ids[:h], text[:h],
                      None if cat is None else {"cat": cat[m]}))
        second.append((ctry, X[~m], y[~m], q[~m] - h, cand[~m], truth[h:], ids[h:],
                       text[h:], None if cat is None else {"cat": cat[~m]}))
    n_cal_assembled = sum(len(p[6]) for p in first)
    va_parts = first + second
    del first, second

    # Validation entities are renumbered so several countries can share one
    # evaluation pass without their entity indices colliding.
    Xva_l, yva_l, qva_l, cva_l, truth_va, ids_va, ctry_va = [], [], [], [], [], [], []
    text_va, cat_va = [], []
    offset = 0
    for ctry, X, y, q, cand, truth, ids, text, hni in va_parts:
        Xva_l.append(X); yva_l.append(y); qva_l.append(q + offset); cva_l.append(cand)
        text_va.append(text)
        if hni is not None:
            cat_va.append(hni["cat"])
        truth_va += truth; ids_va += ids; ctry_va += [ctry] * len(ids)
        offset += len(ids)
    Xva = np.vstack(Xva_l); yva = np.concatenate(yva_l)
    qva = np.concatenate(qva_l); cva = np.concatenate(cva_l)
    text_va = np.vstack(text_va)
    del Xva_l, yva_l, qva_l, cva_l, va_parts
    gc.collect()
    print(f"\ntraining on {', '.join(countries)}: {len(ytr):,} pairs, "
          f"{len(ids_va):,} validation entities")
    print(f"  positives: train {ytr.mean():.2%}, valid {yva.mean():.2%}")

    t0 = time.time()
    with stage("lgb.train", n=len(ytr), unit="pairs"):
        model = train_pair_model(Xtr, ytr, wtr)
    with stage("model.predict", n=len(yva), unit="pairs"):
        raw = model.predict(Xva)
    if hn:
        # The unchanged recipe on the same run: originally sampled entities,
        # unweighted. Only the pair model differs; validation is identical.
        with stage("lgb.train (baseline)", n=int(orig_tr.sum()), unit="pairs"):
            raw_base = train_pair_model(Xtr[orig_tr], ytr[orig_tr]).predict(Xva)
    print(f"model trained in {time.time()-t0:.0f}s")

    # ---- out-of-fold pair scores for the singleton head ----
    # Folds are over training ENTITIES: an entity's pairs all land in one fold,
    # so no fold model has seen any candidate of the entities it scores.
    t0 = time.time()
    n_tr_ent = len(text_tr)
    fold = rng.permutation(n_tr_ent) % OOF_K
    pair_fold = fold[qtr]
    raw_oof = np.empty(len(ytr))
    for f in range(OOF_K):
        m = pair_fold == f
        with stage("lgb.train (OOF)", n=int((~m).sum()), unit="pairs"):
            fm = train_pair_model(Xtr[~m], ytr[~m], None if wtr is None else wtr[~m])
        with stage("model.predict (OOF)", n=int(m.sum()), unit="pairs"):
            raw_oof[m] = fm.predict(Xtr[m])
        del fm
    # Calibrated like the real scores: the head reads probabilities. Isotonic
    # is fit on the OOF scores themselves, which are all out-of-sample.
    with stage("isotonic (OOF)", n=len(ytr), unit="pairs"):
        p_oof = (IsotonicRegression(out_of_bounds="clip").fit(raw_oof, ytr)
                 .predict(raw_oof))
    del raw_oof, pair_fold
    with stage("singleton head fit", n=n_tr_ent, unit="entities"):
        Xe_tr, e_names = head_features(qtr, p_oof, Xtr, text_tr)
        head = singleton.SingletonHead(feature_names=e_names).fit(Xe_tr, ysing_tr)
    del Xe_tr, p_oof
    print(f"singleton head: {OOF_K}-fold OOF over {n_tr_ent:,} training entities "
          f"({ysing_tr.mean():.2%} singletons) in {time.time()-t0:.0f}s")

    # Calibration matters more than ranking here: the stopping rule consumes
    # probabilities, so a good ranker that is badly calibrated stops in the
    # wrong place. Fit isotonic on the first half of validation ENTITIES and
    # score only the second half, so the reported numbers are out-of-sample.
    # The calibration half still takes part in conflict resolution below: its
    # entities compete for the same records in the real pipeline.
    n_cal = n_cal_assembled
    cal = qva < n_cal
    with stage("isotonic", n=len(raw), unit="pairs"):
        iso = IsotonicRegression(out_of_bounds="clip").fit(raw[cal], yva[cal])
        p = iso.predict(raw)

    if hn:
        dcol = features.NAMES.index("name_dup")
        chain = np.zeros(len(ids_va), bool)
        chain[qva[Xva[:, dcol] > 0]] = True
        cat_va = np.concatenate(cat_va)
        p_base = (IsotonicRegression(out_of_bounds="clip")
                  .fit(raw_base[cal], yva[cal]).predict(raw_base))
        print("\nhard negatives A/B (held-out half, adaptive-k, product P(n=0)):")
        hard_negatives.describe(yva, cat_va, label="validation negatives")
        f_b, p_b = hard_negatives.ab_row("baseline", p_base, qva, cva, truth_va,
                                         n_cal, cat_va, chain)
        f_h, p_h = hard_negatives.ab_row("hard negatives", p, qva, cva, truth_va,
                                         n_cal, cat_va, chain)
        print(f"  hard - baseline: macro F0.5 {f_h - f_b:+.4f}   "
              f"precision {p_h - p_b:+.4f}")
        del raw_base, p_base, cat_va

    # Validation entity groups (qva is sorted: per-country sorted, offsets increasing).
    starts = np.flatnonzero(np.r_[True, qva[1:] != qva[:-1]])
    ends = np.r_[starts[1:], len(qva)]

    # Head P(n=0) on validation, from the full model's scores as at test time.
    # Isotonic on the calibration half corrects the shift from OOF-model scores
    # (3/4 of the data) to full-model scores; the held-out half stays clean.
    with stage("singleton head calibrate+predict", n=len(ids_va), unit="entities"):
        ysing_va = singleton.singleton_labels(truth_va)
        Xe_va, _ = head_features(qva, p, Xva, text_va)
        head.calibrate(Xe_va[:n_cal], ysing_va[:n_cal])
        pz_head = head.predict_proba(Xe_va)
    with stage("product rule P(n=0)", n=len(p), unit="pairs"):
        pz_prod = zero_prob_product(p, starts, ends, len(ids_va), qva)
    del Xe_va

    print("\nfeature importance:")
    for n, g in sorted(zip(PAIR_NAMES, model.feature_importance("gain")),
                       key=lambda x: -x[1])[:10]:
        print(f"  {n:12s} {g:12,.0f}")

    with stage("valstate.pkl write"), open("valstate.pkl", "wb") as fh:
        pickle.dump({"p": p, "cand": cva, "q": qva, "truth": truth_va,
                     "ids": ids_va, "ctry": ctry_va, "p_zero_head": pz_head,
                     "n_cal": n_cal}, fh)
    print("saved valstate.pkl (decision-rule tuning needs no re-blocking)")

    # ---- decision layer ----
    # Scored on the held-out half only; see the calibration note above.
    ev = np.arange(n_cal, len(ids_va))
    per_entity = {}
    for s, e in zip(starts, ends):
        per_entity[qva[s]] = (p[s:e], cva[s:e])

    def run_policy(policy):
        """policy(sorted probs, entity index) -> k. -> per-ev-entity (F0.5, k)."""
        with stage("choose_k eval", n=len(ev), unit="entities"):
            sc = np.empty(len(ev))
            npred = np.zeros(len(ev))
            for j, i in enumerate(ev):
                probs, cand = per_entity.get(i, (np.empty(0), np.empty(0, object)))
                # stable, as decide() below and predict.py: isotonic scores
                # tie often, and diag_matcher.py must reproduce this decision
                order = np.argsort(-probs, kind="stable")
                probs, cand = probs[order], cand[order]
                k = policy(probs, i)
                pred = set(cand[:k])
                npred[j] = k
                sc[j] = f05(len(pred & truth_va[i]), len(truth_va[i]), k)
        return sc, npred

    def evaluate(policy, label):
        sc, npred = run_policy(policy)
        print(f"  {label:26s} macro F0.5 = {sc.mean():.4f}   "
              f"mean k = {npred.mean():.2f}")
        return sc.mean()

    n_true = np.array([len(truth_va[i]) for i in ev])
    print(f"\nvalidation (held-out half): {len(ev):,} entities, "
          f"mean true n = {n_true.mean():.2f}, singletons {(n_true==0).mean():.2%}")
    best_fixed = max(evaluate(lambda pr, i, k=k: min(k, len(pr)), f"fixed top-{k}")
                     for k in (1, 2, 3, 4))
    for thr in (0.3, 0.5, 0.7):
        evaluate(lambda pr, i, t=thr: int((pr > t).sum()), f"threshold {thr}")
    adaptive = evaluate(lambda pr, i: choose_k(pr, pz_prod[i]),
                        "adaptive-k (p > 0.8*F_k)")

    print(f"\n  adaptive - best_fixed = {adaptive - best_fixed:+.4f}")

    # Walk the entity groups once; `qva == i` per entity would be O(n_ent*n_pairs).
    hit = np.zeros(len(ids_va))
    for s, e in zip(starts, ends):
        hit[qva[s]] = len(truth_va[qva[s]] & set(cva[s:e]))
    print(f"  blocking ceiling      = {f05(hit[ev], n_true, hit[ev]).mean():.4f}")

    # ---- singleton head gate ----
    # The operating point is the head's own decision: an entity with candidates
    # is flagged when choose_k(probs, head P(n=0)) returns 0. Entities with no
    # candidates get [] under either rule, so they are left out of precision.
    sc_prod, k_prod = run_policy(lambda pr, i: choose_k(pr, pz_prod[i]))
    sc_head, k_head = run_policy(lambda pr, i: choose_k(pr, pz_head[i]))
    has = np.array([i in per_entity for i in ev])
    is_single = n_true == 0
    flag = has & (k_head == 0)
    flip = flag & (k_prod > 0)          # abstentions the product rule would not make
    prec = is_single[flag].mean() if flag.any() else float("nan")
    prec_flip = is_single[flip].mean() if flip.any() else float("nan")
    f_alt = sc_prod[has & ~is_single].mean()
    # Precision over all abstentions includes the ones the product rule already
    # makes, and ignores singletons the head stops abstaining on, so it alone
    # can pass a head that lowers the score. Require the measured gain too.
    use_head = bool(prec >= SINGLETON_GATE) and sc_head.mean() >= sc_prod.mean()
    print(f"\nsingleton head (held-out half, {has.sum():,} entities with candidates, "
          f"{is_single[has].sum():,} singletons):")
    print(f"  flagged (k=0)         = {flag.sum():,}   precision = {prec:.3f}   "
          f"singleton recall = {flag[is_single & has].mean():.3f}")
    print(f"  flipped vs product    = {flip.sum():,}   precision = {prec_flip:.3f}")
    print(f"  mean P(n=0) head {pz_head[ev].mean():.4f}   product "
          f"{pz_prod[ev].mean():.4f}   true rate {is_single.mean():.4f}")
    print(f"  break-even at measured f_alt {f_alt:.3f} = "
          f"{singleton.break_even_precision(f_alt):.3f}   gate = {SINGLETON_GATE}")
    print(f"  adaptive-k, product P(n=0)  macro F0.5 = {sc_prod.mean():.4f}")
    print(f"  adaptive-k, head P(n=0)     macro F0.5 = {sc_head.mean():.4f}   "
          f"({sc_head.mean() - sc_prod.mean():+.4f})")
    print(f"  gate (precision >= {SINGLETON_GATE} and head F0.5 >= product) "
          f"{'PASSED: head saved for predict.py --singleton on' if use_head else 'FAILED: product rule stays'}")
    print("  top head features:", ", ".join(n for n, _ in head.importance()[:6]))

    with stage("model.pkl write"), open("model.pkl", "wb") as fh:
        pickle.dump({"model": model, "iso": iso,
                     "singleton": head if use_head else None,
                     "singleton_precision": prec,
                     # predict.py warns when its grammar file differs
                     "grammar_sha256": grammar.sha256 if grammar is not None else None}, fh)
    print("saved model.pkl")

    # ---- disjointness (disjoint.py) ----
    # Same adaptive-k decision, but made jointly: every validation entity
    # (both halves) competes for records, and each record keeps one owner.
    # Record IDs are unique across countries, so one integer index covers all.
    _, rec = np.unique(cva, return_inverse=True)

    # P(n=0) is the gated choice: the head if it passed, else the product rule
    # (p_zero=None / computed from the scores actually decided on, as before).
    pz_use = pz_head if use_head else None

    def decide(prob, pz=None):
        if pz is None:
            with stage("product rule P(n=0)", n=len(prob), unit="pairs"):
                pz = zero_prob_product(prob, starts, ends, len(ids_va), qva)
        with stage("choose_k", n=len(prob), unit="pairs"):
            acc = np.zeros(len(prob), bool)
            for s, e in zip(starts, ends):
                o = s + np.argsort(-prob[s:e], kind="stable")
                acc[o[:choose_k(prob[o], pz[qva[s]])]] = True
        return acc

    def resolve(prob, acc, **kw):
        with stage("disjoint resolve", n=len(prob), unit="pairs"):
            return disjoint.resolve_conflicts(qva, rec, prob, acc, p_zero=pz_use,
                                              **kw)

    def score_mask(acc, label):
        with stage("disjoint scoring", n=len(acc), unit="pairs"):
            is_ev = qva >= n_cal
            kk = np.bincount(qva[acc & is_ev], minlength=len(ids_va))[ev]
            cc = np.zeros(len(ids_va))
            for j in np.flatnonzero(acc & is_ev):
                cc[qva[j]] += cva[j] in truth_va[qva[j]]
            owners = np.bincount(rec[acc])
            shared = (owners >= 2).sum() / max((owners >= 1).sum(), 1)
            f = f05(cc[ev], n_true, kk).mean()
            print(f"  {label:26s} macro F0.5 = {f:.4f}   mean k = {kk.mean():.2f}   "
                  f"records shared = {shared:.2%}")
            return f

    print(f"\ndisjointness (one owner per record, P(n=0) from "
          f"{'singleton head' if use_head else 'product rule'}):")
    acc = decide(p, pz_use)
    base = score_mask(acc, "adaptive-k, per entity")
    res = score_mask(resolve(p, acc), "+ resolve_conflicts")
    red = score_mask(resolve(p, acc, redecide=True), "+ resolve, redecide")
    with stage("disjoint sinkhorn", n=len(p), unit="pairs"):
        pn = disjoint.sinkhorn_normalise(qva, rec, p)
    sk = score_mask(resolve(pn, decide(pn, pz_use)), "sinkhorn + resolve")
    print(f"\n  resolve - base        = {res - base:+.4f}")
    print(f"  redecide - base       = {red - base:+.4f}")
    print(f"  sinkhorn+resolve - base = {sk - base:+.4f}")
    PROF.report()


if __name__ == "__main__":
    main()
