"""Run the full pipeline over the test set and emit the two submission files.

Format rules enforced here (from the problem statement):
  * exactly one row per test S1 entity, including entities with no candidates
  * matched_entity_ids empty for predicted singletons
  * no duplicate IDs within a list
  * only S2/S3 IDs that exist in the test set
  * matching_results MUST be a subset of candidate_pairs

Country is a verified-safe hard block, so each country is an independent job and
the pipeline has natural resume points. Entities are processed in batches to
bound peak memory.

Disjointness (disjoint.py): every S2/S3 record belongs to at most one S1
entity. Two entities in different batches can claim the same record, because
every batch in a country searches the same corpus, so scored pairs are buffered
for the whole country and the decision is made once all its batches are done.
  --disjoint off       per-entity choose_k only (the previous behaviour)
  --disjoint resolve   choose_k, then resolve_conflicts
  --disjoint sinkhorn  sinkhorn_normalise, choose_k, then resolve_conflicts
--recall R passes the measured blocking recall to choose_k and
resolve_conflicts alike, so both use the same n_hat (default 1.0).
resolve_conflicts only drops pairs, so matches stay a subset of candidates.

--profile prints wall clock, throughput and peak RSS per pipeline stage, per
country and overall (profiling.py). Default off, and near-free when off.

--singleton on takes P(n=0) from the singleton head (singleton.py) that
train_eval.py saved in model.pkl, in place of prod(1 - p), for choose_k and
resolve_conflicts alike. train_eval.py saves a head only when it passes the
precision gate; without one this falls back to the product rule and says so.
The head reads an entity's own candidates only, so it runs per batch. Default
off, which is the previous behaviour.

Resume (resume.py): each finished country's rows go to
output/partial_<country>.tsv with a sidecar JSON, and are freed. A rerun skips
every country whose partial is complete for the same S1 row count and the same
config (flags, blocking knobs, model.pkl, pipeline code, input parquet), then
concatenates the partials into the two final files, byte-identical to an
uninterrupted run. --resume is the
default; --fresh deletes the partials first. MEM_BUDGET_MB is not part of the
config, since it does not change the output, so a run that died of memory
pressure can resume with a lower budget.

--calibrate France (comma list; default off) calibrates each named country so
its predicted cardinality matches the other countries': the P(n=0) that
choose_k and resolve_conflicts use is scaled by s, and choose_k's stopping bar
by t, per country, fitted on that country's own unlabelled test predictions
(calibrate.py says why that is defensible). The reference is the pooled
choose_k output, before resolve_conflicts, of every other country, or of
--calibrate-ref US,India. Reference countries run first and are never touched;
a calibrated country's config records the target and the reference histogram,
so a resumed run never mixes calibrated and uncalibrated partials, and the
reference countries' partials stay reusable across --calibrate on/off.
Every country prints its cardinality; calibrated ones print before/after.

Usage: python predict.py [--limit N] [--disjoint off|resolve|sinkhorn]
                         [--recall R] [--singleton on|off] [--resume | --fresh]
                         [--calibrate C[,C...]] [--calibrate-ref C[,C...]]
                         [--profile]
"""
import os
import pathlib
import pickle
import sys
import time

import numpy as np
import pyarrow.compute as pc
import pyarrow.dataset as ds

import blocking
import calibrate
import disjoint
import features
import resume
import singleton
import strfeatures
from metric import choose_k
from profiling import PROF, pop_flag, stage
from textnorm import norm
from train_eval import cap_candidates, head_features

ROOT = "data/parquet"
OUT = pathlib.Path("output")
BATCH = 20000
DISJOINT_MODES = ("off", "resolve", "sinkhorn")


def data_stamp(country, split="test"):
    """Path, size and mtime of every parquet file of this country, so rerunning
    prepare_data.py invalidates the country's partial."""
    stamp = []
    for src in (1, 2, 3):
        d = ds.dataset(f"{ROOT}/{split}_source{src}", format="parquet",
                       partitioning="hive")
        for frag in d.get_fragments(filter=pc.field("country") == country):
            st = pathlib.Path(frag.path).stat()
            stamp.append([frag.path, st.st_size, st.st_mtime_ns])
    return sorted(stamp)


def s1_counts(split="test"):
    """S1 rows per country, read from the country column alone."""
    d = ds.dataset(f"{ROOT}/{split}_source1", format="parquet", partitioning="hive")
    vc = pc.value_counts(d.to_table(columns=["country"]).column("country"))
    return {v["values"].as_py(): v["counts"].as_py() for v in vc}


def decide(q, c, p, mode, recall=1.0, p_zero=None, n_entities=None,
           calib_ref=None, ids=None):
    """Accepted-pair mask for one country's pairs (q sorted by entity).

    p_zero: optional P(n=0) per entity, indexed by q. None is the product rule
    prod(1 - p), computed after Sinkhorn when that runs, as before.
    calib_ref: a reference cardinality histogram to calibrate this country to
    (calibrate.py); None leaves the decision exactly as before.
    ids: optional record id per c, so the histogram counts distinct ids per
    entity, as the emitted rows do.

    Returns (accepted mask, p, pre-resolve cardinality histogram over
    n_entities, calibration info or None)."""
    n_entities = int(q.max()) + 1 if n_entities is None else n_entities
    if mode == "sinkhorn":
        with stage("disjoint sinkhorn", n=len(q), unit="pairs"):
            p = disjoint.sinkhorn_normalise(q, c, p)
    info = None
    if calib_ref is not None:
        with stage("calibrate fit", n=len(q), unit="pairs"):
            pz0 = (calibrate.product_p_zero(q, p, n_entities) if p_zero is None
                   else np.asarray(p_zero, float))
            s, t, before, warn = calibrate.fit(q, p, pz0, recall, calib_ref,
                                               n_entities)
        with stage("choose_k", n=len(q), unit="pairs"):
            acc, p_zero = calibrate.apply(q, p, pz0, s, t, recall)
        info = {"s": s, "t": t, "before": before, "warn": warn}
    else:
        with stage("choose_k", n=len(q), unit="pairs"):
            acc = np.zeros(len(q), bool)
            starts = np.flatnonzero(np.r_[True, q[1:] != q[:-1]])
            ends = np.r_[starts[1:], len(q)]
            for s, e in zip(starts, ends):
                o = s + np.argsort(-p[s:e], kind="stable")
                pz = float(np.prod(1.0 - p[o])) if p_zero is None else float(p_zero[q[s]])
                acc[o[:choose_k(p[o], pz, recall)]] = True
    hist = cardinality(q, c, acc, n_entities, ids)
    if mode != "off":
        with stage("disjoint resolve", n=len(q), unit="pairs"):
            acc = disjoint.resolve_conflicts(q, c, p, acc, recall=recall,
                                             p_zero=p_zero)
    return acc, p, hist, info


def cardinality(q, c, acc, n_entities, ids=None):
    """Histogram of accepted DISTINCT record ids per entity, as the matching
    rows are written (they drop duplicate ids)."""
    keep = np.flatnonzero(acc)
    if ids is None:
        return calibrate.k_hist(np.bincount(q[keep], minlength=n_entities))
    k = np.zeros(n_entities, np.int64)
    for e, rid in set(zip(q[keep].tolist(), ids[c[keep]].tolist())):
        k[e] += 1
    return calibrate.k_hist(k)


def country_list(flag):
    """--calibrate / --calibrate-ref value -> list of country names."""
    if flag not in sys.argv:
        return []
    i = sys.argv.index(flag)
    if i + 1 >= len(sys.argv) or sys.argv[i + 1].startswith("--"):
        sys.exit(f"{flag} needs a comma-separated list of countries")
    return [x.strip() for x in sys.argv[i + 1].split(",") if x.strip()]


def main():
    if pop_flag(sys.argv):
        PROF.enable()
    limit = None
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])
    mode = "off"
    if "--disjoint" in sys.argv:
        mode = sys.argv[sys.argv.index("--disjoint") + 1]
        if mode not in DISJOINT_MODES:
            sys.exit(f"--disjoint must be one of {', '.join(DISJOINT_MODES)}")
    recall = 1.0
    if "--recall" in sys.argv:
        recall = float(sys.argv[sys.argv.index("--recall") + 1])
        if not 0.0 < recall <= 1.0:
            sys.exit("--recall must be in (0, 1]")
    use_singleton = False
    if "--singleton" in sys.argv:
        val = sys.argv[sys.argv.index("--singleton") + 1]
        if val not in ("on", "off"):
            sys.exit("--singleton must be on or off")
        use_singleton = val == "on"
    if "--resume" in sys.argv and "--fresh" in sys.argv:
        sys.exit("--resume and --fresh are mutually exclusive")
    fresh = "--fresh" in sys.argv
    calib = country_list("--calibrate")
    calib_ref = country_list("--calibrate-ref")
    if calib_ref and not calib:
        sys.exit("--calibrate-ref needs --calibrate")
    if set(calib) & set(calib_ref):
        sys.exit("a country cannot be both calibrated and a reference")

    # model.pkl is written by train_eval.py in this same repo -- a local build
    # artifact, never a downloaded or user-supplied file.
    with stage("load model"), open("model.pkl", "rb") as f:
        bundle = pickle.load(f)
    model, iso = bundle["model"], bundle["iso"]
    head = bundle.get("singleton") if use_singleton else None
    if use_singleton and head is None:
        prec = bundle.get("singleton_precision")
        print("WARNING: --singleton on, but model.pkl has no singleton head "
              + ("(it predates the head; rerun train_eval.py)" if prec is None else
                 f"(it failed the precision gate at {prec:.3f})")
              + ". Using the product rule.", flush=True)

    n_feat = len(features.NAMES) + len(strfeatures.NAMES)
    n_model = model.num_feature() if hasattr(model, "num_feature") else n_feat
    # A model trained with REL_FEATS=1 is exactly REL_NAMES wider.
    rel = n_model == n_feat + len(features.REL_NAMES)
    if rel:
        n_feat = n_model
    if n_model != n_feat:
        sys.exit(f"model.pkl was trained on {n_model} features but this code "
                 f"builds {n_feat}; retrain it with train_eval.py")
    grammar = strfeatures.load_grammar()
    g_sha = grammar.sha256 if grammar is not None else None
    # Fatal like the feature count: a mismatched grammar scores every pair with
    # features the model never saw. ALLOW_GRAMMAR_MISMATCH=1 runs anyway.
    if bundle.get("grammar_sha256", g_sha) != g_sha:
        msg = (f"model.pkl was trained with corruption grammar "
               f"{str(bundle.get('grammar_sha256'))[:12]} but {strfeatures.GRAMMAR_PATH} "
               f"is {'absent' if g_sha is None else g_sha[:12]}. Retrain or restore "
               f"the file (ALLOW_GRAMMAR_MISMATCH=1 to run anyway).")
        if os.environ.get("ALLOW_GRAMMAR_MISMATCH") != "1":
            sys.exit(msg)
        print("WARNING: " + msg, flush=True)

    # Everything that changes a country's rows. A partial written under a
    # different config is recomputed, never mixed into this run's output.
    # The code hash covers every pipeline module, including constants such as
    # CAND_CAP that have no environment override.
    code = [sys.modules[m].__file__ for m in (
        __name__, "blocking", "calibrate", "disjoint", "features", "metric", "singleton",
        "strfeatures", "textnorm", "train_eval")]
    config = {"limit": limit or None, "disjoint": mode, "recall": recall,
              "singleton": head is not None, "batch": BATCH,
              "model_sha256": resume.file_sha256("model.pkl"),
              "grammar_sha256": g_sha,
              "code_sha256": {pathlib.Path(f).name: resume.file_sha256(f)
                              for f in code},
              "blocking": {k: getattr(blocking, k) for k in (
                  "DF_CAP", "ADDR_DF_CAP", "C3_DF_CAP", "C3_DF_FLOOR",
                  "TOP_K", "ADDR_TOP_K")}}

    OUT.mkdir(exist_ok=True)
    if fresh:
        resume.clear(OUT)
    t_start = time.time()
    n_s1 = s1_counts()
    expected = {}   # country -> (rows, config)
    missing = sorted((set(calib) | set(calib_ref)) - set(n_s1))
    if missing:
        sys.exit(f"--calibrate/--calibrate-ref: no test S1 rows for {', '.join(missing)}")
    if calib and not calib_ref:
        calib_ref = [c for c in sorted(n_s1) if c not in calib]
    if calib and not calib_ref:
        sys.exit("--calibrate leaves no reference country")
    if calib:
        print(f"calibrating {', '.join(calib)} to {', '.join(calib_ref)}", flush=True)
    # Reference countries first, so their cardinality exists before any
    # calibrated country is decided. Output order is unchanged (sorted).
    run_order = sorted(n_s1, key=lambda c: (c in calib, c))

    for country in run_order:
        n_exp = n_s1[country] if not limit else min(n_s1[country], limit)
        cfg = dict(config, data=data_stamp(country))
        ref_hist = None
        if country in calib:
            # Only a calibrated country's config changes, so reference
            # countries' partials are reused across --calibrate on/off. The
            # reference histogram is part of the config: if a reference
            # country is recomputed, so is every country calibrated to it.
            ref_hist = calibrate.pool([resume.cardinality(OUT, r) for r in calib_ref])
            cfg["calibrate"] = {"reference": calib_ref, "ref_hist": ref_hist}
        expected[country] = (n_exp, cfg)
        why = resume.check(OUT, country, n_exp, cfg)
        if why is None:
            print(f"\n{country}: partial complete ({n_exp:,} rows), skipping",
                  flush=True)
            continue
        if why != "no partial":
            print(f"\n{country}: recomputing, {why}", flush=True)
        rows_match, rows_cand = [], []
        PROF.begin(country)
        with stage("load_shard"):
            s1_tab, corpus_tab = blocking.load_shard(ROOT, "test", country)
        s1_ids = s1_tab.column("entity_id").to_pylist()
        c_ids = np.array(corpus_tab.column("entity_id").to_pylist(), dtype=object)
        if limit:
            s1_ids = s1_ids[:limit]
            s1_tab = s1_tab.slice(0, limit)
        print(f"\n{country}: {len(s1_ids):,} S1, {len(c_ids):,} corpus", flush=True)

        t0 = time.time()
        corpus = blocking.build_corpus(corpus_tab, verbose=False,
                                     string_features=True)
        print(f"  indexed in {time.time()-t0:.0f}s", flush=True)
        with stage("idf lut"):
            idf_lut = {t: float(corpus["index"]["idf"][i]) for i, t in
                       enumerate(corpus["index"]["vocab"].to_pylist())
                       if corpus["index"]["idf"][i] > 0}
            s_idf = (singleton.idf_from_index(corpus["index"]) if head is not None
                     else None)

        # name_dup counts over the whole country's S1, as the feature is defined
        # and as train_eval.py computes it; per batch it drops every duplicate
        # that falls in another batch.
        with stage("name_dup", n=len(s1_ids), unit="records"):
            dup = features.name_dup_counts(
                [norm(x) for x in s1_tab.column("business_name").to_pylist()])
        seen = 0
        # Scored pairs for the whole country; entity index is global (lo + q).
        all_q, all_c, all_p = [], [], []
        p_zero = np.ones(len(s1_ids)) if head is not None else None
        for lo in range(0, len(s1_ids), BATCH):
            hi = min(lo + BATCH, len(s1_ids))
            sub = s1_tab.slice(lo, hi - lo)
            with stage("normalise S1", n=hi - lo, unit="records"):
                names, addrs = blocking.normalise(sub)
            q, c, chan = blocking.generate(None, None, names, addrs, corpus,
                                           verbose=False)
            batch_ids = s1_ids[lo:hi]
            cand = {i: [] for i in range(len(batch_ids))}

            if len(q):
                with stage("cap_candidates", n=len(q), unit="pairs"):
                    order = np.argsort(q, kind="stable")
                    q, c, chan = q[order], c[order], chan[order]
                    q, c, chan = cap_candidates(q, c, chan)
                with stage("candidate ids", n=len(q), unit="pairs"):
                    cand_ids = c_ids[c]
                    is_s3 = np.fromiter((s.startswith("S3-") for s in cand_ids),
                                        bool, len(cand_ids))
                with stage("strfeatures.build", n=len(q), unit="pairs"):
                    Xs = strfeatures.build(corpus["recs"], names, addrs, q, c,
                                           idf_lut)
                    if rel:
                        Xs = np.hstack([Xs, features.relative(q, Xs,
                                                              strfeatures.NAMES)])
                with stage("features.build", n=len(q), unit="pairs"):
                    X = np.hstack([features.build(q, chan, is_s3,
                                                  dup[lo:hi]), Xs])
                with stage("model.predict", n=len(q), unit="pairs"):
                    raw = model.predict(X)
                with stage("isotonic", n=len(q), unit="pairs"):
                    p = iso.predict(raw)
                if head is not None:
                    with stage("singleton head", n=hi - lo, unit="entities"):
                        Xe, _ = head_features(q, p, X, singleton.text_features(
                            names, addrs, s_idf))
                        p_zero[lo:hi] = head.predict_proba(Xe)

                with stage("candidate lists", n=len(q), unit="pairs"):
                    starts = np.flatnonzero(np.r_[True, q[1:] != q[:-1]])
                    ends = np.r_[starts[1:], len(q)]
                    for s, e in zip(starts, ends):
                        o = s + np.argsort(-p[s:e])
                        # dict.fromkeys preserves order and removes duplicates
                        cand[int(q[s])] = list(dict.fromkeys(c_ids[c[o]]))
                    all_q.append(q.astype(np.int64) + lo)
                    all_c.append(c.astype(np.int64))
                    all_p.append(p)

            with stage("candidate rows", n=hi - lo, unit="entities"):
                for i, eid in enumerate(batch_ids):
                    rows_cand.append(f"{eid}\t{','.join(cand[i])}")
            seen += hi - lo
            print(f"    {seen:,}/{len(s1_ids):,}  ({time.time()-t0:.0f}s)", flush=True)

        pred = [[] for _ in s1_ids]
        hist = [len(s1_ids)]            # no pairs: every entity abstains
        info = None
        if all_q:
            q, c, p = (np.concatenate(all_q), np.concatenate(all_c),
                       np.concatenate(all_p))
            del all_q, all_c, all_p
            t1 = time.time()
            acc, p, hist, info = decide(q, c, p, mode, recall, p_zero,
                                        len(s1_ids), ref_hist, c_ids)
            with stage("match lists", n=len(q), unit="pairs"):
                # highest probability first within each entity, as before
                keep = np.flatnonzero(acc)
                keep = keep[np.lexsort((-p[keep], q[keep]))]
                for j in keep:
                    pred[q[j]].append(c_ids[c[j]])
            shared = np.bincount(c[acc]) if acc.any() else np.zeros(1, int)
            print(f"  decided ({mode}, recall {recall}, "
                  f"P(n=0) {'head' if head is not None else 'product'}) in {time.time()-t1:.0f}s, records with "
                  f"2+ owners: {(shared >= 2).sum():,}", flush=True)
        final = calibrate.k_hist([len(dict.fromkeys(x)) for x in pred])
        print(calibrate.fmt_header())
        if ref_hist is not None:
            print(calibrate.fmt_row("reference", ref_hist))
            if info is not None:
                print(calibrate.fmt_row("before calibration", info["before"]))
            print(calibrate.fmt_row("after calibration", hist))
            if info is not None:
                print(f"    P(n=0) scale s = {info['s']:.4g}, stopping-bar scale "
                      f"t = {info['t']:.4g}")
                for w in info["warn"]:
                    print(f"    WARNING: {w}")
        else:
            print(calibrate.fmt_row("choose_k", hist))
        if mode != "off":
            print(calibrate.fmt_row("after resolve", final))
        with stage("match rows", n=len(s1_ids), unit="entities"):
            for eid, ids in zip(s1_ids, pred):
                rows_match.append(f"{eid}\t{','.join(dict.fromkeys(ids))}")
        if len(rows_match) != n_exp:
            raise RuntimeError(f"{country}: {len(rows_match)} rows, expected {n_exp}")
        with stage("TSV write", n=len(rows_match), unit="rows"):
            resume.write(OUT, country, rows_match, rows_cand, cfg, hist)
        PROF.end()
        # Free this country before the next shard loads. Rebinding rather than
        # del, since some of these exist only when the country had candidates.
        rows_match = rows_cand = pred = s1_tab = corpus_tab = corpus = None
        hist = final = info = None
        c_ids = idf_lut = s_idf = dup = p_zero = cand = sub = None
        q = c = chan = p = raw = acc = keep = shared = order = None
        X = Xs = Xe = cand_ids = is_s3 = all_q = all_c = all_p = None

    expected = {c: expected[c] for c in sorted(expected)}   # output order
    hdr_m = "source1_entity_id\tmatched_entity_ids"
    hdr_c = "source1_entity_id\tcandidate_entity_ids"
    # newline="\n" is load-bearing on Windows. The default text mode rewrites
    # "\n" as "\r\n", and a scorer that splits on "\n" then sees every trailing
    # ID as "S3-123\r", which matches nothing in the test set and rejects the
    # whole submission. Silent, and it would cost a submission to discover.
    # The partials are written with newline="\n" and concatenated as bytes, so
    # nothing here can translate line endings.
    with stage("TSV concat partials"):
        n_rows, n_pred = resume.concat(OUT, expected,
                                       OUT / "matching_results.tsv",
                                       OUT / "candidate_pairs.tsv", hdr_m, hdr_c)
    print(f"\nwrote {n_rows:,} rows in {(time.time()-t_start)/60:.0f} min")
    print(f"  entities with >=1 match: {n_pred:,} ({n_pred/n_rows:.1%})")
    PROF.report()


if __name__ == "__main__":
    main()
