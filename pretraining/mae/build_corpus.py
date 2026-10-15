"""Build the MAE pre-training corpus with composition-matched train/val/test splits.

Why this exists
---------------
The dataset's own `tier` field cannot give matched splits. Measured across the usable
sessions, tier:train is 46.9% Clip / 26.4% Monet2 / 23.5% Trippy while tier:validation
is 79.6% Clip / 12.5% Monet2 / 1.0% Trippy -- so the previous run selected
`best_encoder.pt` on a distribution it never trained on. In the original 32 sessions
validation contained ZERO Monet2 and Trippy.

So we build our own split instead:

  * eligible pool = `tier == "train"` ONLY. This is the single rule that keeps the
    downstream evaluation tiers (final_test_main / final_test_bonus / live_test_*)
    out of pre-training. Widening it would mean pre-training on our own benchmark.
  * keep stim_type in {Clip, Monet2, Trippy}. stimulus.Frame is dropped: its trials
    are one frame long, so a 2-s window is 60 unrelated stills, not video.
  * a window is assigned to the condition it starts in, and windows whose full 2-s
    span cross a condition boundary are DROPPED, so no clip lands in two splits.
  * conditions are allocated to splits GLOBALLY, so a condition shown in several
    sessions always lands in the same split -- otherwise the same movie clip would
    appear in both train and validation.

Balancing by duration, not by condition count
---------------------------------------------
The first version hashed condition_hash uniformly. That splits the *number of
conditions* 90/5/5, which is NOT the same as splitting *duration* 90/5/5: Monet2 and
Trippy are long continuous blocks while natural Clips are ~10 s each, so 56 Trippy
conditions (5% of them) produced 38% of validation windows and the composition check
failed. Instead we allocate greedily per stim_type against a target WINDOW count, so
every split ends up with the same stimulus mix by construction.

Output (next to this script)
------
  corpus_scan.json    raw per-session trial runs (cache; the slow part, ~4 min)
  corpus_A.json       Clip + Monet2 + Trippy
  corpus_B.json       Clip only
  corpus_report.json  hours / conditions / composition per split, both corpora

Runs are stored as [first_window_index, count] pairs rather than exploded index
lists -- windows of one trial are contiguous in `_valid_screen_times`, so this is
exact and ~1000x smaller than writing 40M integers.

Re-running with corpus_scan.json present skips the scan, so the split logic can
be iterated on in seconds.
"""
import glob
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
from omegaconf import OmegaConf

from experanto.datasets import ChunkDataset

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
# Same default as paths.data_dir in configs/paths/default.yaml.
DATA_DIR = os.environ.get("OMNIMOUSE_DATA_DIR", os.path.join(REPO, "data"))
EXCLUDE_SESSIONS = ("26872", "27204")          # the Sensorium 2022 final_test scans
ELIGIBLE_TIER = "train"                        # never widen this
KEEP_STIM = ("stimulus.Clip", "stimulus.Monet2", "stimulus.Trippy")
CLIP_ONLY = ("stimulus.Clip",)
CHUNK, RATE = 60, 30                           # 60 frames @ 30 Hz = 2.0 s
WINDOW_S = CHUNK / RATE
TARGET = {"train": 0.90, "validation": 0.05, "test": 0.05}
SCAN_CACHE = os.path.join(HERE, "corpus_scan.json")


def vcfg():
    return OmegaConf.create({"screen": {
        "keep_nans": False, "sampling_rate": RATE, "chunk_size": CHUNK,
        "valid_condition": [{"tier": ELIGIBLE_TIER}], "offset": 0,
        "sample_stride": 1, "include_blanks": False,
        "transforms": {"normalization": {"means": [113.0], "stds": [59.0]}},
        "interpolation": {"rescale": False}}})


def scan_session(path):
    """-> list of [first_idx, count, stim_type, condition_hash] contiguous runs.

    Windows belonging to one trial are contiguous in `_valid_screen_times`, so each
    trial becomes a single run.
    """
    ds = ChunkDataset(path, None, None, modality_config=vcfg(), seed=0,
                      out_keys=["screen"], normalize_timestamps=False)
    n = len(ds)
    if n == 0:
        return [], 0
    starts = np.asarray(ds._start_times, dtype=float)
    ends = np.asarray(ds._end_times, dtype=float)
    trials = ds._trials
    times = np.asarray(ds._valid_screen_times, dtype=float)

    ti = np.searchsorted(starts, times, side="right") - 1
    ok = ti >= 0
    # The whole 2-s window must sit inside one trial. _end_times[i] is the START of
    # trial i+1, so this also stops a window running into a neighbouring condition
    # -- or into a blank, since blanks are trials too.
    safe = np.clip(ti, 0, len(ends) - 1)
    ok &= ok & ((times + WINDOW_S) <= ends[safe])

    idx = np.nonzero(ok)[0]
    if idx.size == 0:
        return [], n
    tri = ti[idx]
    # contiguous runs of (consecutive window index AND same trial)
    brk = np.nonzero((np.diff(idx) != 1) | (np.diff(tri) != 0))[0] + 1
    bounds = np.concatenate(([0], brk, [idx.size]))

    meta, runs = {}, []
    for a, b in zip(bounds[:-1], bounds[1:]):
        t = int(tri[a])
        if t not in meta:
            tr = trials[t]
            meta[t] = (str(tr.get_meta("stim_type")),
                       str(tr.get_meta("condition_hash")),
                       str(tr.get_meta("tier")))
        stim, cond, tier = meta[t]
        if tier != ELIGIBLE_TIER:          # belt and braces; the config already filters
            continue
        runs.append([int(idx[a]), int(b - a), stim, cond])

    if runs and {r[2] for r in runs} == {"None"}:
        raise RuntimeError(f"stim_type unreadable for {path} - get_meta returned None")
    return runs, n


def allocate(scans, keep_stim):
    """Assign each condition to a split, balancing WINDOW counts within each stim_type.

    Greedy largest-first: repeatedly give the biggest remaining condition to whichever
    split is furthest below its target. Conditions are keyed globally, so the same
    condition always lands in the same split across sessions.
    """
    per_type = defaultdict(Counter)                      # stim -> cond -> windows
    for rows in scans.values():
        for _, cnt, stim, cond in rows:
            if stim in keep_stim:
                per_type[stim][cond] += cnt

    assign = {}
    for stim, conds in per_type.items():
        total = sum(conds.values())
        want = {sp: TARGET[sp] * total for sp in TARGET}
        got = {sp: 0 for sp in TARGET}
        # deterministic: largest first, condition hash breaks ties
        for cond, cnt in sorted(conds.items(), key=lambda kv: (-kv[1], kv[0])):
            sp = max(TARGET, key=lambda s: want[s] - got[s])
            assign[(stim, cond)] = sp
            got[sp] += cnt
    return assign


def build(scans, keep_stim, assign):
    corpus = {}
    frames = defaultdict(Counter)
    conds = defaultdict(lambda: defaultdict(set))
    for sess, rows in scans.items():
        per = defaultdict(list)
        for first, cnt, stim, cond in rows:
            if stim not in keep_stim:
                continue
            sp = assign[(stim, cond)]
            per[sp].append([first, cnt])
            frames[sp][stim] += cnt
            conds[sp][stim].add(cond)
        if per:
            corpus[sess] = {k: sorted(v) for k, v in per.items()}

    report = {}
    for sp in ("train", "validation", "test"):
        tot = sum(frames[sp].values())
        report[sp] = {
            "windows": tot,
            "independent_clips": tot // CHUNK,
            "hours_of_window_starts": tot / RATE / 3600,
            "composition_pct": {k: round(100 * v / tot, 3) for k, v in frames[sp].items()} if tot else {},
            "unique_conditions": {k: len(v) for k, v in conds[sp].items()},
            "n_sessions": sum(1 for s in corpus if sp in corpus[s]),
        }
    return corpus, report


def check_matched(report, tol_pp=1.0):
    """The whole point of the rebuild: every split must have the same stimulus mix."""
    base = report["train"]["composition_pct"]
    problems = []
    for sp in ("validation", "test"):
        comp = report[sp]["composition_pct"]
        for stim in set(base) | set(comp):
            d = abs(base.get(stim, 0.0) - comp.get(stim, 0.0))
            if d > tol_pp:
                problems.append(f"{sp}/{stim}: {comp.get(stim,0):.2f}% vs "
                                f"train {base.get(stim,0):.2f}% (Δ{d:.2f}pp)")
    return problems


def load_or_scan():
    if os.path.exists(SCAN_CACHE):
        print(f"[scan] reusing cache {SCAN_CACHE}", flush=True)
        d = json.load(open(SCAN_CACHE))
        return d["scans"], d["skipped"]

    sessions = [p for p in sorted(glob.glob(os.path.join(DATA_DIR, "*")))
                if not any(e in p for e in EXCLUDE_SESSIONS)]
    print(f"[scan] {len(sessions)} candidate sessions", flush=True)
    scans, skipped = {}, []
    for i, p in enumerate(sessions, 1):
        name = os.path.basename(p)
        try:
            rows, _ = scan_session(p)
        except Exception as e:
            skipped.append([name, f"{type(e).__name__}: {e}"])
            continue
        if rows:
            scans[name] = rows
        if i % 50 == 0 or i == len(sessions):
            print(f"[scan] {i}/{len(sessions)}  kept={len(scans)}  skipped={len(skipped)}", flush=True)
    json.dump({"scans": scans, "skipped": skipped}, open(SCAN_CACHE, "w"))
    print(f"[scan] cached -> {SCAN_CACHE}", flush=True)
    return scans, skipped


def main():
    scans, skipped = load_or_scan()

    # leakage guard: the two Sensorium scans must never appear
    bad = [s for s in scans if any(e in s for e in EXCLUDE_SESSIONS)]
    assert not bad, f"Sensorium test scans leaked into the corpus: {bad}"

    out = {}
    for tag, keep in (("A", KEEP_STIM), ("B", CLIP_ONLY)):
        assign = allocate(scans, keep)
        corpus, report = build(scans, keep, assign)
        problems = check_matched(report)
        out[tag] = {"report": report, "n_sessions": len(corpus),
                    "keep_stim": list(keep), "composition_problems": problems}
        with open(os.path.join(HERE, f"corpus_{tag}.json"), "w") as f:
            json.dump({"meta": {"eligible_tier": ELIGIBLE_TIER, "keep_stim": list(keep),
                                "chunk": CHUNK, "rate": RATE, "target": TARGET,
                                "excluded": list(EXCLUDE_SESSIONS),
                                "format": "sessions[name][split] = [[first_window_idx, count], ...]"},
                       "sessions": corpus}, f)
        print(f"\n=== corpus {tag}  ({len(corpus)} sessions, stim={keep})", flush=True)
        for sp in ("train", "validation", "test"):
            r = report[sp]
            comp = "  ".join(f"{k.replace('stimulus.','')}={v:.1f}%"
                             for k, v in sorted(r["composition_pct"].items()))
            print(f"  {sp:11s} {r['windows']:>11,} windows  {r['independent_clips']:>8,} clips  "
                  f"{r['hours_of_window_starts']:6.1f} h  {comp}", flush=True)
        print(f"  composition check: "
              f"{'PASS' if not problems else 'FAIL -> ' + '; '.join(problems)}", flush=True)

    out["skipped"] = skipped
    with open(os.path.join(HERE, "corpus_report.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nskipped {len(skipped)} sessions", flush=True)
    bad_tags = [t for t in ("A", "B") if out[t]["composition_problems"]]
    print("\nRESULT:", "FAIL — splits not matched for " + ",".join(bad_tags)
          if bad_tags else "PASS — all splits matched")
    return 1 if bad_tags else 0


if __name__ == "__main__":
    sys.exit(main())
