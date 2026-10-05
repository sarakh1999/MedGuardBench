"""
Generate unrolled patient-profile context for every row of the data splits.

Implements  C_{t+1} = C_t (+) phi_t(x, C_t)  offline with a teacher:

    C_0 = {}
    C_1 = C_0 + patient_constraints(x)      deterministic
    C_2 = C_1 + prescription(x)             deterministic
    C_3 = C_2 + drug_profile(x, C_2)        teacher
    C_4 = C_3 + interactions(x, C_3)        teacher, sees the drug profile
    C_5 = C_4 + dose_check(x, C_4)          teacher, sees profile + constraints

The teacher is BLIND: it never receives Is_Safe, Risk_Categories, or any
reasoning column. Blindness is enforced structurally (primitives.visible_row
and the prompt builders only read non-hidden fields) and checked again on
every prompt before it is sent.

Outputs
-------
  data/unroll_cache.jsonl           one row per Patient ID (resume unit)
  data/unrolled_splits/<split>.csv  the split CSV + Unrolled_Context column

The cache is the source of truth; the split CSVs are derived from it with
--split and can be regenerated at any time.

Usage (from repo root):
    export DEEPSEEK_API_KEY=...
    python context_unrolling/generate_unrolled_context.py --limit 10          # smoke test
    python context_unrolling/generate_unrolled_context.py --splits test       # one split
    python context_unrolling/generate_unrolled_context.py --workers 8         # everything
    python context_unrolling/generate_unrolled_context.py --split             # re-derive CSVs
    python context_unrolling/generate_unrolled_context.py --report
    python context_unrolling/generate_unrolled_context.py --retry-errors
    python context_unrolling/generate_unrolled_context.py --deterministic-only  # no API
"""

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import unroll_config as C
from primitives import (
    deterministic_context, visible_row, PRIMITIVE_BUILDERS,
    parse_primitive_response, TEACHER_SYSTEM_PROMPT,
)

csv.field_size_limit(min(sys.maxsize, 2147483647))

SPLIT_FILES = {
    "train": ["train.csv"],
    "val": ["val.csv", "validation.csv", "valid.csv", "dev.csv"],
    "test": ["test.csv"],
}


# ==============================================================================
# IO
# ==============================================================================

def find_split(folder, split):
    for name in SPLIT_FILES[split]:
        p = Path(folder) / name
        if p.exists():
            return p
    return None


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_cache(path):
    cache = {}
    if not Path(path).exists():
        return cache
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            pid = str(r.get("patient_id"))
            if pid:
                cache[pid] = r      # last write wins
    return cache


def append_cache(path, record):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())


def compact_cache(path):
    """Rewrite the cache so each Patient ID appears once (keeps the latest)."""
    cache = load_cache(path)
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for pid in sorted(cache, key=lambda x: (len(x), x)):
            f.write(json.dumps(cache[pid], ensure_ascii=False, default=str) + "\n")
    os.replace(tmp, path)
    return len(cache)


# ==============================================================================
# Teacher calls
# ==============================================================================

LEAK_TOKENS = ("Is_Safe", "Risk_Categories", "Teacher_Reasoning", '"Reasoning"', "is_safe")


def assert_blind(prompt, pid):
    for tok in LEAK_TOKENS:
        if tok in prompt:
            raise RuntimeError(f"Leak check failed for Patient {pid}: prompt contains {tok!r}")


def call_teacher(client, prompt, pid):
    for attempt in range(C.MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=C.TEACHER_MODEL,
                messages=[{"role": "system", "content": TEACHER_SYSTEM_PROMPT},
                          {"role": "user", "content": prompt}],
                temperature=0.0,
                stream=False,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            wait = C.BACKOFF_BASE * (2 ** attempt)
            print(f"\n[retry {attempt + 1}/{C.MAX_RETRIES}] Patient {pid}: {e}. sleeping {wait}s",
                  file=sys.stderr)
            time.sleep(wait)
    return None


def unroll_row(row, client, deterministic_only=False):
    """Run the full chain for one row. Returns the cache record."""
    pid = str(row.get("Patient ID"))
    ctx = deterministic_context(row)
    notes = {"patient_constraints": "ok", "deterministic": "ok", "prescription": "ok"}
    raw = {}
    status = "ok"

    if not deterministic_only:
        if client is None:
            raise RuntimeError("Teacher client required unless --deterministic-only")
        for name in C.TEACHER_PRIMITIVES:
            prompt = PRIMITIVE_BUILDERS[name](visible_row(row), ctx)
            assert_blind(prompt, pid)
            text = call_teacher(client, prompt, pid)
            if text is None:
                notes[name] = "api_error"
                ctx[name] = None
                status = "api_error"
                raw[name] = None
                continue
            raw[name] = text
            block, note = parse_primitive_response(name, text)
            notes[name] = note
            ctx[name] = block
            if block is None and status == "ok":
                status = "parse_failed"

    return {
        "patient_id": pid,
        "prompt_version": C.PROMPT_VERSION,
        "teacher_model": None if deterministic_only else C.TEACHER_MODEL,
        "status": status,
        "notes": notes,
        "context": ctx,
        "raw": raw,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


# ==============================================================================
# Split derivation
# ==============================================================================

def derive_splits(input_folder, cache, out_dir, splits):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for split in splits:
        src = find_split(input_folder, split)
        if src is None:
            continue
        rows = read_csv(src)
        n_hit = 0
        for r in rows:
            rec = cache.get(str(r.get("Patient ID")))
            if rec and rec.get("status") in ("ok", "parse_failed"):
                r[C.CONTEXT_COLUMN] = json.dumps(rec["context"], ensure_ascii=False, default=str)
                n_hit += 1
            else:
                r[C.CONTEXT_COLUMN] = ""
        fields = list(rows[0].keys()) if rows else []
        if C.CONTEXT_COLUMN not in fields:
            fields.append(C.CONTEXT_COLUMN)
        dst = out_dir / f"{split}.csv"
        with open(dst, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        summary[split] = (n_hit, len(rows))
        print(f"[split] {split:5s} {n_hit}/{len(rows)} rows have context -> {dst}")
    return summary


# ==============================================================================
# Report
# ==============================================================================

def report(cache):
    print(f"cache rows: {len(cache)}")
    print("status:", dict(Counter(r.get("status") for r in cache.values())))
    print("prompt_version:", dict(Counter(r.get("prompt_version") for r in cache.values())))
    for name in C.TEACHER_PRIMITIVES:
        notes = Counter((r.get("notes") or {}).get(name, "missing") for r in cache.values())
        print(f"  {name:14s}", dict(notes))
    sev = Counter()
    mgmt = Counter()
    for r in cache.values():
        inter = ((r.get("context") or {}).get("interactions") or {}).get("interactions") or []
        for it in inter:
            if isinstance(it, dict):
                sev[str(it.get("severity"))] += 1
                mgmt[str(it.get("management_change_required"))] += 1
    print("interaction severities:", dict(sev))
    print("management_change_required:", dict(mgmt))


# ==============================================================================
# Main
# ==============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-folder", default=str(C.SPLITS_DIR))
    ap.add_argument("--cache", default=str(C.UNROLL_CACHE_JSONL))
    ap.add_argument("--out-dir", default=str(C.UNROLLED_SPLITS_DIR))
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--limit", type=int, default=None, help="process at most N new rows")
    ap.add_argument("--ids", default=None, help="comma-separated Patient IDs to process")
    ap.add_argument("--workers", type=int, default=C.DEFAULT_WORKERS)
    ap.add_argument("--deterministic-only", action="store_true",
                    help="only the rule-based primitives; no API calls")
    ap.add_argument("--retry-errors", action="store_true",
                    help="re-run rows whose cached status is api_error or parse_failed")
    ap.add_argument("--split", action="store_true", help="only derive split CSVs from the cache")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--compact", action="store_true", help="dedupe the cache file in place")
    args = ap.parse_args()

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    cache = load_cache(args.cache)

    if args.compact:
        n = compact_cache(args.cache)
        print(f"compacted cache: {n} rows")
        return
    if args.report:
        report(cache)
        return
    if args.split:
        derive_splits(args.input_folder, cache, args.out_dir, splits)
        return

    # Collect rows to process
    rows = []
    seen = set()
    for split in splits:
        src = find_split(args.input_folder, split)
        if src is None:
            print(f"[warn] no {split} csv in {args.input_folder}", file=sys.stderr)
            continue
        for r in read_csv(src):
            pid = str(r.get("Patient ID"))
            if pid in seen:
                continue
            seen.add(pid)
            rows.append(r)
    print(f"{len(rows)} unique rows across {splits}")

    if args.ids:
        want = {x.strip() for x in args.ids.split(",")}
        rows = [r for r in rows if str(r.get("Patient ID")) in want]

    def needs_work(r):
        rec = cache.get(str(r.get("Patient ID")))
        if rec is None:
            return True
        if rec.get("prompt_version") != C.PROMPT_VERSION:
            return True
        if args.retry_errors and rec.get("status") != "ok":
            return True
        if args.deterministic_only:
            return False
        # a deterministic-only record must be upgraded when the teacher runs
        return rec.get("teacher_model") is None

    todo = [r for r in rows if needs_work(r)]
    if args.limit is not None:
        todo = todo[:args.limit]
    print(f"{len(todo)} rows to unroll ({len(rows) - len(todo)} cached)")

    client = None
    if not args.deterministic_only and todo:
        key = os.environ.get(C.TEACHER_API_KEY_ENV)
        if not key:
            sys.exit(f"Set {C.TEACHER_API_KEY_ENV} in the environment (or use --deterministic-only).")
        from openai import OpenAI
        client = OpenAI(api_key=key, base_url=C.TEACHER_BASE_URL)
        print(f"teacher: {C.TEACHER_MODEL}  prompt_version: {C.PROMPT_VERSION}")

    if todo:
        from tqdm import tqdm
        status = Counter()
        workers = 1 if args.deterministic_only else max(1, args.workers)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(unroll_row, r, client, args.deterministic_only): r for r in todo}
            for fut in tqdm(as_completed(futs), total=len(futs), desc="Unrolling"):
                r = futs[fut]
                try:
                    rec = fut.result()
                except Exception as e:
                    rec = {"patient_id": str(r.get("Patient ID")), "prompt_version": C.PROMPT_VERSION,
                           "teacher_model": C.TEACHER_MODEL, "status": "exception",
                           "notes": {"exception": str(e)}, "context": None, "raw": {}}
                status[rec["status"]] += 1
                append_cache(args.cache, rec)
                cache[rec["patient_id"]] = rec
        print("done:", dict(status))

    derive_splits(args.input_folder, cache, args.out_dir, splits)


if __name__ == "__main__":
    main()
