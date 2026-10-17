#!/usr/bin/env python3
"""
Pull real adverse-event reports from FDA FAERS through the openFDA API.

Only reports with a usable patient record are requested: age in years, sex,
weight, and at least one concomitant (drugcharacterization=2) medication in
addition to the suspect drug. openFDA caps `skip` at 25 000 per query, so the
pull is split into monthly receive-date windows and takes a fixed number of
pages from each; that also spreads the sample over time.

Output: data/faers_raw.jsonl, one compact record per report:
  safetyreportid, receivedate, serious, seriousness flags, age, sex, weight,
  reactions (MedDRA PT), drugs [{role, product, generic, indication, dose}]

Usage: python fetch_faers.py [--pages-per-month 20] [--start 2022-01] [--end 2025-09]
"""
import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "data" / "faers_raw.jsonl"
API = "https://api.fda.gov/drug/event.json"
BASE_Q = ("patient.patientonsetageunit:801 AND patient.drug.drugcharacterization:2 "
          "AND _exists_:patient.patientweight AND _exists_:patient.patientsex "
          "AND _exists_:patient.patientonsetage")


def month_windows(start, end):
    y, m = map(int, start.split("-"))
    ye, me = map(int, end.split("-"))
    while (y, m) <= (ye, me):
        last = 31 if m in (1, 3, 5, 7, 8, 10, 12) else 30 if m != 2 else 29
        yield f"{y}{m:02d}01", f"{y}{m:02d}{last}"
        m += 1
        if m == 13:
            y, m = y + 1, 1


def get(url, retries=5):
    for i in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=90) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if e.code == 429:
                time.sleep(15)
                continue
            time.sleep(3 * (i + 1))
        except Exception:
            time.sleep(3 * (i + 1))
    return None


def compact(r):
    p = r.get("patient", {})
    try:
        age = float(p["patientonsetage"])
        weight = float(p["patientweight"])
    except (KeyError, ValueError, TypeError):
        return None
    drugs = []
    for d in p.get("drug", []):
        role = d.get("drugcharacterization")
        if role not in ("1", "2"):
            continue
        gen = d.get("openfda", {}).get("generic_name") or []
        drugs.append({
            "role": "suspect" if role == "1" else "concomitant",
            "product": (d.get("medicinalproduct") or "").strip(),
            "generic": gen[0] if gen else None,
            "indication": (d.get("drugindication") or "").strip() or None,
            "dose": (d.get("drugdosagetext") or "").strip() or None,
            "route": d.get("drugadministrationroute"),
        })
    return {
        "safetyreportid": r.get("safetyreportid"),
        "receivedate": r.get("receivedate"),
        "country": r.get("occurcountry"),
        "serious": r.get("serious"),
        "seriousness": {k: r.get(k) for k in (
            "seriousnessdeath", "seriousnesshospitalization", "seriousnesslifethreatening",
            "seriousnessdisabling", "seriousnesscongenitalanomali", "seriousnessother") if r.get(k)},
        "age": age, "sex": p.get("patientsex"), "weight": weight,
        "reactions": [x.get("reactionmeddrapt") for x in p.get("reaction", []) if x.get("reactionmeddrapt")],
        "drugs": drugs,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages-per-month", type=int, default=20)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--start", default="2022-01")
    ap.add_argument("--end", default="2025-09")
    a = ap.parse_args()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    n = 0
    with open(OUT, "w") as f:
        for lo, hi in month_windows(a.start, a.end):
            q = f"{BASE_Q} AND receivedate:[{lo} TO {hi}]"
            for page in range(a.pages_per_month):
                url = API + "?" + urllib.parse.urlencode(
                    {"search": q, "limit": a.limit, "skip": page * a.limit})
                d = get(url)
                if not d or not d.get("results"):
                    break
                for r in d["results"]:
                    c = compact(r)
                    if c and c["safetyreportid"] not in seen:
                        seen.add(c["safetyreportid"])
                        f.write(json.dumps(c) + "\n")
                        n += 1
                time.sleep(0.3)   # stay under 240 req/min without an API key
            print(f"{lo[:6]}: cumulative {n}", flush=True)
    print(f"wrote {n} reports -> {OUT}")


if __name__ == "__main__":
    main()
