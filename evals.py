"""
Label-free evaluations (eval Part B).

Two things you can measure WITHOUT hand-labelled ground truth:

1. CONSISTENCY - run the same prospect N times through the same model. LLMs are
   non-deterministic, so the question isn't "is it right" but "is it stable".
   A model whose score swings 40 points between identical runs can't be trusted
   to rank leads, no matter how good any single answer looks.

2. BACKEND AGREEMENT - run the same prospects through two backends and compare.
   Where they agree, you can probably trust the cheap one. Where they disagree,
   you've found the genuinely ambiguous prospects - which are exactly the ones
   worth hand-labelling for the accuracy eval (Part C).

Neither needs ground truth, which is why they come first.
"""

import json
import statistics
from collections import Counter

import pipeline

TIER_ORDER = {"cold": 0, "warm": 1, "hot": 2, "review": -1}


# ------------------------------------------------------------- consistency
def consistency_eval(client, url: str, n: int = 5, name: str = "") -> dict:
    """Score the same prospect n times; report how much the answer moves."""
    scores, tiers, errors = [], [], 0
    for i in range(n):
        try:
            a = pipeline.assess_prospect(client, name, url)
            scores.append(a["lead_score"])
            tiers.append(a["tier"])
        except Exception:
            errors += 1

    if not scores:
        return {"url": url, "runs": n, "errors": errors, "usable": 0}

    tier_counts = Counter(tiers)
    top_tier, top_n = tier_counts.most_common(1)[0]

    return {
        "url": url,
        "runs": n,
        "usable": len(scores),
        "errors": errors,
        "scores": scores,
        "mean": round(statistics.mean(scores), 1),
        "stdev": round(statistics.stdev(scores), 1) if len(scores) > 1 else 0.0,
        "min": min(scores),
        "max": max(scores),
        "spread": max(scores) - min(scores),
        "tiers": dict(tier_counts),
        "tier_stability": round(top_n / len(tiers), 2),  # 1.0 = always same tier
        "modal_tier": top_tier,
    }


def consistency_verdict(r: dict) -> str:
    """Turn the numbers into a plain-language judgement."""
    if not r.get("usable"):
        return "UNUSABLE - every run errored"
    if r["tier_stability"] == 1.0 and r["spread"] <= 10:
        return "STABLE - safe to rank on"
    if r["tier_stability"] >= 0.8 and r["spread"] <= 25:
        return "ACCEPTABLE - minor wobble"
    return "UNSTABLE - scores move too much to rank reliably"


# -------------------------------------------------------------- agreement
def agreement_eval(client_a, client_b, urls, label_a="A", label_b="B") -> dict:
    """Run the same prospects through two backends and compare their verdicts."""
    rows, tier_matches, score_diffs = [], 0, []

    for url in urls:
        try:
            a = pipeline.assess_prospect(client_a, "", url)
        except Exception as e:
            rows.append({"url": url, "error": f"{label_a}: {e}"})
            continue
        try:
            b = pipeline.assess_prospect(client_b, "", url)
        except Exception as e:
            rows.append({"url": url, "error": f"{label_b}: {e}"})
            continue

        same = a["tier"] == b["tier"]
        diff = abs(a["lead_score"] - b["lead_score"])
        tier_matches += 1 if same else 0
        score_diffs.append(diff)

        rows.append({
            "url": url,
            f"{label_a}_tier": a["tier"], f"{label_a}_score": a["lead_score"],
            f"{label_b}_tier": b["tier"], f"{label_b}_score": b["lead_score"],
            "tier_match": same,
            "score_diff": diff,
        })

    compared = len(score_diffs)
    return {
        "compared": compared,
        "tier_agreement": round(tier_matches / compared, 2) if compared else 0.0,
        "mean_score_diff": round(statistics.mean(score_diffs), 1) if score_diffs else 0.0,
        "max_score_diff": max(score_diffs) if score_diffs else 0,
        "rows": rows,
        # The prospects worth hand-labelling first: the models disagree, so a
        # human verdict actually resolves something.
        "disagreements": [r for r in rows if r.get("tier_match") is False],
    }


# ----------------------------------------------------------------- output
def print_consistency(results):
    print("\n=== CONSISTENCY (same prospect, repeated runs) ===")
    for r in results:
        if not r.get("usable"):
            print(f"  {r['url']}: all {r['runs']} runs errored")
            continue
        print(f"\n  {r['url']}")
        print(f"    scores       : {r['scores']}")
        print(f"    mean/stdev   : {r['mean']} / {r['stdev']}")
        print(f"    spread       : {r['spread']} points  (min {r['min']}, max {r['max']})")
        print(f"    tiers        : {r['tiers']}  stability {r['tier_stability']}")
        print(f"    verdict      : {consistency_verdict(r)}")


def print_agreement(res, label_a="A", label_b="B"):
    print(f"\n=== BACKEND AGREEMENT ({label_a} vs {label_b}) ===")
    print(f"  compared        : {res['compared']}")
    print(f"  tier agreement  : {res['tier_agreement']:.0%}")
    print(f"  mean score diff : {res['mean_score_diff']}  (max {res['max_score_diff']})")
    if res["disagreements"]:
        print(f"\n  disagreements (label these first for Part C):")
        for d in res["disagreements"]:
            print(f"    {d['url']}")
            print(f"      {label_a}: {d[f'{label_a}_tier']:6} {d[f'{label_a}_score']:3}"
                  f"   {label_b}: {d[f'{label_b}_tier']:6} {d[f'{label_b}_score']:3}"
                  f"   (diff {d['score_diff']})")
    else:
        print("  no tier disagreements")


def save(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    return path