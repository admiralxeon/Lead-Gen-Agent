TIERS = {"hot", "warm", "cold"}


def normalize(raw: dict, fallback_name: str, url: str) -> dict:
    """Coerce a model's JSON into a safe, complete assessment dict."""

    def to_int(v, default=0):
        try:
            return max(0, min(100, int(round(float(v)))))
        except (TypeError, ValueError):
            return default

    def to_list(v):
        if isinstance(v, list):
            return [str(x).strip() for x in v if str(x).strip()]
        if isinstance(v, str) and v.strip():
            return [v.strip()]
        return []

    score = to_int(raw.get("lead_score"))
    tier = "hot" if score >= 70 else "warm" if score >= 45 else "cold"

    return {
        "url": url,
        "company_name": str(raw.get("company_name") or fallback_name).strip(),
        "website_quality_score": to_int(raw.get("website_quality_score")),
        "lead_score": to_int(raw.get("lead_score")),
        "tier": tier,
        "observations": to_list(raw.get("observations")),
        "opportunities": to_list(raw.get("opportunities")),
        "summary": str(raw.get("summary", "")).strip(),
    }
