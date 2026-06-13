"""
msn/benchmarking.py
Competitive Benchmarking Engine.
Compares your API against industry reference molecules.
Shows percentile rankings across key manufacturing and regulatory dimensions.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict

logger = logging.getLogger(__name__)

# Industry reference database — real approved APIs with known properties
INDUSTRY_REFERENCE = [
    {"name":"Metformin",      "mw":129.16, "logp":-1.43, "api_score":0.93, "solubility":"HIGH",      "chiral":0, "genotox":0, "mfg_score":15},
    {"name":"Aspirin",        "mw":180.16, "logp":1.31,  "api_score":0.86, "solubility":"MODERATE",  "chiral":0, "genotox":0, "mfg_score":18},
    {"name":"Paracetamol",    "mw":151.16, "logp":0.46,  "api_score":0.86, "solubility":"MODERATE",  "chiral":0, "genotox":0, "mfg_score":14},
    {"name":"Ibuprofen",      "mw":206.28, "logp":3.97,  "api_score":0.78, "solubility":"LOW",       "chiral":1, "genotox":0, "mfg_score":28},
    {"name":"Omeprazole",     "mw":345.42, "logp":2.23,  "api_score":0.70, "solubility":"LOW",       "chiral":1, "genotox":0, "mfg_score":35},
    {"name":"Ciprofloxacin",  "mw":331.35, "logp":1.58,  "api_score":0.76, "solubility":"LOW",       "chiral":0, "genotox":0, "mfg_score":32},
    {"name":"Amlodipine",     "mw":408.88, "logp":3.00,  "api_score":0.65, "solubility":"LOW",       "chiral":1, "genotox":0, "mfg_score":42},
    {"name":"Losartan",       "mw":422.91, "logp":4.01,  "api_score":0.61, "solubility":"LOW",       "chiral":0, "genotox":0, "mfg_score":45},
    {"name":"Pantoprazole",   "mw":383.37, "logp":1.53,  "api_score":0.68, "solubility":"LOW",       "chiral":1, "genotox":0, "mfg_score":40},
    {"name":"Atorvastatin",   "mw":558.64, "logp":6.45,  "api_score":0.41, "solubility":"VERY LOW",  "chiral":2, "genotox":0, "mfg_score":58},
    {"name":"Metoprolol",     "mw":267.36, "logp":1.88,  "api_score":0.72, "solubility":"MODERATE",  "chiral":1, "genotox":0, "mfg_score":30},
    {"name":"Lisinopril",     "mw":405.49, "logp":-1.60, "api_score":0.69, "solubility":"MODERATE",  "chiral":3, "genotox":0, "mfg_score":55},
    {"name":"Amoxicillin",    "mw":365.40, "logp":0.87,  "api_score":0.74, "solubility":"MODERATE",  "chiral":2, "genotox":0, "mfg_score":38},
    {"name":"Azithromycin",   "mw":748.98, "logp":4.02,  "api_score":0.38, "solubility":"LOW",       "chiral":9, "genotox":0, "mfg_score":72},
    {"name":"Cetirizine",     "mw":388.89, "logp":2.05,  "api_score":0.67, "solubility":"MODERATE",  "chiral":1, "genotox":0, "mfg_score":36},
    {"name":"Diclofenac",     "mw":296.15, "logp":4.51,  "api_score":0.62, "solubility":"LOW",       "chiral":0, "genotox":0, "mfg_score":33},
    {"name":"Simvastatin",    "mw":418.57, "logp":4.68,  "api_score":0.50, "solubility":"VERY LOW",  "chiral":4, "genotox":0, "mfg_score":62},
    {"name":"Fluconazole",    "mw":306.27, "logp":0.50,  "api_score":0.75, "solubility":"MODERATE",  "chiral":1, "genotox":0, "mfg_score":29},
    {"name":"Gabapentin",     "mw":171.24, "logp":-1.10, "api_score":0.82, "solubility":"HIGH",      "chiral":1, "genotox":0, "mfg_score":22},
    {"name":"Ondansetron",    "mw":293.37, "logp":2.47,  "api_score":0.71, "solubility":"MODERATE",  "chiral":1, "genotox":0, "mfg_score":34},
]


@dataclass
class BenchmarkResult:
    api_name: str
    api_score: float
    mfg_score: int
    percentiles: Dict[str, float]
    rankings: Dict[str, str]
    radar_data: Dict[str, float]
    comparable_apis: List[Dict]
    better_than_pct: float
    industry_position: str
    competitive_summary: str
    strengths: List[str]
    weaknesses: List[str]


def benchmark_api(
    api_name: str,
    profile_data: dict,
    mfg_score: int = 0,
) -> BenchmarkResult:
    """
    Benchmark an API against 20 industry reference molecules.
    Returns percentile rankings across 6 dimensions.
    """
    p      = profile_data
    score  = p.get("api_score", 0.5)
    mw     = p.get("mw", 300)
    logp   = p.get("logp", 2.0)
    chiral = p.get("chiral_centers", 0)
    genotox= len(p.get("genotox_alerts", []))
    sol    = p.get("aqueous_solubility", "MODERATE")

    sol_score_map = {"HIGH": 4, "MODERATE": 3, "LOW": 2, "VERY LOW": 1}
    sol_num = sol_score_map.get(sol, 2)

    ref = INDUSTRY_REFERENCE

    # ── Percentile calculations ───────────────────────────────────
    def percentile(value, ref_values, higher_is_better=True):
        if higher_is_better:
            below = sum(1 for v in ref_values if v < value)
        else:
            below = sum(1 for v in ref_values if v > value)
        return round((below / len(ref_values)) * 100, 1)

    pct_score   = percentile(score,   [r["api_score"]  for r in ref], True)
    pct_mfg     = percentile(mfg_score,[r["mfg_score"] for r in ref], False)
    pct_sol     = percentile(sol_num,  [sol_score_map.get(r["solubility"],2) for r in ref], True)
    pct_mw      = percentile(mw,       [r["mw"]        for r in ref], False)
    pct_chiral  = percentile(chiral,   [r["chiral"]    for r in ref], False)
    pct_genotox = percentile(genotox,  [r["genotox"]   for r in ref], False)

    percentiles = {
        "API Quality Score":       pct_score,
        "Manufacturing Simplicity":pct_mfg,
        "Aqueous Solubility":      pct_sol,
        "Molecular Simplicity":    pct_mw,
        "Chiral Simplicity":       pct_chiral,
        "Safety Profile":          pct_genotox,
    }

    def rank_label(pct):
        if pct >= 80: return "Top 20%"
        if pct >= 60: return "Top 40%"
        if pct >= 40: return "Average"
        if pct >= 20: return "Below average"
        return "Bottom 20%"

    rankings = {k: rank_label(v) for k, v in percentiles.items()}

    # Overall position
    overall_pct = round(sum(percentiles.values()) / len(percentiles), 1)
    better_than = round(overall_pct, 1)

    if overall_pct >= 75:
        position = "★★ INDUSTRY LEADER — top quartile across all dimensions"
    elif overall_pct >= 55:
        position = "★ COMPETITIVE — above industry average"
    elif overall_pct >= 35:
        position = "◆ AVERAGE — in line with industry"
    else:
        position = "▲ BELOW AVERAGE — improvement opportunities identified"

    # Comparable APIs (similar profile)
    comparable = sorted(
        ref,
        key=lambda r: abs(r["api_score"] - score) + abs(r["mw"] - mw)/500,
    )[:4]

    # Radar data (0-1 normalised for chart)
    radar = {k: round(v/100, 3) for k, v in percentiles.items()}

    # Strengths and weaknesses
    strengths  = [k for k, v in percentiles.items() if v >= 65]
    weaknesses = [k for k, v in percentiles.items() if v < 35]

    summary = (
        f"{api_name} ranks better than {better_than}% of industry reference APIs. "
        f"{'Key advantages: ' + ', '.join(strengths[:2]) + '.' if strengths else 'No dominant advantages detected.'} "
        f"{'Main gaps vs industry: ' + ', '.join(weaknesses[:2]) + '.' if weaknesses else 'No significant gaps vs industry.'}"
    )

    return BenchmarkResult(
        api_name=api_name,
        api_score=score,
        mfg_score=mfg_score,
        percentiles=percentiles,
        rankings=rankings,
        radar_data=radar,
        comparable_apis=comparable,
        better_than_pct=better_than,
        industry_position=position,
        competitive_summary=summary,
        strengths=strengths,
        weaknesses=weaknesses,
    )
