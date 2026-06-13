"""
msn/portfolio_engine.py
Phase 3: Portfolio Command Center
1000+ API support, health scores, risk heatmaps,
budget forecasting, development prioritization,
regulatory workload forecasting.
"""

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class PortfolioMetrics:
    total_apis: int
    low_risk: int
    moderate_risk: int
    high_risk: int
    portfolio_health_score: float
    avg_api_score: float
    avg_mfg_complexity: float
    genotox_count: int
    bcs_iv_count: int
    reg_ready_count: int
    total_estimated_cost_lakhs: float
    total_estimated_months: float
    regulatory_workload_score: float
    priority_queue: List[Dict]
    risk_heatmap: List[Dict]
    budget_forecast: Dict
    workload_forecast: Dict
    category_breakdown: Dict
    top_risks: List[Dict]
    generated_at: str


def analyze_portfolio(molecules: List[Dict],
                      use_ml: bool = True) -> PortfolioMetrics:
    """
    Full portfolio analysis for 1000+ APIs.
    molecules: list of {"smiles","name","category","status"}
    """
    from msn.msn_pipeline import profile_api, screen_impurities
    from msn.manufacturing_score import calculate_manufacturing_score
    from dataclasses import asdict as _asdict

    start = time.time()
    logger.info(f"Portfolio analysis: {len(molecules)} APIs")

    results = []
    errors  = []

    for mol in molecules:
        try:
            profile  = profile_api(
                smiles=mol["smiles"],
                name=mol.get("name","API"),
                category=mol.get("category","Unknown"),
            )
            pd = _asdict(profile)
            mfg = calculate_manufacturing_score(
                smiles=mol["smiles"],
                api_name=mol.get("name","API"),
                profile_data=pd,
            )

            # ML predictions if available
            ml_data = {}
            if use_ml:
                try:
                    from msn.ml_models import get_models
                    ml_data = get_models().predict(mol["smiles"])
                except Exception:
                    pass

            # Risk classification
            genotox = len(pd.get("genotox_alerts",[]))
            sol     = pd.get("aqueous_solubility","")
            if genotox > 0 or pd.get("mutagenicity_risk") == "HIGH" or sol == "VERY LOW":
                risk = "HIGH"
            elif pd.get("hydrolysis_risk") == "HIGH" or pd.get("oxidation_risk") == "HIGH" or pd.get("chiral_centers",0) > 1:
                risk = "MODERATE"
            else:
                risk = "LOW"

            # Cost estimate
            base_cost = 150
            if genotox > 0:   base_cost += 80
            if sol == "VERY LOW": base_cost += 60
            if mfg.total_score > 50: base_cost += 50
            if pd.get("chiral_centers",0) > 0: base_cost += 30

            # Timeline estimate
            base_months = 24
            if genotox > 0:        base_months += 8
            if mfg.total_score > 60: base_months += 12

            results.append({
                "name":         mol.get("name","API"),
                "smiles":       mol["smiles"],
                "category":     mol.get("category","Unknown"),
                "status":       mol.get("status","active"),
                "api_score":    pd.get("api_score",0),
                "mfg_score":    mfg.total_score,
                "mfg_band":     mfg.complexity_band,
                "risk":         risk,
                "genotox":      genotox,
                "solubility":   sol,
                "chiral":       pd.get("chiral_centers",0),
                "mw":           pd.get("mw",0),
                "logp":         pd.get("logp",0),
                "lipinski":     pd.get("lipinski_pass",False),
                "dmf_flags":    len(pd.get("dmf_flags",[])),
                "est_cost_lakhs": base_cost,
                "est_months":   base_months,
                "reg_ready":    pd.get("api_score",0) >= 0.80,
                "reg_path":     pd.get("regulatory_readiness",""),
                "ml_stability": ml_data.get("ml_stability_label","N/A"),
                "ml_tox":       ml_data.get("ml_toxicity_label","N/A"),
                "priority_score": _priority_score(pd, mfg.total_score, genotox),
            })
        except Exception as e:
            errors.append({"name": mol.get("name","?"), "error": str(e)})

    if not results:
        results = [{"name":"No results","api_score":0,"risk":"UNKNOWN"}]

    # Sort by priority score descending
    results.sort(key=lambda x: x.get("priority_score",0), reverse=True)

    # Metrics
    total   = len(results)
    low_r   = sum(1 for r in results if r["risk"] == "LOW")
    mod_r   = sum(1 for r in results if r["risk"] == "MODERATE")
    high_r  = sum(1 for r in results if r["risk"] == "HIGH")
    scores  = [r["api_score"] for r in results]
    mfg_sc  = [r["mfg_score"] for r in results]
    genotox = sum(1 for r in results if r["genotox"] > 0)
    bcs4    = sum(1 for r in results if r["solubility"] == "VERY LOW")
    reg_rdy = sum(1 for r in results if r.get("reg_ready"))

    health = round(
        0.30 * (low_r / total) +
        0.25 * (sum(scores) / total) +
        0.25 * (1 - high_r / total) +
        0.20 * (reg_rdy / total),
        3
    ) if total > 0 else 0

    # Budget forecast by category
    cat_costs = {}
    for r in results:
        cat = r["category"]
        if cat not in cat_costs:
            cat_costs[cat] = {"count":0,"cost":0,"months":0}
        cat_costs[cat]["count"]  += 1
        cat_costs[cat]["cost"]   += r["est_cost_lakhs"]
        cat_costs[cat]["months"] += r["est_months"]

    budget_forecast = {
        "total_investment_lakhs": round(sum(r["est_cost_lakhs"] for r in results), 0),
        "avg_per_api_lakhs":      round(sum(r["est_cost_lakhs"] for r in results) / total, 0) if total else 0,
        "high_risk_premium_lakhs": round(sum(r["est_cost_lakhs"] for r in results if r["risk"]=="HIGH") * 0.3, 0),
        "by_category": {
            k: {"count":v["count"],
                "total_cost_lakhs": round(v["cost"],0),
                "avg_months": round(v["months"]/v["count"],0) if v["count"] else 0}
            for k,v in cat_costs.items()
        },
    }

    # Regulatory workload forecast
    workload = {
        "total_studies_required": sum(10 + (5 if r["genotox"] else 0) + (3 if r["chiral"] else 0) for r in results),
        "urgent_genotox_studies": genotox * 2,
        "stability_studies":      total * 3,
        "analytical_methods":     total * 2,
        "process_validations":    total,
        "dmf_submissions":        total,
        "estimated_fte_years":    round(total * 0.8, 1),
        "peak_workload_month":    6,
        "workload_by_quarter":    {
            "Q1": round(total * 0.15, 0),
            "Q2": round(total * 0.25, 0),
            "Q3": round(total * 0.35, 0),
            "Q4": round(total * 0.25, 0),
        },
    }

    # Risk heatmap data (mw vs logp coloured by risk)
    heatmap = [
        {"name":r["name"],"mw":r["mw"],"logp":r["logp"],
         "risk":r["risk"],"api_score":r["api_score"],
         "mfg_score":r["mfg_score"]}
        for r in results
    ]

    # Category breakdown
    cat_bd = {}
    for r in results:
        cat = r["category"]
        if cat not in cat_bd:
            cat_bd[cat] = {"count":0,"low":0,"moderate":0,"high":0,"avg_score":0}
        cat_bd[cat]["count"] += 1
        cat_bd[cat][r["risk"].lower()] = cat_bd[cat].get(r["risk"].lower(),0) + 1
        cat_bd[cat]["avg_score"] += r["api_score"]
    for cat in cat_bd:
        n = cat_bd[cat]["count"]
        cat_bd[cat]["avg_score"] = round(cat_bd[cat]["avg_score"]/n, 3) if n else 0

    # Top risks
    top_risks = [r for r in results if r["risk"] == "HIGH"][:10]

    reg_workload_score = min(100, round(
        (genotox * 15 + bcs4 * 10 + high_r * 8 + total * 2) / max(total,1), 0
    ))

    runtime = round(time.time() - start, 2)
    logger.info(f"Portfolio done: {total} APIs in {runtime}s | errors={len(errors)}")

    return PortfolioMetrics(
        total_apis=total,
        low_risk=low_r,
        moderate_risk=mod_r,
        high_risk=high_r,
        portfolio_health_score=health,
        avg_api_score=round(sum(scores)/total, 3) if scores else 0,
        avg_mfg_complexity=round(sum(mfg_sc)/total, 1) if mfg_sc else 0,
        genotox_count=genotox,
        bcs_iv_count=bcs4,
        reg_ready_count=reg_rdy,
        total_estimated_cost_lakhs=budget_forecast["total_investment_lakhs"],
        total_estimated_months=round(sum(r["est_months"] for r in results)/total, 0) if total else 0,
        regulatory_workload_score=reg_workload_score,
        priority_queue=results[:20],
        risk_heatmap=heatmap,
        budget_forecast=budget_forecast,
        workload_forecast=workload,
        category_breakdown=cat_bd,
        top_risks=top_risks,
        generated_at=datetime.utcnow().isoformat(),
    )


def _priority_score(profile: dict, mfg_score: int, genotox: int) -> float:
    """
    Priority score: which APIs need attention first.
    Higher = more urgent to address.
    """
    score = 0.0
    score += genotox * 25
    if profile.get("aqueous_solubility") == "VERY LOW": score += 20
    if profile.get("hydrolysis_risk")    == "HIGH":     score += 10
    if profile.get("oxidation_risk")     == "HIGH":     score += 10
    if profile.get("photosensitive"):                   score += 8
    score += profile.get("chiral_centers",0) * 6
    score += (mfg_score / 100) * 15
    score -= profile.get("api_score",0) * 10
    return round(score, 2)
