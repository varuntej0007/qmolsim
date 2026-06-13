"""
msn/digital_twin.py
Phase 4: Digital Twin
User changes salt form, particle size, packaging, storage, polymorph —
system predicts impact on stability, cost, regulatory burden, manufacturing.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class TwinScenario:
    scenario_name: str
    changes: Dict
    stability_impact: Dict
    cost_impact: Dict
    regulatory_impact: Dict
    manufacturing_impact: Dict
    overall_risk_delta: float
    recommendation: str
    warnings: List[str]
    opportunities: List[str]


@dataclass
class DigitalTwinResult:
    api_name: str
    smiles: str
    baseline: Dict
    scenarios: List[Dict]
    best_scenario: str
    worst_scenario: str
    optimization_summary: str


# Impact lookup tables — based on pharmaceutical literature
SALT_FORM_IMPACTS = {
    "HCl":         {"solubility": +0.25, "stability": +0.10, "cost": +5,   "reg_complexity": +1},
    "sodium":      {"solubility": +0.30, "stability": +0.05, "cost": +8,   "reg_complexity": +1},
    "free_base":   {"solubility":  0.00, "stability":  0.00, "cost":  0,   "reg_complexity":  0},
    "mesylate":    {"solubility": +0.20, "stability": +0.15, "cost": +12,  "reg_complexity": +2},
    "maleate":     {"solubility": +0.15, "stability": +0.08, "cost": +10,  "reg_complexity": +2},
    "phosphate":   {"solubility": +0.18, "stability": +0.05, "cost": +7,   "reg_complexity": +1},
    "tartrate":    {"solubility": +0.12, "stability": -0.05, "cost": +9,   "reg_complexity": +2},
    "co_crystal":  {"solubility": +0.40, "stability": +0.20, "cost": +30,  "reg_complexity": +4},
    "amorphous":   {"solubility": +0.50, "stability": -0.25, "cost": +20,  "reg_complexity": +3},
}

PARTICLE_SIZE_IMPACTS = {
    "coarse":      {"solubility": -0.10, "stability": +0.10, "cost": -5,  "mfg_complexity": -5},
    "standard":    {"solubility":  0.00, "stability":  0.00, "cost":  0,  "mfg_complexity":  0},
    "micronized":  {"solubility": +0.20, "stability": -0.05, "cost": +15, "mfg_complexity": +10},
    "nanosized":   {"solubility": +0.45, "stability": -0.15, "cost": +40, "mfg_complexity": +25},
    "spray_dried": {"solubility": +0.35, "stability": -0.10, "cost": +25, "mfg_complexity": +15},
}

PACKAGING_IMPACTS = {
    "hdpe_bottle":  {"stability": +0.05, "cost": -10, "moisture_protection": "MODERATE"},
    "blister_alu":  {"stability": +0.20, "cost": +15, "moisture_protection": "HIGH"},
    "amber_glass":  {"stability": +0.25, "cost": +20, "moisture_protection": "HIGH"},
    "strip_pack":   {"stability": +0.10, "cost": +5,  "moisture_protection": "LOW"},
    "sachet":       {"stability": +0.05, "cost": +3,  "moisture_protection": "LOW"},
}

STORAGE_IMPACTS = {
    "room_temp":    {"stability": 0.00,  "cost":  0,  "shelf_life_months": 24},
    "cool_2_8":     {"stability": +0.20, "cost": +30, "shelf_life_months": 36},
    "frozen":       {"stability": +0.35, "cost": +50, "shelf_life_months": 48},
    "controlled_25":{"stability": +0.10, "cost": +10, "shelf_life_months": 30},
    "desiccant":    {"stability": +0.12, "cost": +8,  "shelf_life_months": 28},
}

POLYMORPH_IMPACTS = {
    "form_I":   {"stability": +0.15, "solubility": -0.05, "cost": 0,   "reg_complexity": 0},
    "form_II":  {"stability": +0.05, "solubility": +0.10, "cost": +5,  "reg_complexity": +1},
    "form_III": {"stability": -0.05, "solubility": +0.20, "cost": +10, "reg_complexity": +2},
    "amorphous":{"stability": -0.20, "solubility": +0.45, "cost": +25, "reg_complexity": +3},
    "solvate":  {"stability": +0.10, "solubility": +0.05, "cost": +8,  "reg_complexity": +2},
    "hydrate":  {"stability": +0.08, "solubility": -0.08, "cost": +5,  "reg_complexity": +1},
}


def run_digital_twin(
    smiles: str,
    api_name: str,
    scenarios: List[Dict],
    profile_data: dict = None,
) -> DigitalTwinResult:
    """
    Run digital twin simulations for given scenarios.

    scenarios: list of dicts, each with keys:
      name, salt_form, particle_size, packaging, storage, polymorph
    """
    from msn.msn_pipeline import profile_api
    from msn.manufacturing_score import calculate_manufacturing_score
    from dataclasses import asdict

    # Get baseline
    if not profile_data:
        profile = profile_api(smiles=smiles, name=api_name, category="Unknown")
        profile_data = asdict(profile)

    mfg = calculate_manufacturing_score(smiles=smiles, api_name=api_name,
                                         profile_data=profile_data)

    baseline = {
        "stability_score":      _stability_score(profile_data),
        "solubility_score":     _solubility_score(profile_data),
        "cost_index":           mfg.total_score,
        "mfg_complexity":       mfg.total_score,
        "regulatory_complexity":len(profile_data.get("dmf_flags",[])),
        "shelf_life_months":    24,
        "api_score":            profile_data.get("api_score",0),
    }

    sim_results = []
    for sc in scenarios:
        sim = _simulate_scenario(sc, baseline, profile_data)
        sim_results.append(asdict(sim))

    # Find best/worst
    if sim_results:
        best  = max(sim_results, key=lambda x: -x["overall_risk_delta"])
        worst = min(sim_results, key=lambda x: -x["overall_risk_delta"])
                # Add rank to each scenario
        ranked = sorted(sim_results, key=lambda x: x["overall_risk_delta"], reverse=True)
        for i, sc in enumerate(ranked):
            sc["rank"] = i + 1
        best_name  = best["scenario_name"]
        worst_name = worst["scenario_name"]
    else:
        best_name = worst_name = "N/A"

    opt_summary = _optimization_summary(baseline, sim_results, profile_data)

    return DigitalTwinResult(
        api_name=api_name,
        smiles=smiles,
        baseline=baseline,
        scenarios=sim_results,
        best_scenario=best_name,
        worst_scenario=worst_name,
        optimization_summary=opt_summary,
    )


def _stability_score(profile: dict) -> float:
    score = 0.5
    if profile.get("hydrolysis_risk") == "LOW":   score += 0.15
    if profile.get("hydrolysis_risk") == "HIGH":  score -= 0.20
    if profile.get("oxidation_risk")  == "LOW":   score += 0.10
    if profile.get("oxidation_risk")  == "HIGH":  score -= 0.15
    if not profile.get("photosensitive"):         score += 0.10
    else:                                          score -= 0.10
    return round(max(0, min(1, score)), 3)


def _solubility_score(profile: dict) -> float:
    sol_map = {"HIGH":0.85,"MODERATE":0.55,"LOW":0.30,"VERY LOW":0.10}
    return sol_map.get(profile.get("aqueous_solubility","MODERATE"), 0.50)


def _simulate_scenario(scenario: dict, baseline: dict,
                        profile: dict) -> TwinScenario:
    name = scenario.get("name", "Scenario")
    delta_stability  = 0.0
    delta_solubility = 0.0
    delta_cost       = 0
    delta_mfg        = 0
    delta_reg        = 0
    delta_shelf_life = 0
    warnings     = []
    opportunities = []

    # Salt form
    salt = scenario.get("salt_form","free_base")
    if salt in SALT_FORM_IMPACTS:
        imp = SALT_FORM_IMPACTS[salt]
        delta_stability  += imp.get("stability",0)
        delta_solubility += imp.get("solubility",0)
        delta_cost       += imp.get("cost",0)
        delta_reg        += imp.get("reg_complexity",0)
        if imp.get("solubility",0) > 0.2:
            opportunities.append(f"Salt form {salt} significantly improves solubility (+{imp['solubility']:.0%})")
        if imp.get("reg_complexity",0) > 2:
            warnings.append(f"Salt form {salt} adds regulatory complexity — new stability studies required")

    # Particle size
    ps = scenario.get("particle_size","standard")
    if ps in PARTICLE_SIZE_IMPACTS:
        imp = PARTICLE_SIZE_IMPACTS[ps]
        delta_solubility += imp.get("solubility",0)
        delta_stability  += imp.get("stability",0)
        delta_cost       += imp.get("cost",0)
        delta_mfg        += imp.get("mfg_complexity",0)
        if ps == "nanosized":
            warnings.append("Nanosizing requires specialized equipment and extensive characterization")
        if ps == "micronized" and profile.get("oxidation_risk") == "HIGH":
            warnings.append("Micronization increases surface area — elevated oxidation risk for this API")

    # Packaging
    pkg = scenario.get("packaging","hdpe_bottle")
    if pkg in PACKAGING_IMPACTS:
        imp = PACKAGING_IMPACTS[pkg]
        delta_stability  += imp.get("stability",0)
        delta_cost       += imp.get("cost",0)
        delta_shelf_life += imp.get("shelf_life_months",24) - 24
        if profile.get("hydrolysis_risk") == "HIGH" and imp.get("moisture_protection") == "LOW":
            warnings.append("High hydrolysis risk API with low moisture protection packaging — not recommended")
        if profile.get("photosensitive") and pkg != "amber_glass":
            warnings.append("Photosensitive API — amber glass packaging strongly recommended")

    # Storage
    storage = scenario.get("storage","room_temp")
    if storage in STORAGE_IMPACTS:
        imp = STORAGE_IMPACTS[storage]
        delta_stability  += imp.get("stability",0)
        delta_cost       += imp.get("cost",0)
        delta_shelf_life += imp.get("shelf_life_months",24) - 24
        if storage == "cool_2_8":
            opportunities.append("Cold chain storage extends shelf life — consider for high-value APIs")

    # Polymorph
    poly = scenario.get("polymorph","form_I")
    if poly in POLYMORPH_IMPACTS:
        imp = POLYMORPH_IMPACTS[poly]
        delta_stability  += imp.get("stability",0)
        delta_solubility += imp.get("solubility",0)
        delta_cost       += imp.get("cost",0)
        delta_reg        += imp.get("reg_complexity",0)
        if poly == "amorphous":
            warnings.append("Amorphous form: high solubility gain but significant stability risk and regulatory burden")
        if imp.get("solubility",0) > 0.15 and profile.get("aqueous_solubility") == "VERY LOW":
            opportunities.append(f"Polymorph {poly} addresses BCS Class IV solubility — significant commercial advantage")

    # Calculate impacts
    new_stability   = round(max(0, min(1, baseline["stability_score"]  + delta_stability)), 3)
    new_solubility  = round(max(0, min(1, baseline["solubility_score"] + delta_solubility)), 3)
    new_cost        = baseline["cost_index"] + delta_cost
    new_mfg         = baseline["mfg_complexity"] + delta_mfg
    new_shelf_life  = baseline["shelf_life_months"] + delta_shelf_life

    overall_risk_delta = round(delta_stability * 0.3 + delta_solubility * 0.2
                               - delta_reg * 0.1 - (delta_cost/100) * 0.1, 3)

    # Stability impact narrative
    stab_change = new_stability - baseline["stability_score"]
    if stab_change > 0.1:
        stab_desc = f"Improved stability (+{stab_change:.0%}) — estimated shelf life {new_shelf_life:.0f} months"
    elif stab_change < -0.1:
        stab_desc = f"Reduced stability ({stab_change:.0%}) — additional stability studies required"
    else:
        stab_desc = f"Stability largely unchanged (delta: {stab_change:+.2%})"

    # Cost narrative
    cost_inr = delta_cost * 10
    if delta_cost > 20:
        cost_desc = f"Significant cost increase: +₹{cost_inr} Lakhs per batch — justify with commercial benefit"
    elif delta_cost < 0:
        cost_desc = f"Cost reduction: ₹{abs(cost_inr)} Lakhs savings per batch"
    else:
        cost_desc = f"Minor cost impact: +₹{cost_inr} Lakhs"

    # Regulatory narrative
    if delta_reg > 2:
        reg_desc = f"High regulatory impact (+{delta_reg} additional studies) — 6-12 month timeline addition"
    elif delta_reg > 0:
        reg_desc = f"Moderate regulatory impact (+{delta_reg} studies) — 3-6 months additional"
    else:
        reg_desc = "No additional regulatory studies required"

    # Manufacturing narrative
    if delta_mfg > 15:
        mfg_desc = f"Significantly increases manufacturing complexity (+{delta_mfg} points) — specialized equipment needed"
    elif delta_mfg > 0:
        mfg_desc = f"Moderate manufacturing complexity increase (+{delta_mfg} points)"
    else:
        mfg_desc = "No significant manufacturing complexity change"

    if overall_risk_delta > 0.1:
        rec = f"RECOMMENDED — Net positive impact. {opportunities[0] if opportunities else 'Proceed with development.'}"
    elif overall_risk_delta < -0.1:
        rec = f"NOT RECOMMENDED — Net negative impact. {warnings[0] if warnings else 'Review risks carefully.'}"
    else:
        rec = "NEUTRAL — Marginal benefit. Evaluate cost-benefit carefully before proceeding."

    return TwinScenario(
        scenario_name=name,
        changes=scenario,
        stability_impact={
            "score": new_stability,
            "baseline": baseline["stability_score"],
            "delta": round(stab_change, 3),
            "description": stab_desc,
            "shelf_life_months": new_shelf_life,
        },
        cost_impact={
            "index": new_cost,
            "baseline": baseline["cost_index"],
            "delta_lakhs": cost_inr,
            "description": cost_desc,
        },
        regulatory_impact={
            "complexity": baseline["regulatory_complexity"] + delta_reg,
            "baseline": baseline["regulatory_complexity"],
            "delta": delta_reg,
            "description": reg_desc,
        },
        manufacturing_impact={
            "complexity": new_mfg,
            "baseline": baseline["mfg_complexity"],
            "delta": delta_mfg,
            "description": mfg_desc,
        },
        overall_risk_delta=overall_risk_delta,
        recommendation=rec,
        warnings=warnings,
        opportunities=opportunities,
    )


def _optimization_summary(baseline: dict, scenarios: list,
                            profile: dict) -> str:
    if not scenarios:
        return "No scenarios simulated."
    best = max(scenarios, key=lambda x: x["overall_risk_delta"])
    worst = min(scenarios, key=lambda x: x["overall_risk_delta"])
    sol = profile.get("aqueous_solubility","")
    lines = [
        f"Digital Twin analysis of {len(scenarios)} scenario(s).",
        f"Best scenario: '{best['scenario_name']}' (risk delta: +{best['overall_risk_delta']:.2f}).",
        f"Worst scenario: '{worst['scenario_name']}' (risk delta: {worst['overall_risk_delta']:.2f}).",
    ]
    if sol in ["LOW","VERY LOW"]:
        lines.append("Priority recommendation: solubility enhancement (salt screening or particle engineering) before process development.")
    if profile.get("hydrolysis_risk") == "HIGH":
        lines.append("Packaging selection is critical — moisture barrier packaging will significantly improve stability.")
    if profile.get("photosensitive"):
        lines.append("Amber glass or opaque packaging mandatory — test all scenarios with light protection.")
    return " ".join(lines)
