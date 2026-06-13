"""
msn/roadmap_engine.py
Development Roadmap Engine.
Auto-generates a prioritised CMC study plan for any API.
Output mirrors what a senior CMC consultant would recommend.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict

logger = logging.getLogger(__name__)


@dataclass
class RoadmapStudy:
    phase: str
    study: str
    regulatory_basis: str
    priority: str
    estimated_duration: str
    estimated_cost_inr: str
    dependency: str
    rationale: str


@dataclass
class DevelopmentRoadmap:
    api_name: str
    total_studies: int
    estimated_total_duration: str
    estimated_total_cost_inr: str
    critical_path_items: List[str]
    phases: Dict[str, List[dict]]
    go_no_go_gates: List[dict]
    risk_summary: str


def generate_roadmap(
    api_name: str,
    profile_data: dict,
    manufacturing_score: int = 0,
) -> DevelopmentRoadmap:
    """
    Generate a complete CMC development roadmap.
    Based on ICH guidelines Q1-Q11, M7, S2.
    """

    studies = []
    p = profile_data

    genotox     = p.get("genotox_alerts", [])
    hydrolysis  = p.get("hydrolysis_risk", "LOW")
    oxidation   = p.get("oxidation_risk", "LOW")
    photosens   = p.get("photosensitive", False)
    chiral      = p.get("chiral_centers", 0)
    solubility  = p.get("aqueous_solubility", "MODERATE")
    mw          = p.get("mw", 300)
    lipinski    = p.get("lipinski_pass", True)
    bcs         = p.get("solubility_class", "")

    # ── PHASE 1: Characterisation (Month 0-3) ────────────────────
    studies.append(RoadmapStudy(
        phase="Phase 1 — Characterisation",
        study="Structural Confirmation (NMR, MS, IR)",
        regulatory_basis="ICH Q6A",
        priority="CRITICAL",
        estimated_duration="2-4 weeks",
        estimated_cost_inr="₹2-5 Lakhs",
        dependency="API synthesis complete",
        rationale="Mandatory identity confirmation before any biological or stability testing",
    ))
    studies.append(RoadmapStudy(
        phase="Phase 1 — Characterisation",
        study="Physicochemical Characterisation (pKa, LogP, solubility)",
        regulatory_basis="ICH Q6A",
        priority="CRITICAL",
        estimated_duration="4-6 weeks",
        estimated_cost_inr="₹3-6 Lakhs",
        dependency="Structural confirmation",
        rationale="Defines formulation strategy and sets solubility specification",
    ))
    studies.append(RoadmapStudy(
        phase="Phase 1 — Characterisation",
        study="Polymorphism Screening",
        regulatory_basis="ICH Q6A",
        priority="HIGH",
        estimated_duration="6-10 weeks",
        estimated_cost_inr="₹8-15 Lakhs",
        dependency="API batch available",
        rationale="Identifies all solid forms — wrong polymorph causes bioavailability failure",
    ))

    if chiral > 0:
        studies.append(RoadmapStudy(
            phase="Phase 1 — Characterisation",
            study=f"Chiral Purity Assessment ({chiral} stereocenter(s))",
            regulatory_basis="ICH Q6A, FDA Guidance on Stereoisomers",
            priority="CRITICAL",
            estimated_duration="4-8 weeks",
            estimated_cost_inr="₹5-10 Lakhs",
            dependency="Chiral HPLC method development",
            rationale=f"{chiral} chiral center(s) require enantiopurity specification and validated chiral method",
        ))

    # ── PHASE 2: Stability Studies (Month 2-18) ──────────────────
    studies.append(RoadmapStudy(
        phase="Phase 2 — Stability Studies",
        study="Forced Degradation Study (acid, base, oxidative, thermal, photolytic)",
        regulatory_basis="ICH Q1A(R2)",
        priority="CRITICAL",
        estimated_duration="6-8 weeks",
        estimated_cost_inr="₹4-8 Lakhs",
        dependency="Analytical method development",
        rationale="Establishes degradation pathways — mandatory for ICH Q3A impurity qualification",
    ))
    studies.append(RoadmapStudy(
        phase="Phase 2 — Stability Studies",
        study="Accelerated Stability Study (40°C/75%RH, 6 months)",
        regulatory_basis="ICH Q1A(R2)",
        priority="CRITICAL",
        estimated_duration="6 months",
        estimated_cost_inr="₹6-12 Lakhs",
        dependency="Forced degradation complete, packaging selected",
        rationale="Required for all regulatory submissions — predicts shelf life",
    ))
    studies.append(RoadmapStudy(
        phase="Phase 2 — Stability Studies",
        study="Long-term Stability Study (25°C/60%RH, 12-24 months)",
        regulatory_basis="ICH Q1A(R2)",
        priority="CRITICAL",
        estimated_duration="12-24 months",
        estimated_cost_inr="₹10-20 Lakhs",
        dependency="Accelerated study initiated",
        rationale="Establishes shelf life and retest period for DMF submission",
    ))

    if photosens:
        studies.append(RoadmapStudy(
            phase="Phase 2 — Stability Studies",
            study="Photostability Testing (ICH Q1B Option 1 & 2)",
            regulatory_basis="ICH Q1B",
            priority="HIGH",
            estimated_duration="4-6 weeks",
            estimated_cost_inr="₹3-6 Lakhs",
            dependency="Forced degradation complete",
            rationale="Photosensitive structure detected — light-protected packaging required",
        ))

    if hydrolysis == "HIGH":
        studies.append(RoadmapStudy(
            phase="Phase 2 — Stability Studies",
            study="Humidity Stress Testing & Moisture Uptake Study",
            regulatory_basis="ICH Q1A(R2)",
            priority="HIGH",
            estimated_duration="4-6 weeks",
            estimated_cost_inr="₹2-4 Lakhs",
            dependency="Forced degradation complete",
            rationale="High hydrolysis risk — moisture-controlled packaging and processing justified",
        ))

    if oxidation == "HIGH":
        studies.append(RoadmapStudy(
            phase="Phase 2 — Stability Studies",
            study="Oxidative Stability & Antioxidant Screening",
            regulatory_basis="ICH Q1A(R2)",
            priority="HIGH",
            estimated_duration="4-6 weeks",
            estimated_cost_inr="₹2-5 Lakhs",
            dependency="Forced degradation complete",
            rationale="High oxidation risk — antioxidant excipient selection and nitrogen purge evaluation",
        ))

    # ── PHASE 3: Impurity & Safety (Month 4-12) ──────────────────
    studies.append(RoadmapStudy(
        phase="Phase 3 — Impurity & Safety",
        study="Impurity Identification & Structure Elucidation",
        regulatory_basis="ICH Q3A(R2)",
        priority="CRITICAL",
        estimated_duration="8-12 weeks",
        estimated_cost_inr="₹10-20 Lakhs",
        dependency="Forced degradation, process development",
        rationale="All impurities >0.10% must be identified and qualified before NDA/ANDA",
    ))
    studies.append(RoadmapStudy(
        phase="Phase 3 — Impurity & Safety",
        study="Genotoxic Impurity Assessment (ICH M7 Risk Analysis)",
        regulatory_basis="ICH M7",
        priority="CRITICAL" if genotox else "HIGH",
        estimated_duration="4-8 weeks",
        estimated_cost_inr="₹3-8 Lakhs",
        dependency="Impurity identification complete",
        rationale="Mandatory for all APIs — genotoxic impurities require TDI calculation and control",
    ))

    if genotox:
        studies.append(RoadmapStudy(
            phase="Phase 3 — Impurity & Safety",
            study=f"Ames Test — Bacterial Reverse Mutation Assay ({', '.join(genotox)})",
            regulatory_basis="ICH M7, ICH S2(R1)",
            priority="URGENT",
            estimated_duration="6-8 weeks",
            estimated_cost_inr="₹8-15 Lakhs",
            dependency="ICH M7 risk analysis",
            rationale=f"Structural alert(s) {genotox} require in vitro mutagenicity data before any clinical use",
        ))
        studies.append(RoadmapStudy(
            phase="Phase 3 — Impurity & Safety",
            study="In Vitro Mammalian Cell Gene Mutation Assay (L5178Y or CHO)",
            regulatory_basis="ICH S2(R1)",
            priority="URGENT",
            estimated_duration="8-10 weeks",
            estimated_cost_inr="₹12-20 Lakhs",
            dependency="Ames test complete",
            rationale="Required when Ames test positive or structural alert present",
        ))

    # ── PHASE 4: Analytical Methods (Month 2-8) ──────────────────
    studies.append(RoadmapStudy(
        phase="Phase 4 — Analytical Methods",
        study="HPLC Assay & Related Substances Method Development",
        regulatory_basis="ICH Q2(R1)",
        priority="CRITICAL",
        estimated_duration="8-12 weeks",
        estimated_cost_inr="₹6-12 Lakhs",
        dependency="API characterisation",
        rationale="Primary purity method — required for all release and stability testing",
    ))
    studies.append(RoadmapStudy(
        phase="Phase 4 — Analytical Methods",
        study="Analytical Method Validation",
        regulatory_basis="ICH Q2(R1)",
        priority="CRITICAL",
        estimated_duration="8-12 weeks",
        estimated_cost_inr="₹8-15 Lakhs",
        dependency="Method development complete",
        rationale="Mandatory validation: specificity, linearity, accuracy, precision, LOD, LOQ",
    ))

    if solubility in ["LOW", "VERY LOW"]:
        studies.append(RoadmapStudy(
            phase="Phase 4 — Analytical Methods",
            study="Dissolution Method Development & Validation",
            regulatory_basis="USP <711>, FDA Guidance",
            priority="HIGH",
            estimated_duration="6-10 weeks",
            estimated_cost_inr="₹5-10 Lakhs",
            dependency="Formulation selected",
            rationale=f"BCS Class II/IV API — dissolution is bioavailability-limiting step",
        ))

    # ── PHASE 5: Manufacturing & Scale-up (Month 8-24) ───────────
    studies.append(RoadmapStudy(
        phase="Phase 5 — Manufacturing & Scale-up",
        study="Process Development & Optimisation",
        regulatory_basis="ICH Q8(R2), Q11",
        priority="HIGH",
        estimated_duration="3-6 months",
        estimated_cost_inr="₹15-40 Lakhs",
        dependency="Lab-scale synthesis established",
        rationale="Defines critical process parameters — basis for commercial manufacturing",
    ))
    studies.append(RoadmapStudy(
        phase="Phase 5 — Manufacturing & Scale-up",
        study="Process Validation (3 consecutive commercial batches)",
        regulatory_basis="FDA Process Validation Guidance 2011",
        priority="HIGH",
        estimated_duration="6-12 months",
        estimated_cost_inr="₹30-80 Lakhs",
        dependency="Process development complete",
        rationale="Required for commercial approval — demonstrates reproducibility at scale",
    ))
    studies.append(RoadmapStudy(
        phase="Phase 5 — Manufacturing & Scale-up",
        study="Drug Master File (DMF) Compilation & Submission",
        regulatory_basis="FDA 21 CFR 314.420",
        priority="CRITICAL",
        estimated_duration="2-4 months",
        estimated_cost_inr="₹5-15 Lakhs",
        dependency="All characterisation, stability, impurity data complete",
        rationale="Type II DMF required for all APIs supplied to regulated markets",
    ))

    if not lipinski:
        studies.append(RoadmapStudy(
            phase="Phase 5 — Manufacturing & Scale-up",
            study="BCS Biowaiver Assessment & Bioequivalence Study",
            regulatory_basis="FDA BCS Guidance, ICH M9",
            priority="HIGH",
            estimated_duration="6-12 months",
            estimated_cost_inr="₹50-150 Lakhs",
            dependency="Dissolution method validated",
            rationale="Lipinski violation — oral bioavailability uncertain, BE study likely required",
        ))

    # Organise by phase
    phases = {}
    for s in studies:
        ph = s.phase
        if ph not in phases:
            phases[ph] = []
        from dataclasses import asdict
        phases[ph].append(asdict(s))

    # Critical path
    critical = [s.study for s in studies if s.priority in ["CRITICAL","URGENT"]]

    # Go/no-go gates
    gates = [
        {
            "gate": "Gate 1 — Candidate Selection",
            "timing": "Month 0-3",
            "criteria": "Structural confirmation, physicochemical profile, polymorphism screen",
            "decision": "Proceed to stability studies",
        },
        {
            "gate": "Gate 2 — Safety Clearance",
            "timing": "Month 4-8",
            "criteria": "Genotoxicity assays clear, ICH M7 assessment complete",
            "decision": "Proceed to clinical/commercial manufacturing",
        },
        {
            "gate": "Gate 3 — Stability Confidence",
            "timing": "Month 9-12",
            "criteria": "6-month accelerated stability data acceptable, shelf life projectable",
            "decision": "Proceed to process validation",
        },
        {
            "gate": "Gate 4 — DMF Readiness",
            "timing": "Month 18-30",
            "criteria": "Process validation complete, 12-month long-term stability available",
            "decision": "Submit Drug Master File",
        },
    ]

    # Cost and timeline estimate
    total_months = 24 if not genotox else 30
    if manufacturing_score >= 70:
        total_months += 12

    base_cost_low  = 150
    base_cost_high = 400
    if genotox:
        base_cost_low  += 50
        base_cost_high += 100
    if manufacturing_score >= 50:
        base_cost_low  += 50
        base_cost_high += 150

    risk_summary = (
        f"HIGH RISK — {len(genotox)} genotoxic alert(s) require immediate safety testing. "
        f"Estimated additional cost: ₹50-100 Lakhs. Recommend parallel-tracking safety studies."
        if genotox else
        f"MODERATE RISK — No genotoxic alerts. "
        f"Main complexity drivers: {', '.join([s.study[:30] for s in studies if s.priority=='CRITICAL'][:2])}."
        if manufacturing_score >= 40 else
        "LOW RISK — Straightforward development path. Standard ICH package expected."
    )

    return DevelopmentRoadmap(
        api_name=api_name,
        total_studies=len(studies),
        estimated_total_duration=f"{total_months}-{total_months+12} months",
        estimated_total_cost_inr=f"₹{base_cost_low}-{base_cost_high} Lakhs",
        critical_path_items=critical,
        phases=phases,
        go_no_go_gates=gates,
        risk_summary=risk_summary,
    )
