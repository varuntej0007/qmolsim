"""
msn/manufacturing_score.py
Manufacturing Complexity Score — 0 to 100.
Higher score = more complex = higher cost + longer timeline.
Used by MSN production planning teams.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict

logger = logging.getLogger(__name__)


@dataclass
class ManufacturingScore:
    api_name: str
    smiles: str
    total_score: int
    complexity_band: str
    estimated_cogs_index: str
    estimated_timeline: str
    breakdown: List[Dict]
    top_drivers: List[str]
    recommendations: List[str]
    score_color: str


def calculate_manufacturing_score(
    smiles: str,
    api_name: str = "API",
    profile_data: dict = None,
) -> ManufacturingScore:
    """
    Score manufacturing complexity 0-100 across 10 factors.
    Each factor contributes 0-10 points.
    """
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors, Lipinski

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    # Use profile data if provided, else compute
    if profile_data:
        mw          = profile_data.get("mw", Descriptors.MolWt(mol))
        logp        = profile_data.get("logp", Descriptors.MolLogP(mol))
        hbd         = profile_data.get("hbd", Lipinski.NumHDonors(mol))
        hba         = profile_data.get("hba", Lipinski.NumHAcceptors(mol))
        rot         = profile_data.get("rotatable_bonds", rdMolDescriptors.CalcNumRotatableBonds(mol))
        arom        = profile_data.get("aromatic_rings", rdMolDescriptors.CalcNumAromaticRings(mol))
        chiral      = profile_data.get("chiral_centers", 0)
        hydrolysis  = profile_data.get("hydrolysis_risk", "LOW")
        oxidation   = profile_data.get("oxidation_risk", "LOW")
        solubility  = profile_data.get("aqueous_solubility", "MODERATE")
        genotox     = profile_data.get("genotox_alerts", [])
        photosens   = profile_data.get("photosensitive", False)
        reactive    = profile_data.get("reactive_groups", [])
    else:
        mw          = Descriptors.MolWt(mol)
        logp        = Descriptors.MolLogP(mol)
        hbd         = Lipinski.NumHDonors(mol)
        hba         = Lipinski.NumHAcceptors(mol)
        rot         = rdMolDescriptors.CalcNumRotatableBonds(mol)
        arom        = rdMolDescriptors.CalcNumAromaticRings(mol)
        chiral      = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
        hydrolysis  = "LOW"
        oxidation   = "LOW"
        solubility  = "MODERATE"
        genotox     = []
        photosens   = False
        reactive    = []

    breakdown = []

    # ── Factor 1: Molecular size (0-10) ──────────────────────────
    if mw > 600:      f1, f1_note = 10, f"MW {mw:.0f} Da — very large molecule, complex synthesis"
    elif mw > 450:    f1, f1_note = 7,  f"MW {mw:.0f} Da — large molecule, multi-step synthesis likely"
    elif mw > 300:    f1, f1_note = 4,  f"MW {mw:.0f} Da — medium molecule, standard synthesis"
    else:             f1, f1_note = 2,  f"MW {mw:.0f} Da — small molecule, straightforward"
    breakdown.append({"factor":"Molecular Size","score":f1,"max":10,"note":f1_note})

    # ── Factor 2: Chiral centers (0-10) ──────────────────────────
    if chiral >= 4:   f2, f2_note = 10, f"{chiral} chiral centers — asymmetric synthesis or chiral resolution critical"
    elif chiral == 3: f2, f2_note = 8,  f"{chiral} chiral centers — complex stereocontrol required"
    elif chiral == 2: f2, f2_note = 6,  f"{chiral} chiral centers — enantioselective synthesis needed"
    elif chiral == 1: f2, f2_note = 4,  f"{chiral} chiral center — chiral HPLC or resolution step required"
    else:             f2, f2_note = 0,  "No chiral centers — racemic synthesis possible"
    breakdown.append({"factor":"Chirality","score":f2,"max":10,"note":f2_note})

    # ── Factor 3: Solubility / formulation (0-10) ─────────────────
    if solubility == "VERY LOW":    f3, f3_note = 10, "BCS Class IV — nanoparticle or amorphous dispersion formulation required"
    elif solubility == "LOW":       f3, f3_note = 6,  "BCS Class II — micronization or co-solvent formulation likely"
    elif solubility == "MODERATE":  f3, f3_note = 3,  "Moderate solubility — standard formulation approach"
    else:                           f3, f3_note = 1,  "High solubility — minimal formulation effort"
    breakdown.append({"factor":"Solubility & Formulation","score":f3,"max":10,"note":f3_note})

    # ── Factor 4: Chemical stability — hydrolysis (0-10) ─────────
    if hydrolysis == "HIGH":   f4, f4_note = 8, "High hydrolysis risk — anhydrous processing, moisture-controlled manufacturing"
    elif hydrolysis == "MODERATE": f4, f4_note = 4, "Moderate hydrolysis — humidity-controlled storage"
    else:                      f4, f4_note = 1, "Low hydrolysis risk — standard processing"
    breakdown.append({"factor":"Hydrolysis Stability","score":f4,"max":10,"note":f4_note})

    # ── Factor 5: Oxidation stability (0-10) ─────────────────────
    if oxidation == "HIGH":    f5, f5_note = 7, "High oxidation risk — inert atmosphere, antioxidants, special packaging"
    elif oxidation == "MODERATE": f5, f5_note = 4, "Moderate oxidation — nitrogen blanketing during processing"
    else:                      f5, f5_note = 1, "Low oxidation risk — standard processing"
    breakdown.append({"factor":"Oxidation Stability","score":f5,"max":10,"note":f5_note})

    # ── Factor 6: Photosensitivity (0-10) ────────────────────────
    if photosens:  f6, f6_note = 6, "Photosensitive — amber glass, light-protected manufacturing required"
    else:          f6, f6_note = 0, "No photosensitivity — standard lighting acceptable"
    breakdown.append({"factor":"Photosensitivity","score":f6,"max":10,"note":f6_note})

    # ── Factor 7: Reactive functional groups (0-10) ───────────────
    if len(reactive) >= 3:     f7, f7_note = 9, f"{len(reactive)} reactive groups — specialized handling, PPE, safety protocols"
    elif len(reactive) == 2:   f7, f7_note = 6, f"{len(reactive)} reactive groups — careful handling required"
    elif len(reactive) == 1:   f7, f7_note = 3, f"1 reactive group ({reactive[0]}) — standard safety protocols"
    else:                      f7, f7_note = 0, "No highly reactive groups detected"
    breakdown.append({"factor":"Reactive Groups","score":f7,"max":10,"note":f7_note})

    # ── Factor 8: Genotoxic impurity risk (0-10) ─────────────────
    if len(genotox) >= 2:  f8, f8_note = 10, f"Multiple ICH M7 alerts — dedicated genotox-compliant manufacturing suite required"
    elif len(genotox) == 1: f8, f8_note = 7, f"ICH M7 alert ({genotox[0]}) — enhanced impurity controls, dedicated equipment"
    else:                   f8, f8_note = 0, "No genotoxic alerts — standard containment"
    breakdown.append({"factor":"Genotoxic Impurity Risk","score":f8,"max":10,"note":f8_note})

    # ── Factor 9: Molecular complexity (0-10) ────────────────────
    ha = mol.GetNumHeavyAtoms()
    rings = rdMolDescriptors.CalcNumRings(mol)
    complexity_raw = (ha * 0.15) + (arom * 1.5) + (rings * 0.8) + (rot * 0.3)
    if complexity_raw > 20:    f9, f9_note = 9, f"Very complex structure ({ha} heavy atoms, {rings} rings) — long synthesis route"
    elif complexity_raw > 12:  f9, f9_note = 6, f"Complex structure ({ha} heavy atoms, {arom} aromatic rings)"
    elif complexity_raw > 6:   f9, f9_note = 3, f"Moderate complexity ({ha} heavy atoms)"
    else:                      f9, f9_note = 1, f"Simple structure ({ha} heavy atoms)"
    breakdown.append({"factor":"Structural Complexity","score":f9,"max":10,"note":f9_note})

    # ── Factor 10: Purification difficulty (0-10) ────────────────
    if logp > 5:    f10, f10_note = 8, f"LogP {logp:.2f} — very lipophilic, complex chromatographic purification"
    elif logp > 3:  f10, f10_note = 5, f"LogP {logp:.2f} — lipophilic, reverse-phase purification"
    elif logp < 0:  f10, f10_note = 6, f"LogP {logp:.2f} — very hydrophilic, ion-exchange chromatography likely"
    else:           f10, f10_note = 2, f"LogP {logp:.2f} — moderate lipophilicity, standard purification"
    breakdown.append({"factor":"Purification Difficulty","score":f10,"max":10,"note":f10_note})

    total = sum(b["score"] for b in breakdown)

    # Complexity band
    if total >= 70:
        band      = "VERY HIGH"
        cogs      = "₹800-2000/kg — specialist manufacturing required"
        timeline  = "18-36 months to commercial scale"
        color     = "#e05050"
    elif total >= 50:
        band      = "HIGH"
        cogs      = "₹300-800/kg — experienced CMO required"
        timeline  = "12-24 months to commercial scale"
        color     = "#f59e0b"
    elif total >= 30:
        band      = "MODERATE"
        cogs      = "₹100-300/kg — standard API manufacturing"
        timeline  = "6-12 months to commercial scale"
        color     = "#4a9eff"
    else:
        band      = "LOW"
        cogs      = "₹50-100/kg — commodity API manufacturing"
        timeline  = "3-6 months to commercial scale"
        color     = "#059669"

    # Top drivers
    top_drivers = [
        b["factor"] for b in sorted(breakdown, key=lambda x: x["score"], reverse=True)
        if b["score"] >= 5
    ][:3]

    # Recommendations
    recs = []
    if chiral > 0:
        recs.append(f"Evaluate asymmetric synthesis vs chiral resolution for {chiral} stereocenter(s) — cost impact significant at scale")
    if solubility in ["LOW","VERY LOW"]:
        recs.append("Commission solubility enhancement study — nanoparticle, amorphous solid dispersion, or co-crystal screening")
    if hydrolysis == "HIGH":
        recs.append("Invest in moisture-controlled manufacturing suite — reduces batch failure rate")
    if len(genotox) > 0:
        recs.append("Dedicate separate manufacturing equipment for this API — cross-contamination liability")
    if oxidation == "HIGH":
        recs.append("Evaluate antioxidant excipient selection and nitrogen blanketing ROI at commercial scale")
    if not recs:
        recs.append("Standard API manufacturing infrastructure sufficient — no specialized investment required")

    return ManufacturingScore(
        api_name=api_name,
        smiles=smiles,
        total_score=total,
        complexity_band=band,
        estimated_cogs_index=cogs,
        estimated_timeline=timeline,
        breakdown=breakdown,
        top_drivers=top_drivers,
        recommendations=recs,
        score_color=color,
    )
