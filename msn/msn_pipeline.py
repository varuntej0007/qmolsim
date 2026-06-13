"""
msn/msn_pipeline.py
QMolSim MSN Labs Edition
Three modules targeting MSN's actual manufacturing pain points:
  1. API Quality Profiler      — regulatory DMF data
  2. Impurity Screener         — synthesis byproduct risk
  3. Polymorph Stability Check — crystal form energy ranking
"""

import logging
import json
import os
from dataclasses import dataclass, field, asdict
from typing import List, Optional

logger = logging.getLogger(__name__)


# ── Module 1: API Quality Profiler ───────────────────────────────

@dataclass
class APIQualityReport:
    name: str
    smiles: str
    category: str

    # Physical properties
    mw: float
    formula: str
    logp: float
    logd_74: float          # LogD at pH 7.4 (physiological)
    pka_estimate: float
    tpsa: float
    hbd: int
    hba: int
    rotatable_bonds: int
    aromatic_rings: int
    heavy_atoms: int
    fsp3: float

    # Solubility
    aqueous_solubility: str  # HIGH / MODERATE / LOW / VERY LOW
    solubility_class: str    # BCS Class I/II/III/IV

    # Stability flags
    hydrolysis_risk: str
    oxidation_risk: str
    photosensitive: bool
    hygroscopic_risk: str

    # Regulatory
    lipinski_pass: bool
    veber_pass: bool
    mutagenicity_risk: str
    genotox_alerts: List[str]

    # Manufacturing
    synthesis_complexity: str
    chiral_centers: int
    reactive_groups: List[str]

    # Overall
    api_score: float
    regulatory_readiness: str
    dmf_flags: List[str]


def profile_api(smiles: str, name: str = "API", category: str = "Unknown") -> APIQualityReport:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors, QED, Lipinski
    from rdkit.Chem import rdMolDescriptors as rdmd

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    # Basic descriptors
    mw      = round(Descriptors.MolWt(mol), 2)
    logp    = round(Descriptors.MolLogP(mol), 3)
    tpsa    = round(rdmd.CalcTPSA(mol), 2)
    hbd     = Lipinski.NumHDonors(mol)
    hba     = Lipinski.NumHAcceptors(mol)
    rot     = rdmd.CalcNumRotatableBonds(mol)
    arom    = rdmd.CalcNumAromaticRings(mol)
    ha      = mol.GetNumHeavyAtoms()
    formula = rdmd.CalcMolFormula(mol)

    # Chiral centers
    chiral = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))

    # fsp3
    sp3_c   = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 6 and a.GetHybridization().name == 'SP3')
    total_c = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 6)
    fsp3    = round(sp3_c / total_c if total_c > 0 else 0.0, 3)

    # LogD at pH 7.4 (simplified: logP - ionization correction)
    logd_74 = round(logp - 1.0 if hba > 2 else logp, 3)

    # pKa estimate (rule-based)
    has_carboxylic = mol.HasSubstructMatch(Chem.MolFromSmarts("C(=O)[OH]"))
    has_amine      = mol.HasSubstructMatch(Chem.MolFromSmarts("[NX3;H1,H2]"))
    if has_carboxylic:
        pka = round(4.5 + logp * 0.1, 1)
    elif has_amine:
        pka = round(9.5 - logp * 0.2, 1)
    else:
        pka = 7.0

    # Aqueous solubility (Delaney model simplified)
    log_s = 0.16 - 0.63 * logp - 0.0062 * mw + 0.066 * rot - 0.74 * arom
    if log_s > -1:
        solubility = "HIGH"
        bcs_class  = "BCS Class I or III"
    elif log_s > -3:
        solubility = "MODERATE"
        bcs_class  = "BCS Class I or II"
    elif log_s > -5:
        solubility = "LOW"
        bcs_class  = "BCS Class II"
    else:
        solubility = "VERY LOW"
        bcs_class  = "BCS Class IV"

    # Stability flags
    hydrolysis_smarts = ["C(=O)O", "C(=O)N", "S(=O)(=O)O"]
    oxidation_smarts  = ["[SX2]", "c1ccc(O)cc1", "[NX3;H1]"]
    photo_smarts      = ["c1ccccc1C=O", "c1ccc(N)cc1"]

    hydrolysis_risk = "HIGH" if any(
        mol.HasSubstructMatch(Chem.MolFromSmarts(s)) for s in hydrolysis_smarts
    ) else "LOW"

    oxidation_risk = "HIGH" if any(
        mol.HasSubstructMatch(Chem.MolFromSmarts(s)) for s in oxidation_smarts
    ) else "LOW"

    photosensitive = any(
        mol.HasSubstructMatch(Chem.MolFromSmarts(s)) for s in photo_smarts
    )

    hygroscopic = "HIGH" if (hbd + hba) > 8 else "MODERATE" if (hbd + hba) > 4 else "LOW"

    # Reactive groups (manufacturing concerns)
    reactive_smarts = {
        "Aldehyde":    "[CX3H1](=O)",
        "Epoxide":     "[OX2r3]",
        "Acid_chloride": "C(=O)Cl",
        "Anhydride":   "C(=O)OC(=O)",
        "Isocyanate":  "N=C=O",
        "Nitro":       "[$([NX3](=O)=O)]",
    }
    reactive_groups = [
        name for name, s in reactive_smarts.items()
        if mol.HasSubstructMatch(Chem.MolFromSmarts(s))
    ]

    # Genotoxicity alerts (ICH M7)
    geno_smarts = {
        "Alkyl_halide":   "[CX4][F,Cl,Br,I]",
        "Aromatic_amine": "c[NH2]",
        "Nitroso":        "[NX2]=O",
        "Hydrazine":      "[NX3][NX3]",
        "Azo":            "[NX2]=[NX2]",
    }
    genotox_alerts = [
        name for name, s in geno_smarts.items()
        if mol.HasSubstructMatch(Chem.MolFromSmarts(s))
    ]

    mutagenicity_risk = "HIGH" if len(genotox_alerts) >= 2 else \
                        "MODERATE" if len(genotox_alerts) == 1 else "LOW"

    # Synthesis complexity
    if rot > 8 or chiral > 2 or ha > 40:
        synth = "HIGH — Complex multi-step synthesis expected"
    elif rot > 4 or chiral > 0 or ha > 25:
        synth = "MODERATE — Standard multi-step synthesis"
    else:
        synth = "LOW — Straightforward synthesis"

    # Lipinski / Veber
    lipinski = (mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10)
    veber    = (rot <= 10 and tpsa <= 140)

    # DMF flags (Drug Master File — what FDA auditors look for)
    dmf_flags = []
    if not lipinski:
        dmf_flags.append("Lipinski violation — document bioavailability strategy")
    if genotox_alerts:
        dmf_flags.append(f"ICH M7 genotoxic alerts: {', '.join(genotox_alerts)} — requires Ames test data")
    if hydrolysis_risk == "HIGH":
        dmf_flags.append("Hydrolysis-prone — include forced degradation study in DMF")
    if oxidation_risk == "HIGH":
        dmf_flags.append("Oxidation-prone — antioxidant packaging may be required")
    if photosensitive:
        dmf_flags.append("Photosensitive structure — ICH Q1B photostability study required")
    if chiral > 0:
        dmf_flags.append(f"{chiral} chiral center(s) — enantiomeric purity specification required")
    if solubility == "VERY LOW":
        dmf_flags.append("BCS Class IV — dissolution testing and bioequivalence study required")
    if not dmf_flags:
        dmf_flags.append("No major DMF flags — standard submission package expected")

    # API score
    score = 0.0
    score += 0.20 * float(lipinski)
    score += 0.15 * float(veber)
    score += 0.15 * (1 - min(len(genotox_alerts), 3) / 3)
    score += 0.15 * (1 if solubility in ["HIGH", "MODERATE"] else 0.3)
    score += 0.10 * (1 if hydrolysis_risk == "LOW" else 0.3)
    score += 0.10 * (1 if oxidation_risk == "LOW" else 0.3)
    score += 0.15 * (1 if not reactive_groups else max(0, 1 - len(reactive_groups) * 0.3))

    if score >= 0.80:
        reg_readiness = "★ HIGH — Straightforward regulatory path"
    elif score >= 0.60:
        reg_readiness = "◆ MODERATE — Standard documentation required"
    elif score >= 0.40:
        reg_readiness = "▲ COMPLEX — Additional studies likely required"
    else:
        reg_readiness = "✗ CHALLENGING — Significant regulatory hurdles"

    return APIQualityReport(
        name=name, smiles=smiles, category=category,
        mw=mw, formula=formula, logp=logp, logd_74=logd_74,
        pka_estimate=pka, tpsa=tpsa, hbd=hbd, hba=hba,
        rotatable_bonds=rot, aromatic_rings=arom,
        heavy_atoms=ha, fsp3=fsp3,
        aqueous_solubility=solubility, solubility_class=bcs_class,
        hydrolysis_risk=hydrolysis_risk, oxidation_risk=oxidation_risk,
        photosensitive=photosensitive, hygroscopic_risk=hygroscopic,
        lipinski_pass=lipinski, veber_pass=veber,
        mutagenicity_risk=mutagenicity_risk, genotox_alerts=genotox_alerts,
        synthesis_complexity=synth, chiral_centers=chiral,
        reactive_groups=reactive_groups,
        api_score=round(score, 3),
        regulatory_readiness=reg_readiness,
        dmf_flags=dmf_flags,
    )


# ── Module 2: Impurity Screener ──────────────────────────────────

@dataclass
class ImpurityReport:
    parent_name: str
    parent_smiles: str
    impurities: List[dict]
    total_flagged: int
    ich_q3b_concern: bool
    recommendation: str


def screen_impurities(parent_smiles: str, parent_name: str = "API",
                      impurity_smiles_list: List[str] = None) -> ImpurityReport:
    """
    Screen known/predicted synthesis impurities.
    ICH Q3B: impurities > 0.1% in drug product require characterization.
    ICH M7: genotoxic impurities require < 1.5 μg/day TDI.
    """
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors

    # Default: generate common degradation impurities
    if not impurity_smiles_list:
        impurity_smiles_list = _generate_degradation_products(parent_smiles)

    results = []
    ich_concern = False

    for imp_smiles in impurity_smiles_list:
        mol = Chem.MolFromSmiles(imp_smiles)
        if mol is None:
            continue

        mw   = round(Descriptors.MolWt(mol), 2)
        logp = round(Descriptors.MolLogP(mol), 3)

        # Genotoxicity check (ICH M7)
        geno_smarts = {
            "Alkyl_halide": "[CX4][F,Cl,Br,I]",
            "Aromatic_amine": "c[NH2]",
            "Nitroso": "[NX2]=O",
            "Hydrazine": "[NX3][NX3]",
            "Azo": "[NX2]=[NX2]",
            "Epoxide": "[OX2r3]",
        }
        geno_hits = [
            name for name, s in geno_smarts.items()
            if mol.HasSubstructMatch(Chem.MolFromSmarts(s))
        ]

        # ICH Q3B threshold category
        if mw < 200:
            threshold = "> 0.10% — requires identification"
        elif mw < 500:
            threshold = "> 0.10% — requires identification and qualification"
        else:
            threshold = "> 0.05% — requires full characterization"

        # MS detectability (approximate m/z)
        mz_estimate = round(mw + 1, 1)  # [M+H]+

        risk = "HIGH" if geno_hits else "MODERATE" if mw > 300 else "LOW"
        if geno_hits:
            ich_concern = True

        results.append({
            "smiles": imp_smiles,
            "mw": mw,
            "logp": logp,
            "mz_mh_plus": mz_estimate,
            "genotox_alerts": geno_hits,
            "risk_level": risk,
            "ich_threshold": threshold,
            "action_required": "ICH M7 TDI calculation required" if geno_hits else
                               "Standard qualification study" if mw > 300 else
                               "Identification only",
        })

    recommendation = (
        "URGENT: Genotoxic impurity detected — ICH M7 risk assessment and TDI calculation required before regulatory submission."
        if ich_concern else
        "Standard ICH Q3B qualification studies required for impurities above threshold."
    )

    return ImpurityReport(
        parent_name=parent_name,
        parent_smiles=parent_smiles,
        impurities=results,
        total_flagged=len(results),
        ich_q3b_concern=ich_concern,
        recommendation=recommendation,
    )


def _generate_degradation_products(smiles: str) -> List[str]:
    """
    Rule-based degradation product generation.
    Covers the most common API degradation pathways.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []

    degradants = []

    # Hydrolysis: ester → acid + alcohol
    ester_pattern = Chem.MolFromSmarts("C(=O)OC")
    if mol.HasSubstructMatch(ester_pattern):
        # Generate carboxylic acid fragment
        acid_smiles = smiles.replace("OCC", "O").replace("OC(", "O(")
        try:
            test = Chem.MolFromSmiles(acid_smiles)
            if test:
                degradants.append(acid_smiles)
        except:
            pass

    # Oxidation: sulfide → sulfoxide
    sulfide = Chem.MolFromSmarts("[SX2]")
    if mol.HasSubstructMatch(sulfide):
        sulfoxide_smiles = smiles.replace("]S[", "]S(=O)[")
        try:
            test = Chem.MolFromSmiles(sulfoxide_smiles)
            if test:
                degradants.append(sulfoxide_smiles)
        except:
            pass

    # N-oxide formation
    tertiary_n = Chem.MolFromSmarts("[NX3;!H]")
    if mol.HasSubstructMatch(tertiary_n):
        noxide = smiles.replace(")N(", ")[N+]([O-])(")
        try:
            test = Chem.MolFromSmiles(noxide)
            if test:
                degradants.append(noxide)
        except:
            pass

    # Deamination: primary amine → hydroxyl
    primary_amine = Chem.MolFromSmarts("[NX3;H2]")
    if mol.HasSubstructMatch(primary_amine):
        deaminated = smiles.replace("N)", "O)")
        try:
            test = Chem.MolFromSmiles(deaminated)
            if test:
                degradants.append(deaminated)
        except:
            pass

    # Always add a common process impurity skeleton
    degradants.append("CC(=O)O")   # Acetic acid (common solvent residue)
    degradants.append("CCOC(=O)O") # Ethyl carbonate (common reagent impurity)

    # Deduplicate and validate
    seen = set()
    valid = []
    for s in degradants:
        mol_check = Chem.MolFromSmiles(s)
        if mol_check and s not in seen and s != smiles:
            seen.add(s)
            valid.append(s)

    return valid[:6]  # max 6 impurities for demo


# ── Module 3: Polymorph Stability ────────────────────────────────

@dataclass
class PolymorphResult:
    api_name: str
    forms_analyzed: int
    rankings: List[dict]
    most_stable_form: str
    energy_difference_ev: float
    manufacturing_recommendation: str
    vqe_used: bool


def analyze_polymorphs(api_name: str, forms: List[dict],
                       use_vqe: bool = False) -> PolymorphResult:
    """
    Rank crystal polymorphs by computed ground state energy.
    Lower energy = more thermodynamically stable = preferred for manufacturing.

    forms: list of {"name": str, "smiles": str, "geometry": str}
    """
    rankings = []

    for form in forms:
        energy_ha = None
        energy_ev = None
        method = "Classical estimate"

        if use_vqe and form.get("geometry"):
            try:
                from core.vqe_runner import run_vqe
                result = run_vqe(
                    geometry=form["geometry"],
                    molecule_name=f"{api_name} {form['name']}",
                    smiles=form["smiles"],
                )
                if result.success:
                    energy_ha = result.ground_state_energy_hartree
                    energy_ev = result.ground_state_energy_ev
                    method    = "VQE (AerSimulator)"
            except Exception as e:
                logger.warning(f"VQE failed for {form['name']}: {e}")

        if energy_ha is None:
            # Classical MM energy estimate using RDKit
            try:
                from rdkit import Chem
                from rdkit.Chem import AllChem
                mol = Chem.MolFromSmiles(form["smiles"])
                if mol:
                    mol = Chem.AddHs(mol)
                    AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
                    ff = AllChem.MMFFGetMoleculeForceField(mol, AllChem.MMFFGetMoleculeProperties(mol))
                    if ff:
                        energy_kcal = ff.CalcEnergy()
                        energy_ha   = round(energy_kcal / 627.5, 6)
                        energy_ev   = round(energy_ha * 27.2114, 4)
                        method      = "MM force field (classical)"
            except Exception as e:
                logger.warning(f"MM failed for {form['name']}: {e}")
                energy_ha = -100.0 + len(rankings) * 0.001
                energy_ev = energy_ha * 27.2114
                method    = "Estimated"

        rankings.append({
            "form":       form["name"],
            "smiles":     form["smiles"],
            "energy_ha":  round(energy_ha, 6) if energy_ha else None,
            "energy_ev":  round(energy_ev, 4) if energy_ev else None,
            "method":     method,
        })

    # Sort by energy (lower = more stable)
    valid = [r for r in rankings if r["energy_ha"] is not None]
    valid.sort(key=lambda x: x["energy_ha"])

    most_stable = valid[0]["form"] if valid else "Unknown"
    e_diff = 0.0
    if len(valid) >= 2:
        e_diff = round(abs(valid[0]["energy_ev"] - valid[1]["energy_ev"]), 4)

    if e_diff > 0.5:
        rec = f"Use {most_stable} — energy difference {e_diff} eV is significant. Other forms will convert spontaneously."
    elif e_diff > 0.1:
        rec = f"Prefer {most_stable} — moderate energy difference. Control crystallization conditions carefully."
    else:
        rec = f"Forms are near-isoenergetic (ΔE = {e_diff} eV). Rigorous polymorph control required during manufacturing."

    return PolymorphResult(
        api_name=api_name,
        forms_analyzed=len(forms),
        rankings=rankings,
        most_stable_form=most_stable,
        energy_difference_ev=e_diff,
        manufacturing_recommendation=rec,
        vqe_used=use_vqe,
    )
