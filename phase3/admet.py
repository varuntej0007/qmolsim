"""
phase3/admet.py
ADMET property calculator using RDKit molecular descriptors.
Covers the 5 filters that kill drug candidates in clinical trials.

Rules used:
  Lipinski RO5  — oral bioavailability
  Veber         — oral bioavailability (rotatable bonds, TPSA)
  PAINS         — pan-assay interference (false positives)
  BBB           — blood-brain barrier penetration
  Toxicity flags— structural alerts for hepatotoxicity, mutagenicity
"""

import logging
from dataclasses import dataclass, field
from typing import List

logger = logging.getLogger(__name__)


@dataclass
class ADMETResult:
    smiles: str
    mw: float                    # molecular weight (Da)
    logp: float                  # lipophilicity
    hbd: int                     # H-bond donors
    hba: int                     # H-bond acceptors
    tpsa: float                  # topological polar surface area (Å²)
    rotatable_bonds: int
    aromatic_rings: int
    heavy_atoms: int
    fsp3: float                  # fraction sp3 carbons
    qed_rdkit: float             # drug-likeness score

    # Rule-based filters
    lipinski_pass: bool
    veber_pass: bool
    bbb_likely: bool
    gi_absorption: str           # HIGH / LOW
    pains_alerts: int
    tox_alerts: List[str] = field(default_factory=list)

    # Final verdict
    admet_score: float = 0.0     # 0-1 composite
    admet_verdict: str = ""


# Structural toxicity SMARTS alerts
TOX_SMARTS = {
    "Michael_acceptor":   "[CX3]=[CX3][CX3]=[O,S,N]",
    "Aldehyde":           "[CX3H1](=O)",
    "Epoxide":            "[OX2r3]",
    "Nitro":              "[$([NX3](=O)=O),$([NX3+](=O)[O-])]",
    "Quinone":            "O=C1C=CC(=O)C=C1",
    "Halogenated_chain":  "[Cl,Br,I][CX4]",
    "Azo":                "[NX2]=[NX2]",
    "Thiol":              "[SX2H]",
}

# PAINS SMARTS (subset — key patterns)
PAINS_SMARTS = {
    "Catechol":           "c1ccc(O)c(O)c1",
    "Rhodanine":          "O=C1CSC(=S)N1",
    "Quinone_methide":    "O=Cc1ccccc1O",
    "Enamine":            "[NX3][CX3]=[CX3]",
}


def calculate_admet(smiles: str) -> ADMETResult:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors, QED, Lipinski
    from rdkit.Chem import FilterCatalog

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    # ── Basic descriptors ─────────────────────────────────────────
    mw   = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd  = Lipinski.NumHDonors(mol)
    hba  = Lipinski.NumHAcceptors(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    rot  = rdMolDescriptors.CalcNumRotatableBonds(mol)
    arom = rdMolDescriptors.CalcNumAromaticRings(mol)
    ha   = mol.GetNumHeavyAtoms()
    qed  = float(QED.qed(mol))

    # fsp3: fraction of sp3 carbons
    sp3_c = sum(1 for a in mol.GetAtoms()
                if a.GetAtomicNum() == 6
                and a.GetHybridization().name == 'SP3')
    total_c = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 6)
    fsp3 = sp3_c / total_c if total_c > 0 else 0.0

    # ── Lipinski Rule of 5 ────────────────────────────────────────
    # Oral bioavailability: MW≤500, logP≤5, HBD≤5, HBA≤10
    lipinski_pass = (mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10)

    # ── Veber rules ───────────────────────────────────────────────
    # Oral bioavailability: rotatable bonds ≤ 10, TPSA ≤ 140
    veber_pass = (rot <= 10 and tpsa <= 140)

    # ── GI absorption ─────────────────────────────────────────────
    gi_absorption = "HIGH" if (tpsa <= 140 and rot <= 10) else "LOW"

    # ── BBB penetration ───────────────────────────────────────────
    # CNS drugs: MW<400, logP 1-3, TPSA<90, HBD≤3
    bbb_likely = (mw < 400 and 1 <= logp <= 3 and tpsa < 90 and hbd <= 3)

    # ── Toxicity alerts ───────────────────────────────────────────
    tox_alerts = []
    for name, smarts in TOX_SMARTS.items():
        pattern = Chem.MolFromSmarts(smarts)
        if pattern and mol.HasSubstructMatch(pattern):
            tox_alerts.append(name)

    # ── PAINS alerts ──────────────────────────────────────────────
    pains_count = 0
    for name, smarts in PAINS_SMARTS.items():
        pattern = Chem.MolFromSmarts(smarts)
        if pattern and mol.HasSubstructMatch(pattern):
            pains_count += 1

    # ── Composite ADMET score (0-1) ───────────────────────────────
    score = 0.0
    score += 0.25 * float(lipinski_pass)
    score += 0.20 * float(veber_pass)
    score += 0.25 * qed
    score += 0.15 * (1.0 - min(len(tox_alerts), 3) / 3.0)
    score += 0.15 * (1.0 - min(pains_count, 2) / 2.0)

    # Verdict
    if score >= 0.75:
        verdict = "★ EXCELLENT"
    elif score >= 0.55:
        verdict = "◆ ACCEPTABLE"
    elif score >= 0.35:
        verdict = "▲ MARGINAL"
    else:
        verdict = "✗ POOR"

    return ADMETResult(
        smiles=smiles,
        mw=round(mw, 2),
        logp=round(logp, 3),
        hbd=hbd, hba=hba,
        tpsa=round(tpsa, 2),
        rotatable_bonds=rot,
        aromatic_rings=arom,
        heavy_atoms=ha,
        fsp3=round(fsp3, 3),
        qed_rdkit=round(qed, 3),
        lipinski_pass=lipinski_pass,
        veber_pass=veber_pass,
        bbb_likely=bbb_likely,
        gi_absorption=gi_absorption,
        pains_alerts=pains_count,
        tox_alerts=tox_alerts,
        admet_score=round(score, 3),
        admet_verdict=verdict,
    )


def print_admet(result: ADMETResult):
    sep = "=" * 58
    print(f"\n{sep}")
    print(f"  ADMET Profile — {result.smiles[:45]}")
    print(sep)
    print(f"  MW              : {result.mw} Da")
    print(f"  LogP            : {result.logp}")
    print(f"  H-Bond Donors   : {result.hbd}")
    print(f"  H-Bond Acceptors: {result.hba}")
    print(f"  TPSA            : {result.tpsa} Å²")
    print(f"  Rotatable bonds : {result.rotatable_bonds}")
    print(f"  Aromatic rings  : {result.aromatic_rings}")
    print(f"  fsp3            : {result.fsp3}")
    print(f"  QED             : {result.qed_rdkit}")
    print("-" * 58)
    print(f"  Lipinski RO5    : {'✓ PASS' if result.lipinski_pass else '✗ FAIL'}")
    print(f"  Veber           : {'✓ PASS' if result.veber_pass else '✗ FAIL'}")
    print(f"  GI absorption   : {result.gi_absorption}")
    print(f"  BBB penetration : {'Likely' if result.bbb_likely else 'Unlikely'}")
    print(f"  PAINS alerts    : {result.pains_alerts}")
    print(f"  Tox alerts      : {', '.join(result.tox_alerts) if result.tox_alerts else 'None'}")
    print("-" * 58)
    print(f"  ADMET score     : {result.admet_score:.3f}")
    print(f"  Verdict         : {result.admet_verdict}")
    print(f"{sep}\n")
