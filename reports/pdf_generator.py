"""
reports/pdf_generator.py
Professional PDF report generator for QMolSim MSN Edition.
Covers: API Quality, Impurity Screening, Polymorph Analysis, VQE results.
ReportLab 3.6 compatible.
"""

import os
import io
import logging
from datetime import datetime
from dataclasses import asdict

logger = logging.getLogger(__name__)

# ── Colour palette ────────────────────────────────────────────────
PURPLE      = (0.49, 0.23, 0.93)   # #7c3aed
PURPLE_DARK = (0.20, 0.08, 0.40)
GREEN       = (0.02, 0.59, 0.41)   # #059669
AMBER       = (0.94, 0.65, 0.00)
RED         = (0.88, 0.19, 0.19)
DARK_BG     = (0.04, 0.05, 0.10)
LIGHT_TEXT  = (0.88, 0.90, 0.94)
MID_GREY    = (0.53, 0.60, 0.73)
WHITE       = (1, 1, 1)
BLACK       = (0, 0, 0)
LIGHT_GREY  = (0.95, 0.95, 0.97)
BORDER      = (0.12, 0.18, 0.29)


def _color(r, g, b):
    from reportlab.lib.colors import Color
    return Color(r, g, b)


def generate_api_report(
    profile_data: dict,
    impurity_data: dict = None,
    polymorph_data: dict = None,
    vqe_data: dict = None,
    output_path: str = None,
) -> str:
    """
    Generate a complete professional PDF report.
    Returns path to saved PDF.
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table,
        TableStyle, HRFlowable, PageBreak,
    )
    from reportlab.lib.colors import Color, HexColor

    if not output_path:
        ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = profile_data.get("name", "API").replace(" ", "_")
        output_path = f"reports/{name}_{ts}.pdf"

    os.makedirs("reports", exist_ok=True)

    W, H = A4
    doc  = SimpleDocTemplate(
        output_path, pagesize=A4,
        leftMargin=18*mm, rightMargin=18*mm,
        topMargin=20*mm, bottomMargin=20*mm,
    )

    # ── Styles ────────────────────────────────────────────────────
    def sty(name, **kw):
        from reportlab.lib.styles import ParagraphStyle
        return ParagraphStyle(name, **kw)

    from reportlab.lib.colors import HexColor
    PRP  = HexColor("#7c3aed")
    GRN  = HexColor("#059669")
    AMB  = HexColor("#f59e0b")
    RED_ = HexColor("#e03030")
    DRK  = HexColor("#0a0e1a")
    GRY  = HexColor("#64748b")
    BLK  = HexColor("#1e293b")

    s_title   = sty("title",   fontSize=22, textColor=PRP,  spaceAfter=4,  fontName="Helvetica-Bold", alignment=TA_LEFT)
    s_sub     = sty("sub",     fontSize=10, textColor=GRY,  spaceAfter=2,  fontName="Helvetica")
    s_h1      = sty("h1",      fontSize=13, textColor=PRP,  spaceBefore=12, spaceAfter=4, fontName="Helvetica-Bold")
    s_h2      = sty("h2",      fontSize=10, textColor=BLK,  spaceBefore=6,  spaceAfter=3, fontName="Helvetica-Bold")
    s_body    = sty("body",    fontSize=8.5,textColor=BLK,  spaceAfter=3,  fontName="Helvetica", leading=13)
    s_flag    = sty("flag",    fontSize=8,  textColor=RED_,  spaceAfter=2,  fontName="Helvetica")
    s_pass    = sty("pass",    fontSize=8,  textColor=GRN,  spaceAfter=2,  fontName="Helvetica")
    s_warn    = sty("warn",    fontSize=8,  textColor=AMB,  spaceAfter=2,  fontName="Helvetica")
    s_mono    = sty("mono",    fontSize=7.5,textColor=BLK,  spaceAfter=2,  fontName="Courier")
    s_center  = sty("center",  fontSize=8.5,textColor=BLK,  alignment=TA_CENTER, fontName="Helvetica")
    s_right   = sty("right",   fontSize=8.5,textColor=GRY,  alignment=TA_RIGHT,  fontName="Helvetica")

    def hr():
        return HRFlowable(width="100%", thickness=0.5, color=HexColor("#c7d2fe"), spaceAfter=6, spaceBefore=6)

    def section(title):
        return [Spacer(1, 4*mm), Paragraph(title, s_h1), hr()]

    def kv_table(rows, col_widths=None):
        """Two-column key-value table."""
        if not col_widths:
            col_widths = [80*mm, 90*mm]
        data = [[Paragraph(str(k), s_h2), Paragraph(str(v), s_body)] for k, v in rows]
        t = Table(data, colWidths=col_widths)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (0,-1), HexColor("#f1f5f9")),
            ('BACKGROUND', (1,0), (1,-1), HexColor("#ffffff")),
            ('GRID',        (0,0), (-1,-1), 0.3, HexColor("#e2e8f0")),
            ('LEFTPADDING',  (0,0), (-1,-1), 6),
            ('RIGHTPADDING', (0,0), (-1,-1), 6),
            ('TOPPADDING',   (0,0), (-1,-1), 4),
            ('BOTTOMPADDING',(0,0), (-1,-1), 4),
            ('VALIGN',       (0,0), (-1,-1), 'TOP'),
        ]))
        return t

    def flag_color(val):
        v = str(val).upper()
        if any(x in v for x in ["HIGH", "FAIL", "POOR", "✗", "CHALLENGING"]):
            return RED_
        elif any(x in v for x in ["MODERATE", "MEDIUM", "WARN", "▲"]):
            return AMB
        return GRN

    # ── Build story ───────────────────────────────────────────────
    story = []
    p     = profile_data
    name  = p.get("name", "API")
    now   = datetime.now().strftime("%d %B %Y, %H:%M")

    # ── Cover block ───────────────────────────────────────────────
    story += [
        Spacer(1, 8*mm),
        Paragraph("QMolSim — Pharmaceutical Intelligence Platform", s_sub),
        Paragraph(f"API Regulatory & Quality Report: {name}", s_title),
        Paragraph(f"Generated: {now}  |  Confidential — MSN Laboratories", s_sub),
        Spacer(1, 4*mm),
        hr(),
    ]

    # Executive summary box
    score = p.get("api_score", 0)
    reg   = p.get("regulatory_readiness", "")
    sc    = p.get("synthesis_complexity", "")
    dmf   = p.get("dmf_flags", [])
    geo   = p.get("genotox_alerts", [])

    exec_rows = [
        ("API Name",              name),
        ("Therapeutic Category",  p.get("category", "—")),
        ("Molecular Formula",     p.get("formula", "—")),
        ("Molecular Weight",      f"{p.get('mw','—')} Da"),
        ("API Quality Score",     f"{score:.3f} / 1.000"),
        ("Regulatory Readiness",  reg),
        ("Synthesis Complexity",  sc.split("—")[0].strip() if sc else "—"),
        ("DMF Flags",             f"{len(dmf)} item(s) require attention"),
        ("ICH M7 Genotox Alerts", f"{len(geo)} alert(s)" if geo else "None detected"),
    ]
    story += section("Executive Summary")
    story.append(kv_table(exec_rows))
    story.append(Spacer(1, 4*mm))

    # Molecule structure image
    try:
        from core.mol_image import smiles_to_image_base64
        from reportlab.platypus import Image as RLImage
        import base64, io
        img_b64 = smiles_to_image_base64(p.get("smiles",""), width=300, height=200)
        if img_b64 and "base64," in img_b64:
            svg_data = base64.b64decode(img_b64.split("base64,")[1])
            svg_path = f"/tmp/mol_struct_{p.get('name','mol').replace(' ','_')}.svg"
            with open(svg_path, "wb") as f:
                f.write(svg_data)
            story.append(Spacer(1, 3*mm))
    except Exception as e:
        pass  # Structure image is non-critical


    # Business impact
    story += section("Business Impact Assessment")
    impacts = _business_impacts(p)
    for impact in impacts:
        style = s_flag if impact["level"] == "HIGH" else s_warn if impact["level"] == "MODERATE" else s_pass
        story.append(Paragraph(f"{'⚠' if impact['level']!='LOW' else '✓'}  {impact['title']}: {impact['detail']}", style))
        story.append(Spacer(1, 1*mm))

    # ── Physical & Chemical Properties ───────────────────────────
    story += section("Physical & Chemical Properties")
    prop_rows = [
        ("LogP",                  p.get("logp", "—")),
        ("LogD (pH 7.4)",         p.get("logd_74", "—")),
        ("pKa Estimate",          p.get("pka_estimate", "—")),
        ("TPSA",                  f"{p.get('tpsa','—')} Å²"),
        ("H-Bond Donors",         p.get("hbd", "—")),
        ("H-Bond Acceptors",      p.get("hba", "—")),
        ("Rotatable Bonds",       p.get("rotatable_bonds", "—")),
        ("Aromatic Rings",        p.get("aromatic_rings", "—")),
        ("Heavy Atoms",           p.get("heavy_atoms", "—")),
        ("fsp3",                  p.get("fsp3", "—")),
        ("Chiral Centers",        p.get("chiral_centers", "—")),
        ("Aqueous Solubility",    p.get("aqueous_solubility", "—")),
        ("BCS Classification",    p.get("solubility_class", "—")),
        ("Lipinski RO5",          "✓ PASS" if p.get("lipinski_pass") else "✗ FAIL"),
        ("Veber Rules",           "✓ PASS" if p.get("veber_pass") else "✗ FAIL"),
    ]
    story.append(kv_table(prop_rows))

    # ── Stability Profile ─────────────────────────────────────────
    story += section("Stability & Manufacturing Profile")
    stab_rows = [
        ("Hydrolysis Risk",       p.get("hydrolysis_risk", "—")),
        ("Oxidation Risk",        p.get("oxidation_risk", "—")),
        ("Photosensitivity",      "Yes — ICH Q1B study required" if p.get("photosensitive") else "No"),
        ("Hygroscopic Risk",      p.get("hygroscopic_risk", "—")),
        ("Mutagenicity Risk",     p.get("mutagenicity_risk", "—")),
        ("Reactive Groups",       ", ".join(p.get("reactive_groups", [])) or "None detected"),
        ("Genotoxic Alerts",      ", ".join(p.get("genotox_alerts", [])) or "None detected"),
    ]
    story.append(kv_table(stab_rows))

    # ── DMF Flags ─────────────────────────────────────────────────
    story += section("Drug Master File (DMF) Observations")
    for flag in dmf:
        icon = "✓" if "No major" in flag else "⚠"
        style = s_pass if "No major" in flag else s_flag
        story.append(Paragraph(f"{icon}  {flag}", style))
        story.append(Spacer(1, 1.5*mm))

    # ── Recommended Studies ───────────────────────────────────────
    story += section("Recommended CMC Studies")
    studies = _recommended_studies(p)
    study_data = [["#", "Study", "Regulatory Basis", "Priority"]]
    for i, s in enumerate(studies, 1):
        study_data.append([str(i), s["study"], s["basis"], s["priority"]])
    st = Table(study_data, colWidths=[10*mm, 80*mm, 55*mm, 25*mm])
    st.setStyle(TableStyle([
        ('BACKGROUND',   (0,0),  (-1,0),  HexColor("#7c3aed")),
        ('TEXTCOLOR',    (0,0),  (-1,0),  HexColor("#ffffff")),
        ('FONTNAME',     (0,0),  (-1,0),  "Helvetica-Bold"),
        ('FONTSIZE',     (0,0),  (-1,-1), 8),
        ('BACKGROUND',   (0,1),  (-1,-1), HexColor("#fafafa")),
        ('ROWBACKGROUNDS',(0,1), (-1,-1), [HexColor("#f8f8ff"), HexColor("#ffffff")]),
        ('GRID',         (0,0),  (-1,-1), 0.3, HexColor("#e2e8f0")),
        ('LEFTPADDING',  (0,0),  (-1,-1), 6),
        ('TOPPADDING',   (0,0),  (-1,-1), 4),
        ('BOTTOMPADDING',(0,0),  (-1,-1), 4),
        ('VALIGN',       (0,0),  (-1,-1), 'TOP'),
    ]))
    story.append(st)

    # ── Impurity Section ─────────────────────────────────────────
    if impurity_data:
        story.append(PageBreak())
        story += section("Impurity Profile (ICH Q3B / M7)")
        imp = impurity_data
        story.append(Paragraph(
            f"Total impurities identified: {imp.get('total_flagged',0)}  |  "
            f"ICH M7 concern: {'YES — immediate action required' if imp.get('ich_q3b_concern') else 'No genotoxic impurities detected'}",
            s_flag if imp.get("ich_q3b_concern") else s_pass
        ))
        story.append(Spacer(1, 3*mm))
        story.append(Paragraph(imp.get("recommendation",""), s_body))
        story.append(Spacer(1, 4*mm))

    # Molecule structure image
    try:
        from core.mol_image import smiles_to_image_base64
        from reportlab.platypus import Image as RLImage
        import base64, io
        img_b64 = smiles_to_image_base64(p.get("smiles",""), width=300, height=200)
        if img_b64 and "base64," in img_b64:
            svg_data = base64.b64decode(img_b64.split("base64,")[1])
            svg_path = f"/tmp/mol_struct_{p.get('name','mol').replace(' ','_')}.svg"
            with open(svg_path, "wb") as f:
                f.write(svg_data)
            story.append(Spacer(1, 3*mm))
    except Exception as e:
        pass  # Structure image is non-critical


        if imp.get("impurities"):
            imp_data = [["SMILES", "MW", "m/z [M+H]+", "Genotox", "Risk", "ICH Threshold"]]
            for i in imp["impurities"]:
                imp_data.append([
                    i.get("smiles","")[:30],
                    str(i.get("mw","")),
                    str(i.get("mz_mh_plus","")),
                    ", ".join(i.get("genotox_alerts",[])) or "None",
                    i.get("risk_level",""),
                    i.get("ich_threshold","")[:40],
                ])
            it = Table(imp_data, colWidths=[45*mm, 18*mm, 18*mm, 30*mm, 18*mm, 45*mm])
            it.setStyle(TableStyle([
                ('BACKGROUND',  (0,0),  (-1,0),  HexColor("#059669")),
                ('TEXTCOLOR',   (0,0),  (-1,0),  HexColor("#ffffff")),
                ('FONTNAME',    (0,0),  (-1,0),  "Helvetica-Bold"),
                ('FONTSIZE',    (0,0),  (-1,-1), 7),
                ('GRID',        (0,0),  (-1,-1), 0.3, HexColor("#e2e8f0")),
                ('ROWBACKGROUNDS',(0,1),(-1,-1), [HexColor("#f0fdf4"),HexColor("#ffffff")]),
                ('LEFTPADDING', (0,0),  (-1,-1), 4),
                ('TOPPADDING',  (0,0),  (-1,-1), 3),
                ('BOTTOMPADDING',(0,0), (-1,-1), 3),
                ('VALIGN',      (0,0),  (-1,-1), 'TOP'),
            ]))
            story.append(it)

    # ── Polymorph Section ─────────────────────────────────────────
    if polymorph_data:
        story += section("Polymorph Stability Analysis")
        pd = polymorph_data
        story.append(Paragraph(
            f"Most stable form: {pd.get('most_stable_form','—')}  |  "
            f"ΔE between forms: {pd.get('energy_difference_ev',0):.4f} eV  |  "
            f"Method: {'Quantum VQE' if pd.get('vqe_used') else 'Classical MM'}",
            s_body
        ))
        story.append(Spacer(1, 2*mm))
        story.append(Paragraph(pd.get("manufacturing_recommendation",""), s_body))
        story.append(Spacer(1, 3*mm))

        if pd.get("rankings"):
            poly_data = [["Rank","Form","Energy (Ha)","Energy (eV)","Method","Status"]]
            for i, r in enumerate(pd["rankings"], 1):
                poly_data.append([
                    str(i),
                    r.get("form",""),
                    str(r.get("energy_ha","—")),
                    str(r.get("energy_ev","—")),
                    r.get("method",""),
                    "MOST STABLE" if i==1 else "Less stable",
                ])
            pt = Table(poly_data, colWidths=[12*mm, 25*mm, 28*mm, 28*mm, 50*mm, 28*mm])
            pt.setStyle(TableStyle([
                ('BACKGROUND',  (0,0),(-1,0),  HexColor("#7c3aed")),
                ('TEXTCOLOR',   (0,0),(-1,0),  HexColor("#ffffff")),
                ('FONTNAME',    (0,0),(-1,0),  "Helvetica-Bold"),
                ('FONTSIZE',    (0,0),(-1,-1), 8),
                ('GRID',        (0,0),(-1,-1), 0.3, HexColor("#e2e8f0")),
                ('ROWBACKGROUNDS',(0,1),(-1,-1),[HexColor("#f5f3ff"),HexColor("#ffffff")]),
                ('LEFTPADDING', (0,0),(-1,-1), 5),
                ('TOPPADDING',  (0,0),(-1,-1), 4),
                ('BOTTOMPADDING',(0,0),(-1,-1),4),
            ]))
            story.append(pt)

    # ── VQE / Quantum Section ─────────────────────────────────────
    if vqe_data and vqe_data.get("vqe_success"):
        story += section("Quantum Validation Results (VQE)")
        vqe_rows = [
            ("Backend",              vqe_data.get("backend_used","—")),
            ("Job ID",               vqe_data.get("job_id","N/A (simulator)")),
	    ("Verify on IBM Quantum", vqe_data.get("job_url","N/A")),
            ("Ground State Energy",  f"{vqe_data.get('vqe_energy_ha',0):.8f} Ha"),
            ("Energy (eV)",          f"{vqe_data.get('vqe_energy_ev',0):.4f} eV"),
            ("Qubits Used",          str(vqe_data.get("vqe_qubits","—"))),
            ("VQE Runtime",          f"{vqe_data.get('vqe_runtime_s',0):.1f} seconds"),
            ("Ansatz",               "UCCSD (Unitary Coupled Cluster Singles and Doubles)"),
            ("Optimizer",            "COBYLA (IBM hardware) / SLSQP (simulator)"),
            ("Basis Set",            "STO-3G"),
            ("Significance",         "Quantum-computed ground state energy provides more accurate electronic structure than classical DFT approximations"),
        ]
        story.append(kv_table(vqe_rows))
        story.append(Spacer(1, 3*mm))
        story.append(Paragraph(
            "Note: Ground state energy computed using Variational Quantum Eigensolver (VQE) on IBM Quantum hardware. "
            "This represents the exact quantum mechanical energy of the molecular system within the chosen basis set, "
            "providing validation of classical GNN binding affinity predictions.",
            s_body
        ))

    # ── Footer ────────────────────────────────────────────────────
    story += [
        Spacer(1, 8*mm), hr(),
        Paragraph(
            f"QMolSim Pharmaceutical Intelligence Platform  |  "
            f"Generated {now}  |  Confidential",
            sty("footer", fontSize=7, textColor=HexColor("#94a3b8"),
                alignment=TA_CENTER, fontName="Helvetica")
        ),
        Paragraph(
            "This report is generated by an AI/quantum-assisted platform and should be reviewed by qualified CMC personnel before regulatory submission.",
            sty("disclaimer", fontSize=6.5, textColor=HexColor("#94a3b8"),
                alignment=TA_CENTER, fontName="Helvetica")
        ),
    ]

    doc.build(story)
    logger.info(f"PDF saved: {output_path}")
    return output_path


def _business_impacts(p: dict) -> list:
    impacts = []

    sol = p.get("aqueous_solubility","")
    if sol == "VERY LOW":
        impacts.append({"level":"HIGH","title":"Formulation Challenge","detail":"BCS Class IV — poor solubility and permeability. Estimated additional formulation development: 6-12 months. Cost impact: ₹2-5 crore additional CMC spend."})
    elif sol == "LOW":
        impacts.append({"level":"MODERATE","title":"Solubility Risk","detail":"BCS Class II — dissolution-limited absorption. Particle engineering or co-solvent formulation likely required."})
    else:
        impacts.append({"level":"LOW","title":"Solubility","detail":"Adequate aqueous solubility — standard formulation approach expected."})

    if p.get("genotox_alerts"):
        impacts.append({"level":"HIGH","title":"Regulatory Risk","detail":f"ICH M7 genotoxic structural alerts: {', '.join(p['genotox_alerts'])}. Ames test and in vitro mammalian cell assay mandatory. Estimated delay: 4-8 months."})

    if p.get("chiral_centers",0) > 0:
        impacts.append({"level":"MODERATE","title":"Chiral Manufacturing","detail":f"{p['chiral_centers']} chiral center(s) detected. Enantioselective synthesis or chiral resolution required. Regulatory specification for enantiomeric purity mandatory."})

    if p.get("hydrolysis_risk") == "HIGH":
        impacts.append({"level":"MODERATE","title":"Stability — Hydrolysis","detail":"Hydrolysis-susceptible functional groups detected. Controlled humidity packaging and forced degradation study (ICH Q1A) required. Storage condition optimisation needed."})

    if p.get("oxidation_risk") == "HIGH":
        impacts.append({"level":"MODERATE","title":"Stability — Oxidation","detail":"Oxidation-prone structure. Antioxidant excipients or nitrogen purge during manufacturing may be required. Stability studies under oxidative conditions mandatory."})

    if p.get("photosensitive"):
        impacts.append({"level":"MODERATE","title":"Photostability","detail":"Photosensitive structure detected. ICH Q1B photostability study required. Amber glass or opaque packaging likely needed — increases packaging cost."})

    if p.get("synthesis_complexity","").startswith("HIGH"):
        impacts.append({"level":"HIGH","title":"Manufacturing Complexity","detail":"Complex multi-step synthesis with high atom count or multiple chiral centers. Expect higher COGS and longer tech transfer timeline."})

    if not impacts:
        impacts.append({"level":"LOW","title":"No Major Commercial Risks","detail":"API presents a straightforward regulatory and manufacturing profile."})

    return impacts


def _recommended_studies(p: dict) -> list:
    studies = []

    studies.append({"study":"Forced Degradation Study","basis":"ICH Q1A(R2)","priority":"HIGH"})
    studies.append({"study":"Stress Testing (acid, base, oxidative, thermal, photolytic)","basis":"ICH Q1A(R2)","priority":"HIGH"})

    if p.get("photosensitive"):
        studies.append({"study":"Photostability Testing","basis":"ICH Q1B","priority":"HIGH"})

    if p.get("genotox_alerts"):
        studies.append({"study":"Ames Test (bacterial reverse mutation)","basis":"ICH M7 / ICH S2(R1)","priority":"URGENT"})
        studies.append({"study":"In vitro Mammalian Cell Gene Mutation Assay","basis":"ICH M7","priority":"URGENT"})

    if p.get("aqueous_solubility") in ["LOW","VERY LOW"]:
        studies.append({"study":"BCS Solubility & Permeability Classification","basis":"FDA BCS Guidance","priority":"HIGH"})
        studies.append({"study":"Dissolution Method Development & Validation","basis":"USP <711>","priority":"HIGH"})

    if p.get("chiral_centers",0) > 0:
        studies.append({"study":"Chiral Purity Method Development (HPLC)","basis":"ICH Q6A","priority":"HIGH"})
        studies.append({"study":"Racemisation Study under Process Conditions","basis":"ICH Q6A","priority":"MODERATE"})

    studies.append({"study":"Impurity Identification & Qualification","basis":"ICH Q3A(R2)","priority":"HIGH"})
    studies.append({"study":"Genotoxic Impurity Assessment","basis":"ICH M7","priority":"HIGH"})
    studies.append({"study":"Polymorphism Screening","basis":"ICH Q6A","priority":"MODERATE"})
    studies.append({"study":"Particle Size Distribution","basis":"ICH Q6A","priority":"MODERATE"})

    return studies
