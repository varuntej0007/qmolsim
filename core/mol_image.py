"""
core/mol_image.py
Generate 2D molecule structure images using RDKit.
Returns base64 PNG for embedding in HTML/PDF.
"""

import base64
import os
import logging
from io import BytesIO

logger = logging.getLogger(__name__)


def smiles_to_image_base64(
    smiles: str,
    width: int = 400,
    height: int = 300,
    bg_color: tuple = (18, 24, 39),
    atom_color: tuple = (224, 230, 240),
) -> str:
    """
    Convert SMILES to base64 PNG image.
    Returns base64 string for use in <img src="data:image/png;base64,...">
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw
        from rdkit.Chem.Draw import rdMolDraw2D

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return _placeholder_b64(width, height)

        from rdkit.Chem import rdDepictor
        rdDepictor.Compute2DCoords(mol)

        drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
        drawer.drawOptions().backgroundColour = (
            bg_color[0]/255, bg_color[1]/255, bg_color[2]/255, 1.0
        )
        drawer.drawOptions().padding = 0.15
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()

        # Return SVG as base64
        b64 = base64.b64encode(svg.encode()).decode()
        return f"data:image/svg+xml;base64,{b64}"

    except Exception as e:
        logger.warning(f"Image generation failed for {smiles}: {e}")
        return _placeholder_b64(width, height)


def smiles_to_image_file(
    smiles: str,
    output_path: str,
    width: int = 400,
    height: int = 300,
) -> str:
    """Save molecule image to file. Returns path."""
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw
        from rdkit.Chem.Draw import rdMolDraw2D
        from rdkit.Chem import rdDepictor

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        rdDepictor.Compute2DCoords(mol)

        drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
        drawer.drawOptions().padding = 0.15
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(svg)

        return output_path

    except Exception as e:
        logger.warning(f"Image file generation failed: {e}")
        return None


def _placeholder_b64(width: int, height: int) -> str:
    """Return a simple placeholder SVG for invalid SMILES."""
    svg = f'''<svg width="{width}" height="{height}"
        xmlns="http://www.w3.org/2000/svg">
        <rect width="100%" height="100%" fill="#111827"/>
        <text x="50%" y="50%" text-anchor="middle"
            fill="#4a6a9a" font-size="14" font-family="sans-serif">
            Structure unavailable
        </text>
    </svg>'''
    b64 = base64.b64encode(svg.encode()).decode()
    return f"data:image/svg+xml;base64,{b64}"
