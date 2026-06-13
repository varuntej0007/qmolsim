"""
graph.py
Converts SMILES → molecular graph with atom and bond features.
Pure numpy — no PyTorch required.
"""

import numpy as np
import logging
from dataclasses import dataclass
from typing import List, Tuple

logger = logging.getLogger(__name__)

# Atom feature encoding
ATOM_TYPES     = ['C','N','O','S','F','P','Cl','Br','I','H','Li','Na','other']
HYBRIDIZATION  = ['S','SP','SP2','SP3','SP3D','SP3D2','other']
DEGREES        = [0,1,2,3,4,5,6]
CHARGES        = [-2,-1,0,1,2,3]
NUM_HS         = [0,1,2,3,4]


def one_hot(val, vocab: list) -> np.ndarray:
    vec = np.zeros(len(vocab), dtype=np.float32)
    idx = vocab.index(val) if val in vocab else len(vocab) - 1
    vec[idx] = 1.0
    return vec


def atom_features(atom) -> np.ndarray:
    """39-dimensional atom feature vector."""
    from rdkit.Chem import rdchem
    hyb_map = {
        rdchem.HybridizationType.S:     'S',
        rdchem.HybridizationType.SP:    'SP',
        rdchem.HybridizationType.SP2:   'SP2',
        rdchem.HybridizationType.SP3:   'SP3',
        rdchem.HybridizationType.SP3D:  'SP3D',
        rdchem.HybridizationType.SP3D2: 'SP3D2',
    }
    hyb_str = hyb_map.get(atom.GetHybridization(), 'other')

    feats = np.concatenate([
        one_hot(atom.GetSymbol(), ATOM_TYPES),           # 13
        one_hot(atom.GetDegree(), DEGREES),              # 7
        one_hot(atom.GetFormalCharge(), CHARGES),        # 6
        one_hot(atom.GetTotalNumHs(), NUM_HS),           # 5
        one_hot(hyb_str, HYBRIDIZATION),                 # 7
        [float(atom.GetIsAromatic())],                   # 1
    ])
    return feats.astype(np.float32)   # 39-dim


def bond_features(bond) -> np.ndarray:
    """6-dimensional bond feature vector."""
    from rdkit.Chem import rdchem
    bt = bond.GetBondTypeAsDouble()
    return np.array([
        float(bt == 1.0),   # single
        float(bt == 1.5),   # aromatic
        float(bt == 2.0),   # double
        float(bt == 3.0),   # triple
        float(bond.GetIsConjugated()),
        float(bond.IsInRing()),
    ], dtype=np.float32)    # 6-dim


@dataclass
class MolGraph:
    smiles: str
    node_features: np.ndarray   # (N, 39)
    edge_index: np.ndarray      # (2, E)  — src, dst pairs
    edge_features: np.ndarray   # (E, 6)
    num_atoms: int
    num_bonds: int
    label_dg: float             # ΔG binding affinity (target)
    label_qed: float            # drug-likeness 0-1 (target)


def smiles_to_graph(smiles: str, label_dg: float = 0.0, label_qed: float = 0.0) -> MolGraph:
    """Convert SMILES to MolGraph. Returns None if invalid."""
    from rdkit import Chem
    from rdkit.Chem import QED

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Atom features
    node_feats = np.array([atom_features(a) for a in mol.GetAtoms()], dtype=np.float32)

    # Edge index + edge features (bidirectional)
    src_list, dst_list, edge_feat_list = [], [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = bond_features(bond)
        src_list += [i, j]
        dst_list += [j, i]
        edge_feat_list += [bf, bf]

    if len(src_list) == 0:
        # Single atom molecule — no edges
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_feats = np.zeros((0, 6), dtype=np.float32)
    else:
        edge_index = np.array([src_list, dst_list], dtype=np.int64)
        edge_feats = np.array(edge_feat_list, dtype=np.float32)

    # Auto-compute QED if not provided
    if label_qed == 0.0:
        try:
            label_qed = float(QED.qed(mol))
        except:
            label_qed = 0.5

    return MolGraph(
        smiles=smiles,
        node_features=node_feats,
        edge_index=edge_index,
        edge_features=edge_feats,
        num_atoms=len(mol.GetAtoms()),
        num_bonds=len(mol.GetBonds()),
        label_dg=label_dg,
        label_qed=label_qed,
    )
