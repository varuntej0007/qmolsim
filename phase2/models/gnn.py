"""
gnn.py
Graph Neural Network — pure numpy implementation.
Message Passing Neural Network (MPNN) architecture.
Predicts: ΔG (binding affinity) and QED (drug-likeness).
No PyTorch. No dependencies beyond numpy.
"""

import numpy as np
import logging
import json
import os

logger = logging.getLogger(__name__)

ATOM_FEAT_DIM = 39
BOND_FEAT_DIM = 6


def relu(x):       return np.maximum(0, x)
def sigmoid(x):    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
def tanh(x):       return np.tanh(x)
def leaky_relu(x): return np.where(x > 0, x, 0.01 * x)


class LinearLayer:
    """Fully connected layer with He initialization."""
    def __init__(self, in_dim: int, out_dim: int, rng: np.random.Generator):
        scale = np.sqrt(2.0 / in_dim)
        self.W = rng.normal(0, scale, (in_dim, out_dim)).astype(np.float32)
        self.b = np.zeros(out_dim, dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x @ self.W + self.b

    def params(self):
        return [self.W, self.b]


class MessagePassingLayer:
    """
    One round of message passing:
    1. For each edge (i→j): compute message = concat(h_i, h_j, edge_feat)
    2. Aggregate messages at each node (mean pooling)
    3. Update node hidden state via linear transform + activation
    """
    def __init__(self, node_dim: int, edge_dim: int, out_dim: int, rng):
        msg_in = node_dim * 2 + edge_dim
        self.msg_layer  = LinearLayer(msg_in, out_dim, rng)
        self.upd_layer  = LinearLayer(node_dim + out_dim, out_dim, rng)

    def forward(self, h: np.ndarray, edge_index: np.ndarray, edge_feats: np.ndarray) -> np.ndarray:
        N = h.shape[0]
        out_dim = self.msg_layer.W.shape[1]

        if edge_index.shape[1] == 0:
            # No edges — identity transform
            agg = np.zeros((N, out_dim), dtype=np.float32)
        else:
            src, dst = edge_index[0].astype(np.int32), edge_index[1].astype(np.int32)
            # Build messages
            msg_input = np.concatenate([h[src], h[dst], edge_feats], axis=1)
            messages   = relu(self.msg_layer.forward(msg_input))  # (E, out_dim)

            # Aggregate at destination nodes
            agg = np.zeros((N, out_dim), dtype=np.float32)
            np.add.at(agg, dst, messages)
            # Mean instead of sum
            counts = np.bincount(dst.astype(np.int32), minlength=N).astype(np.float32).reshape(-1, 1)
            counts = np.maximum(counts, 1)
            agg = agg / counts

        # Update: concat current h with aggregated messages
        upd_input = np.concatenate([h, agg], axis=1)
        h_new = relu(self.upd_layer.forward(upd_input))
        return h_new


class MolGNN:
    """
    3-layer MPNN for molecular property prediction.
    
    Architecture:
        Input encoder  : 39 → hidden_dim
        MP Layer 1     : hidden_dim → hidden_dim
        MP Layer 2     : hidden_dim → hidden_dim  
        MP Layer 3     : hidden_dim → hidden_dim
        Graph readout  : mean pool all node embeddings
        Head ΔG        : hidden_dim → 64 → 1  (binding affinity, kcal/mol)
        Head QED       : hidden_dim → 64 → 1  (drug-likeness, 0-1 sigmoid)
    """

    def __init__(self, hidden_dim: int = 128, seed: int = 42):
        self.hidden_dim = hidden_dim
        self.seed = seed
        rng = np.random.default_rng(seed)

        # Input encoder
        self.encoder = LinearLayer(ATOM_FEAT_DIM, hidden_dim, rng)

        # Message passing layers
        self.mp1 = MessagePassingLayer(hidden_dim, BOND_FEAT_DIM, hidden_dim, rng)
        self.mp2 = MessagePassingLayer(hidden_dim, BOND_FEAT_DIM, hidden_dim, rng)
        self.mp3 = MessagePassingLayer(hidden_dim, BOND_FEAT_DIM, hidden_dim, rng)

        # Prediction heads
        self.dg_h1  = LinearLayer(hidden_dim, 64, rng)
        self.dg_out = LinearLayer(64, 1, rng)

        self.qed_h1  = LinearLayer(hidden_dim, 64, rng)
        self.qed_out = LinearLayer(64, 1, rng)

        logger.info(f"MolGNN initialized | hidden={hidden_dim} | params≈{self._count_params()}")

    def _count_params(self) -> int:
        total = 0
        for layer in [self.encoder, self.dg_h1, self.dg_out, self.qed_h1, self.qed_out]:
            for p in layer.params():
                total += p.size
        for mp in [self.mp1, self.mp2, self.mp3]:
            for layer in [mp.msg_layer, mp.upd_layer]:
                for p in layer.params():
                    total += p.size
        return total

    def forward(self, graph) -> dict:
        """
        Forward pass for one molecule graph.
        Returns dict with 'dg' (binding affinity) and 'qed' (drug-likeness).
        """
        h = relu(self.encoder.forward(graph.node_features))  # (N, hidden)

        h = self.mp1.forward(h, graph.edge_index, graph.edge_features)
        h = self.mp2.forward(h, graph.edge_index, graph.edge_features)
        h = self.mp3.forward(h, graph.edge_index, graph.edge_features)

        # Graph-level readout: mean pooling
        graph_embed = h.mean(axis=0)   # (hidden,)

        # ΔG head — unbounded regression (kcal/mol, typically -15 to +5)
        dg = relu(self.dg_h1.forward(graph_embed))
        dg = float(self.dg_out.forward(dg)[0])

        # QED head — sigmoid output 0-1
        qed = relu(self.qed_h1.forward(graph_embed))
        qed = float(sigmoid(self.qed_out.forward(qed))[0])

        return {'dg': dg, 'qed': qed}

    def predict(self, smiles: str) -> dict:
        """High-level API: SMILES → predictions."""
        from phase2.models.graph import smiles_to_graph
        graph = smiles_to_graph(smiles)
        if graph is None:
            return {'error': f'Invalid SMILES: {smiles}', 'dg': None, 'qed': None}
        preds = self.forward(graph)
        preds['smiles'] = smiles
        preds['num_atoms'] = graph.num_atoms
        preds['num_bonds'] = graph.num_bonds
        return preds

    def save(self, path: str):
        """Save all weights to JSON."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        data = {
            'hidden_dim': self.hidden_dim,
            'seed': self.seed,
            'weights': {}
        }

        def save_layer(name, layer):
            data['weights'][f'{name}_W'] = layer.W.tolist()
            data['weights'][f'{name}_b'] = layer.b.tolist()

        save_layer('encoder', self.encoder)
        for i, mp in enumerate([self.mp1, self.mp2, self.mp3], 1):
            save_layer(f'mp{i}_msg', mp.msg_layer)
            save_layer(f'mp{i}_upd', mp.upd_layer)
        save_layer('dg_h1',  self.dg_h1)
        save_layer('dg_out', self.dg_out)
        save_layer('qed_h1',  self.qed_h1)
        save_layer('qed_out', self.qed_out)

        with open(path, 'w') as f:
            json.dump(data, f)
        logger.info(f"Model saved → {path}")

    @classmethod
    def load(cls, path: str) -> 'MolGNN':
        """Load weights from JSON."""
        with open(path) as f:
            data = json.load(f)
        model = cls(hidden_dim=data['hidden_dim'], seed=data['seed'])

        def load_layer(name, layer):
            layer.W = np.array(data['weights'][f'{name}_W'], dtype=np.float32)
            layer.b = np.array(data['weights'][f'{name}_b'], dtype=np.float32)

        load_layer('encoder', model.encoder)
        for i, mp in enumerate([model.mp1, model.mp2, model.mp3], 1):
            load_layer(f'mp{i}_msg', mp.msg_layer)
            load_layer(f'mp{i}_upd', mp.upd_layer)
        load_layer('dg_h1',  model.dg_h1)
        load_layer('dg_out', model.dg_out)
        load_layer('qed_h1',  model.qed_h1)
        load_layer('qed_out', model.qed_out)

        logger.info(f"Model loaded ← {path}")
        return model
