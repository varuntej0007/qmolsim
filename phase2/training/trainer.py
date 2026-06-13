"""
trainer.py
Proper training with analytical gradients through output heads.
Full chain rule from loss → output layer → hidden layer → graph embedding.
"""

import numpy as np
import logging
import time

logger = logging.getLogger(__name__)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def relu(x):
    return np.maximum(0, x)


def relu_grad(x):
    return (x > 0).astype(np.float32)


def forward_with_cache(model, graph):
    """
    Full forward pass, caching all intermediate values for backprop.
    Returns predictions + cache dict.
    """
    h = relu(model.encoder.forward(graph.node_features))

    h = model.mp1.forward(h, graph.edge_index, graph.edge_features)
    h = model.mp2.forward(h, graph.edge_index, graph.edge_features)
    h = model.mp3.forward(h, graph.edge_index, graph.edge_features)

    graph_embed = h.mean(axis=0)  # (hidden,)

    # ΔG head
    dg_h1_pre  = model.dg_h1.forward(graph_embed)
    dg_h1_post = relu(dg_h1_pre)
    dg_out_pre = model.dg_out.forward(dg_h1_post)
    dg_pred    = float(dg_out_pre[0])

    # QED head
    qed_h1_pre  = model.qed_h1.forward(graph_embed)
    qed_h1_post = relu(qed_h1_pre)
    qed_out_pre = model.qed_out.forward(qed_h1_post)
    qed_pred    = float(sigmoid(qed_out_pre)[0])

    cache = {
        'graph_embed': graph_embed,
        'dg_h1_pre':   dg_h1_pre,
        'dg_h1_post':  dg_h1_post,
        'dg_pred':     dg_pred,
        'qed_h1_pre':  qed_h1_pre,
        'qed_h1_post': qed_h1_post,
        'qed_pred':    qed_pred,
    }
    return cache


def backward_heads(model, graph, cache, lr):
    """
    Analytical backprop through both prediction heads.
    Chain rule: dL/dW = dL/dpred * dpred/dh1 * dh1/dW
    """
    ge   = cache['graph_embed']
    loss = 0.0

    # ── ΔG head ──────────────────────────────────────────────────
    err_dg = cache['dg_pred'] - graph.label_dg
    loss  += err_dg ** 2

    # dL/d(dg_out_pre) = 2 * err  (linear output, no activation)
    d_dg_out = np.array([2.0 * err_dg], dtype=np.float32)

    # Gradients for dg_out layer
    dg_h1_post = cache['dg_h1_post']  # (64,)
    grad_dg_out_W = np.outer(dg_h1_post, d_dg_out)   # (64,1)
    grad_dg_out_b = d_dg_out                          # (1,)

    # Backprop through relu in dg_h1
    d_dg_h1_post = model.dg_out.W @ d_dg_out          # (64,)
    d_dg_h1_pre  = d_dg_h1_post * relu_grad(cache['dg_h1_pre'])

    # Gradients for dg_h1 layer
    grad_dg_h1_W = np.outer(ge, d_dg_h1_pre)          # (hidden,64)
    grad_dg_h1_b = d_dg_h1_pre                        # (64,)

    # ── QED head ─────────────────────────────────────────────────
    err_qed = cache['qed_pred'] - graph.label_qed
    loss   += err_qed ** 2

    # dL/d(qed_out_pre): sigmoid derivative = sigmoid*(1-sigmoid)
    sig_val   = cache['qed_pred']
    d_qed_out = np.array([2.0 * err_qed * sig_val * (1.0 - sig_val)], dtype=np.float32)

    # Gradients for qed_out layer
    qed_h1_post = cache['qed_h1_post']
    grad_qed_out_W = np.outer(qed_h1_post, d_qed_out)
    grad_qed_out_b = d_qed_out

    # Backprop through relu in qed_h1
    d_qed_h1_post = model.qed_out.W @ d_qed_out
    d_qed_h1_pre  = d_qed_h1_post * relu_grad(cache['qed_h1_pre'])

    # Gradients for qed_h1 layer
    grad_qed_h1_W = np.outer(ge, d_qed_h1_pre)
    grad_qed_h1_b = d_qed_h1_pre

    # ── Apply gradients with gradient clipping ────────────────────
    clip = 1.0
    def update(layer, gW, gb):
        gW = np.clip(gW, -clip, clip)
        gb = np.clip(gb, -clip, clip)
        layer.W -= lr * gW
        layer.b -= lr * gb

    update(model.dg_out,  grad_dg_out_W,  grad_dg_out_b)
    update(model.dg_h1,   grad_dg_h1_W,   grad_dg_h1_b)
    update(model.qed_out, grad_qed_out_W,  grad_qed_out_b)
    update(model.qed_h1,  grad_qed_h1_W,   grad_qed_h1_b)

    return loss


def train_epoch(model, graphs, lr):
    total_loss = 0.0
    indices = np.random.permutation(len(graphs))
    for idx in indices:
        g = graphs[idx]
        cache = forward_with_cache(model, g)
        loss  = backward_heads(model, g, cache, lr)
        total_loss += loss
    return total_loss / len(graphs)


def train(model, train_graphs, val_graphs,
          epochs=60, lr=0.01,
          save_path="phase2/data/model.json"):

    history = {'train_loss': [], 'val_loss': [], 'epochs': []}
    best_val = float('inf')

    logger.info(
        f"Training MolGNN | epochs={epochs} | lr={lr} | "
        f"train={len(train_graphs)} | val={len(val_graphs)}"
    )

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_graphs, lr)

        val_losses = []
        for g in val_graphs:
            cache = forward_with_cache(model, g)
            vl = (cache['dg_pred'] - g.label_dg)**2 + \
                 (cache['qed_pred'] - g.label_qed)**2
            val_losses.append(vl)
        val_loss = float(np.mean(val_losses))

        history['train_loss'].append(float(train_loss))
        history['val_loss'].append(float(val_loss))
        history['epochs'].append(epoch)

        logger.info(
            f"Epoch {epoch:3d}/{epochs} | "
            f"train={train_loss:.4f} | val={val_loss:.4f} | "
            f"time={time.time()-t0:.1f}s"
        )

        if val_loss < best_val:
            best_val = val_loss
            model.save(save_path)
            logger.info(f"  ✓ Best saved (val={val_loss:.4f})")

    return history
