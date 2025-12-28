import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def _to_numeric(decision):
    if decision == "safety":
        return 1
    elif decision == "multi":
        return -1
    return 0


def _save_figure(out_path, suffix):
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path.replace('.png', f'_{suffix}.png'), dpi=150, bbox_inches='tight')
    plt.close()


def visualize_layer_decisions(decisions, norm_s_list, norm_m_list, d_values=None, out_path=None):
    num_layers = len(decisions)
    decisions_numeric = [_to_numeric(d) for d in decisions]

    plt.figure(figsize=(12, 2))
    sns.heatmap(
        np.array(decisions_numeric).reshape(1, -1),
        annot=True, cmap="coolwarm", cbar=False,
        xticklabels=range(num_layers), yticklabels=[""], fmt="d"
    )
    plt.xlabel("Layer")
    plt.title("Layer Decisions (1: Safety, -1: Multi, 0: Blend)")
    plt.tight_layout()
    _save_figure(out_path, "decisions")

    layers = np.arange(num_layers)
    ns, nm = np.array(norm_s_list), np.array(norm_m_list)
    ps, pm = ns / (ns.sum() + 1e-8), nm / (nm.sum() + 1e-8)
    
    plt.figure(figsize=(10, 5))
    plt.plot(layers, ps, 'o-', label="Safety", linewidth=2)
    plt.plot(layers, pm, 's-', label="Multi", linewidth=2)
    plt.plot(layers, d_values if d_values is not None else (ps - pm), 'x--', label="Difference", linewidth=2)
    plt.axhline(0, color='gray', linestyle=':', linewidth=1)
    plt.xlabel("Layer")
    plt.ylabel("Normalized Ratio")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    _save_figure(out_path, "ratio")


def visualize_module_decisions(decisions, norm_s_dict, norm_m_dict, out_path=None):
    layers = sorted(set(k[0] for k in decisions.keys()))
    modules = [('self_attn', 'Attention'), ('mlp', 'FFN')]
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 3), sharex=True)
    for ax, (mod, label) in zip(axes, modules):
        data = np.array([_to_numeric(decisions.get((l, mod), "multi")) for l in layers]).reshape(1, -1)
        sns.heatmap(data, annot=True, cmap="coolwarm", cbar=False,
                    xticklabels=layers, yticklabels=[label], fmt="d", ax=ax)
    axes[-1].set_xlabel("Layer")
    fig.suptitle("Module Decisions (1: Safety, -1: Multi, 0: Blend)")
    plt.tight_layout()
    _save_figure(out_path, "decisions")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, (mod, title) in zip(axes, modules):
        ns = np.array([norm_s_dict.get((l, mod), 0) for l in layers])
        nm = np.array([norm_m_dict.get((l, mod), 0) for l in layers])
        ps, pm = ns / (ns.sum() + 1e-8), nm / (nm.sum() + 1e-8)
        
        ax.plot(layers, ps, 'o-', label='Safety', linewidth=2)
        ax.plot(layers, pm, 's-', label='Multi', linewidth=2)
        ax.plot(layers, ps - pm, 'x--', label='Difference', linewidth=2)
        ax.axhline(0, color='gray', linestyle=':', linewidth=1)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Normalized Ratio")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save_figure(out_path, "ratio")
