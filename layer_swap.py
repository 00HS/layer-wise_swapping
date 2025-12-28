"""Layer-wise Model Swapping"""

import argparse
import os
import re

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from visualization import visualize_layer_decisions

TARGET_PARAM_PREFIXES = [
    'self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
    'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj'
]


def _is_target_param(name: str) -> bool:
    if "model.layers" not in name:
        return False
    return any(f".{prefix}." in name for prefix in TARGET_PARAM_PREFIXES)


def compute_layer_norms(model_base, model_safe, model_multi):
    norms_safe, norms_multi = {}, {}
    base_params = dict(model_base.named_parameters())
    safe_params = dict(model_safe.named_parameters())
    multi_params = dict(model_multi.named_parameters())
    
    for name in base_params.keys():
        if _is_target_param(name):
            base_p, safe_p, multi_p = base_params[name], safe_params[name], multi_params[name]
            base_norm = torch.norm(base_p).item()
            
            if base_norm > 1e-8:
                norms_safe[name] = torch.norm(safe_p - base_p).item() / base_norm
                norms_multi[name] = torch.norm(multi_p - base_p).item() / base_norm
            else:
                norms_safe[name] = torch.norm(safe_p - base_p).item()
                norms_multi[name] = torch.norm(multi_p - base_p).item()
    
    return norms_safe, norms_multi


def compute_layer_difference(norm_s, norm_m, tau=0.001, alpha=0.5):
    norm_s, norm_m = np.array(norm_s), np.array(norm_m)
    ps = norm_s / (norm_s.sum() + 1e-8)
    pm = norm_m / (norm_m.sum() + 1e-8)
    d = ps - pm
    
    decisions = []
    for diff in d:
        if diff > tau:
            decisions.append("safety")
        elif diff < -tau:
            decisions.append("multi")
        else:
            decisions.append(f"blend({alpha:.2f})")
    return decisions, d


def layer_swap(model_safe, model_multi, decisions, alpha=0.5):
    safe_layers = model_safe.model.layers
    multi_layers = model_multi.model.layers

    with torch.no_grad():
        for i, (layer_s, layer_m) in enumerate(zip(safe_layers, multi_layers)):
            if decisions[i] == "safety":
                layer_m.load_state_dict(layer_s.state_dict())
            elif decisions[i] != "multi":
                for name, param in layer_m.named_parameters():
                    safe_param = dict(layer_s.named_parameters())[name]
                    param.data.copy_(alpha * safe_param.data + (1 - alpha) * param.data)

        for name, param in model_multi.named_parameters():
            if "model.layers" not in name:
                safe_param = dict(model_safe.named_parameters())[name]
                param.data.copy_(alpha * safe_param.data + (1 - alpha) * param.data)

    return model_multi


def main():
    parser = argparse.ArgumentParser(description="Layer-wise model swapping")
    parser.add_argument("-b", "--base-model", required=True, help="Base model path")
    parser.add_argument("-s", "--safety-model", required=True, help="Safety-tuned model path")
    parser.add_argument("-m", "--multi-model", required=True, help="Multilingual model path")
    parser.add_argument("-o", "--output", required=True, help="Output path")
    parser.add_argument("--tau", type=float, default=0.001, help="Decision threshold")
    parser.add_argument("--alpha", type=float, default=0.5, help="Blend ratio")
    parser.add_argument("--figure-dir", default=None, help="Figure output directory")
    args = parser.parse_args()

    print("Loading models...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model_base = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16)
    model_safe = AutoModelForCausalLM.from_pretrained(args.safety_model, torch_dtype=torch.bfloat16)
    model_multi = AutoModelForCausalLM.from_pretrained(args.multi_model, torch_dtype=torch.bfloat16)

    print("Computing layer norms...")
    norms_safe, norms_multi = compute_layer_norms(model_base, model_safe, model_multi)

    layer_norms_safe, layer_norms_multi = {}, {}
    for name in norms_safe.keys():
        if _is_target_param(name):
            match = re.search(r'model\.layers\.(\d+)', name)
            if match:
                layer_num = int(match.group(1))
                layer_norms_safe[layer_num] = layer_norms_safe.get(layer_num, 0) + norms_safe[name]
                layer_norms_multi[layer_num] = layer_norms_multi.get(layer_num, 0) + norms_multi[name]

    sorted_layers = sorted(layer_norms_safe.keys())
    norm_s_list = [layer_norms_safe[l] for l in sorted_layers]
    norm_m_list = [layer_norms_multi[l] for l in sorted_layers]

    decisions, d_values = compute_layer_difference(norm_s_list, norm_m_list, tau=args.tau, alpha=args.alpha)

    print("\nMerging models...")
    hybrid_model = layer_swap(model_safe, model_multi, decisions, alpha=args.alpha)

    figure_dir = args.figure_dir or os.path.join(args.output, "figures")
    visualize_layer_decisions(decisions, norm_s_list, norm_m_list, d_values,
                              out_path=os.path.join(figure_dir, "layer_analysis.png"))

    os.makedirs(args.output, exist_ok=True)
    hybrid_model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()