"""Module-wise Model Swapping (Attention/FFN granularity)"""

import argparse
import os
import re

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from visualization import visualize_module_decisions

TARGET_PARAM_PREFIXES = [
    'self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
    'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj'
]

def _is_target_param(name: str) -> bool:
    if "model.layers" not in name:
        return False
    return any(f".{prefix}." in name for prefix in TARGET_PARAM_PREFIXES)


def compute_module_norms(model_base, model_safe, model_multi):
    base_params = dict(model_base.named_parameters())
    safe_params = dict(model_safe.named_parameters())
    multi_params = dict(model_multi.named_parameters())
    module_norms_safe, module_norms_multi = {}, {}
    
    for name in base_params.keys():
        if _is_target_param(name):
            match = re.search(r'model\.layers\.(\d+)\.(.+?)\.', name)
            if match:
                layer_num = int(match.group(1))
                module_type = match.group(2)
                module_key = (layer_num, module_type)
                
                if module_key not in module_norms_safe:
                    module_norms_safe[module_key] = 0
                    module_norms_multi[module_key] = 0
                
                base_p, safe_p, multi_p = base_params[name], safe_params[name], multi_params[name]
                base_norm = torch.norm(base_p).item()
                
                if base_norm > 1e-8:
                    relative_safe = torch.norm(safe_p - base_p).item() / base_norm
                    relative_multi = torch.norm(multi_p - base_p).item() / base_norm
                else:
                    relative_safe = torch.norm(safe_p - base_p).item()
                    relative_multi = torch.norm(multi_p - base_p).item()
                
                module_norms_safe[module_key] += relative_safe ** 2
                module_norms_multi[module_key] += relative_multi ** 2
    
    for key in module_norms_safe:
        module_norms_safe[key] = np.sqrt(module_norms_safe[key])
        module_norms_multi[key] = np.sqrt(module_norms_multi[key])
    
    return module_norms_safe, module_norms_multi


def compute_module_difference(norm_s_dict, norm_m_dict, tau=0.001, alpha=0.5):
    keys = sorted(norm_s_dict.keys())
    norm_s = np.array([norm_s_dict[k] for k in keys])
    norm_m = np.array([norm_m_dict[k] for k in keys])
    ps = norm_s / (norm_s.sum() + 1e-8)
    pm = norm_m / (norm_m.sum() + 1e-8)
    
    d = ps - pm
    decisions = {}
    
    for i, key in enumerate(keys):
        if d[i] > tau:
            decisions[key] = "safety"
        elif d[i] < -tau:
            decisions[key] = "multi"
        else:
            decisions[key] = f"blend({alpha:.2f})"
    
    return decisions, d, keys


def module_swap(model_safe, model_multi, decisions, alpha=0.5):
    safe_layers = model_safe.model.layers
    multi_layers = model_multi.model.layers
    
    with torch.no_grad():
        for (layer_num, module_type), decision in decisions.items():
            safe_layer = safe_layers[layer_num]
            multi_layer = multi_layers[layer_num]
            
            if module_type == "self_attn":
                safe_module = safe_layer.self_attn
                multi_module = multi_layer.self_attn
            elif module_type == "mlp":
                safe_module = safe_layer.mlp
                multi_module = multi_layer.mlp
            else:
                continue
            
            if decision == "safety":
                multi_module.load_state_dict(safe_module.state_dict())
            elif decision != "multi":
                for name, param in multi_module.named_parameters():
                    safe_param = dict(safe_module.named_parameters())[name]
                    param.data.copy_(alpha * safe_param.data + (1 - alpha) * param.data)

        for name, param in model_multi.named_parameters():
            if "model.layers" not in name:
                safe_param = dict(model_safe.named_parameters())[name]
                param.data.copy_(alpha * safe_param.data + (1 - alpha) * param.data)

    return model_multi

def main():
    parser = argparse.ArgumentParser(description="Module-wise model swapping")
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

    print("Computing module norms...")
    module_norms_safe, module_norms_multi = compute_module_norms(model_base, model_safe, model_multi)

    decisions, _, _ = compute_module_difference(
        module_norms_safe, module_norms_multi, tau=args.tau, alpha=args.alpha
    )

    print("\nMerging models...")
    hybrid_model = module_swap(model_safe, model_multi, decisions, alpha=args.alpha)

    figure_dir = args.figure_dir or os.path.join(args.output, "figures")
    visualize_module_decisions(decisions, module_norms_safe, module_norms_multi,
                               out_path=os.path.join(figure_dir, "module_analysis.png"))

    os.makedirs(args.output, exist_ok=True)
    hybrid_model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()