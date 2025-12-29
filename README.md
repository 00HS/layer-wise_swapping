# Layer-wise Swapping

Merge safety-tuned and multilingual-tuned LLMs by analyzing per-layer/module weight changes.

## Usage

### Layer-wise Swapping

Swap at **layer** granularity (entire transformer layers):

```bash
python layer_swap.py \
    -b meta-llama/Llama-3.1-8B-Instruct \
    -s ./checkpoint/safety_model \
    -m ./checkpoint/multi_model \
    -o ./checkpoint/merged
```

### Module-wise Swapping

Swap at **module** granularity (Attention vs FFN separately):

```bash
python module_swap.py \
    -b meta-llama/Llama-3.1-8B-Instruct \
    -s ./checkpoint/safety_model \
    -m ./checkpoint/multi_model \
    -o ./checkpoint/merged
```

### Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `-b`, `--base-model` | Base model (HuggingFace ID or path) | Required |
| `-s`, `--safety-model` | Safety-tuned model path | Required |
| `-m`, `--multi-model` | Multilingual-tuned model path | Required |
| `-o`, `--output` | Output path | Required |
| `--tau` | Threshold for decision | 0.001 |
| `--alpha` | Blend ratio | 0.5 |
| `--figure-dir` | Directory for figures | `<output>/figures` |

## How It Works

1. Compute relative weight changes (ΔW) compared to base model
2. For each layer/module, compare safety vs multilingual changes
3. Decision per layer/module:
   - `diff > tau` → Use safety
   - `diff < -tau` → Use multilingual
   - Otherwise → Blend with ratio α

## Data Preprocessing

See [data/README.md](data/README.md) for preparing multilingual SFT datasets.
