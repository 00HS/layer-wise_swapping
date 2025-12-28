# Multilingual SFT Data Preprocessing

Convert instruction-following datasets into chat format for SFT training.

## Usage

```bash
# Single language
python preprocess.py -l ko

# All languages
python preprocess.py --all
```

## Output Format

```json
{
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "source": "dataset_name"
}
```

## Supported Languages

| Code | Language | Datasets |
|------|----------|----------|
| ko | Korean | Aya, KoAlpaca, Dolly-Ko, NLLB |
| sw | Swahili | Aya, xP3mt, Inkuba, NLLB |
| te | Telugu | Aya, Telugu-Alpaca, NLLB |
| bn | Bengali | Aya, BongChat, NLLB |
