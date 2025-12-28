import argparse
import json
import os
import random
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from typing import Optional, List, Dict, Any

from datasets import load_dataset
from tqdm import tqdm


TRANSLATION_INSTRUCTIONS = {
    "ko": "아래 문장을 영어에서 한국어로 번역해 주세요.",
    "ja": "以下の文を英語から日本語に翻訳してください。",
    "zh": "请将以下句子从英语翻译成中文。",
    "sw": "Tafadhali tafsiri sentensi ifuatayo kutoka Kiingereza hadi Kiswahili.",
    "te": "దయచేసి క్రింది వాక్యాన్ని ఆంగ్లం నుండి తెలుగు భాషలోకి అనువదించండి.",
    "bn": "দয়া করে নিচের বাক্যটিকে ইংরেজি থেকে বাংলা ভাষায় অনুবাদ করুন।",
    "hi": "कृपया निम्नलिखित वाक्य का अंग्रेजी से हिंदी में अनुवाद करें।",
}

NLLB_CODES = {
    "ko": "kor_Hang", "ja": "jpn_Jpan", "zh": "zho_Hans",
    "sw": "swh_Latn", "te": "tel_Telu", "bn": "ben_Beng", "hi": "hin_Deva",
}

LANGUAGE_NAMES = {
    "ko": "Korean", "ja": "Japanese", "zh": "Chinese",
    "sw": "Swahili", "te": "Telugu", "bn": "Bengali", "hi": "Hindi",
}


@dataclass
class DatasetConfig:
    name: str
    path: str
    subset: Optional[str] = None
    split: str = "train"
    sample_size: Optional[int] = None
    instruction_field: Optional[str] = None
    input_field: Optional[str] = None
    output_field: Optional[str] = None
    filter_field: Optional[str] = None
    filter_value: Optional[str] = None
    is_nllb: bool = False


DATASETS = {
    "ko": [
        DatasetConfig("Aya", "CohereForAI/aya_dataset", filter_field="language_code", filter_value="kor", input_field="inputs", output_field="targets"),
        DatasetConfig("KoAlpaca", "beomi/KoAlpaca-v1.1a", instruction_field="instruction", output_field="output"),
        DatasetConfig("Dolly-Ko", "nlpai-lab/databricks-dolly-15k-ko", instruction_field="instruction", input_field="context", output_field="response"),
        DatasetConfig("NLLB", "allenai/nllb", subset="eng_Latn-kor_Hang", sample_size=30000, is_nllb=True),
    ],
    "sw": [
        DatasetConfig("Aya", "CohereForAI/aya_dataset", filter_field="language", filter_value="Swahili", input_field="inputs", output_field="targets"),
        DatasetConfig("xP3mt", "bigscience/xP3mt", subset="sw", sample_size=25000, input_field="inputs", output_field="targets"),
        DatasetConfig("Inkuba", "lelapa/Inkuba-instruct", split="swahili_train", sample_size=20000, instruction_field="instruction", input_field="inputs", output_field="targets"),
        DatasetConfig("NLLB", "allenai/nllb", subset="eng_Latn-swh_Latn", sample_size=30000, is_nllb=True),
    ],
    "te": [
        DatasetConfig("Aya", "CohereForAI/aya_dataset", filter_field="language", filter_value="Telugu", input_field="inputs", output_field="targets"),
        DatasetConfig("Telugu-Alpaca", "Telugu-LLM-Labs/telugu_alpaca_yahma_cleaned_filtered_romanized", instruction_field="telugu_instruction", input_field="telugu_input", output_field="telugu_output"),
        DatasetConfig("NLLB", "allenai/nllb", subset="eng_Latn-tel_Telu", sample_size=25000, is_nllb=True),
    ],
    "bn": [
        DatasetConfig("Aya", "CohereForAI/aya_dataset", filter_field="language_code", filter_value="ben", input_field="inputs", output_field="targets"),
        DatasetConfig("BongChat", "lumatic-ai/BongChat-v1-253k", sample_size=30000, instruction_field="instruction", input_field="input", output_field="output"),
        DatasetConfig("NLLB", "allenai/nllb", subset="ben_Beng-eng_Latn", sample_size=30000, is_nllb=True),
    ],
}


def to_chat_format(sample: Dict, config: DatasetConfig, language: str) -> Optional[Dict]:
    instruction, input_text, output = None, None, None
    
    if config.is_nllb:
        instruction = TRANSLATION_INSTRUCTIONS[language]
        input_text = sample["translation"]["eng_Latn"]
        output = sample["translation"][NLLB_CODES[language]]
    else:
        instruction = sample.get(config.instruction_field) if config.instruction_field else None
        input_text = sample.get(config.input_field) if config.input_field else None
        output = sample.get(config.output_field) if config.output_field else None
    
    if not output:
        return None
    
    if instruction and input_text:
        user_msg = f"{instruction}\n{input_text}"
    elif instruction:
        user_msg = instruction
    elif input_text:
        user_msg = input_text
    else:
        return None
    
    return {
        "messages": [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": output}
        ],
        "source": config.name
    }


def _convert_wrapper(args):
    return to_chat_format(*args)


def load_dataset_samples(config: DatasetConfig, language: str) -> List[Dict]:
    print(f"  Loading {config.name}...")
    
    if config.subset:
        ds = load_dataset(config.path, config.subset, split=config.split, trust_remote_code=True)
    else:
        ds = load_dataset(config.path, split=config.split, trust_remote_code=True)
    
    if config.filter_field and config.filter_value:
        samples = [x for x in ds if x.get(config.filter_field) == config.filter_value]
    else:
        samples = list(ds)
    
    if config.sample_size and len(samples) > config.sample_size:
        samples = random.sample(samples, config.sample_size)

    return samples


def preprocess(language: str, output_dir: str = "output"):    
    configs = DATASETS[language]
    all_samples = []
    
    for config in configs:
        samples = load_dataset_samples(config, language)
        
        with Pool(cpu_count()) as pool:
            args = [(s, config, language) for s in samples]
            converted = list(tqdm(pool.imap(_convert_wrapper, args), total=len(samples), desc=f"    Converting"))
        all_samples.extend([x for x in converted if x])
    
    random.shuffle(all_samples)
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{language}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_samples, f, ensure_ascii=False, indent=2)
    


def main():
    parser = argparse.ArgumentParser(description="Preprocess multilingual SFT datasets")
    parser.add_argument("-l", "--language", choices=list(DATASETS.keys()), help="Language code")
    parser.add_argument("-o", "--output-dir", default="output", help="Output directory")
    parser.add_argument("--all", action="store_true", help="Process all languages")
    args = parser.parse_args()
    
    if args.all:
        for lang in DATASETS:
            preprocess(lang, args.output_dir)
    elif args.language:
        preprocess(args.language, args.output_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
