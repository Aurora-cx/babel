import json
import re
from pathlib import Path
import os

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from openai import OpenAI
from typing import Tuple

def print_gpu_memory():
    if torch.cuda.is_available():
        print("\nGPU:")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}:")
            print(f"  Total: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
            print(f"  Allocated: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
            print(f"  Reserved: {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB")
    else:
        print("No GPU available")

def zh_decode(input_ids):
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    for i in range(len(tokens)):
        tokens[i] = tokenizer.decode(input_ids[i], skip_special_tokens=False)
    return tokens

client = OpenAI(api_key="sk-xxx")

def check_answer_with_gpt(predicted: str, gold: str, prompt: str, lang: str = "en") -> Tuple[bool, str]:
    system_prompt = {
        "en": "You are an answer evaluator. Your task is to determine if the predicted answer contains the core information from the ground truth answer.\n\nYour response must strictly follow this format:\n\n[0/1]: <your reasoning>\n\nWhere 0 means incorrect and 1 means correct.",
        "zh": "你是一个答案评估者。你需要判断预测答案是否包含了标准答案。\n\n你的回答必须严格遵循以下格式：\n\n[0/1]：<你的理由>\n\n其中0表示错误，1表示正确。"
    }
    messages = [
        {"role": "system", "content": system_prompt[lang]},
        {"role": "user", "content": f"Question: {prompt}\nGround Truth: {gold}\nPredicted Answer: {predicted}\n\nPlease evaluate if the predicted answer is correct and provide your reasoning."}
    ]
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0,
            max_tokens=150
        )
        result = response.choices[0].message.content.strip()
        match = re.search(r'\[([01])\]', result)
        if match:
            is_correct = match.group(1) == "1"
            reason = re.sub(r'^\[[01]\][：:]\s*', '', result).strip()
            return is_correct, reason
        else:
            print(f"Unexpected response format: {result}")
            return gold in predicted, "Unexpected response format, using simple string matching"
    except Exception as e:
        print(f"GPT API call failed: {e}")
        return gold in predicted, "API call failed, using simple string matching"

MODEL_NAME = "Qwen/Qwen1.5-1.8B-Chat"
DATA_DIR = Path("data/paraphrased_causal_data_v2")
OUTPUT_DIR = Path("outputs/qwen15_18b_para_v2_00")
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float16

print(f"Loading model {MODEL_NAME} ...")
print_gpu_memory()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="cuda:0",
    torch_dtype=DTYPE,
    output_hidden_states=True,
    output_attentions=True,
    trust_remote_code=True,
    attn_implementation="eager"
)
model.eval()

print("\nModel loaded.")
print_gpu_memory()

def check_answer(predicted: str, gold: str) -> bool:
    gold = gold.strip()
    predicted = predicted.strip()
    if re.fullmatch(r"[-+]?\d+", gold):
        return re.search(r"[-+]?\d+", predicted) and re.search(r"[-+]?\d+", predicted).group() == gold
    if gold.lower() in {"yes", "no"}:
        return gold.lower() in predicted.lower()
    return gold in predicted

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
jsonl_files = list(DATA_DIR.glob("*.jsonl"))
print(f"\nFound {len(jsonl_files)} files to process")

for data_file in tqdm(jsonl_files, desc="Processing files", unit="file"):
    print(f"\nProcessing {data_file.name}...")
    output_path = OUTPUT_DIR / f"{data_file.stem}.pt"
    output_jsonl = OUTPUT_DIR / f"{data_file.stem}.jsonl"
    print(f"Reading prompts from {data_file} ...")
    with open(data_file, "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f]
    print("Running forward passes ...")
    all_records = []
    for sample in tqdm(dataset, desc="Processing samples", leave=False):
        sample_id = sample.get("id", None)
        prompt_text_en = "\n".join(sample["en_para"])
        prompt_text_zh = "\n".join(sample["zh_para"])
        ground_truth = sample.get("answer", "")
        messages_en = [
            {"role": "system", "content": "Use only one sentence to answer."},
            {"role": "user", "content": prompt_text_en}
        ]
        messages_zh = [
            {"role": "system", "content": "只用一句话回答。"},
            {"role": "user", "content": prompt_text_zh}
        ]
        chat_prompt_en = tokenizer.apply_chat_template(messages_en, tokenize=False, add_generation_prompt=True)
        chat_prompt_zh = tokenizer.apply_chat_template(messages_zh, tokenize=False, add_generation_prompt=True)
        inputs_en = tokenizer(chat_prompt_en, return_tensors="pt").to(DEVICE)
        inputs_zh = tokenizer(chat_prompt_zh, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs_en = model(**inputs_en)
            outputs_zh = model(**inputs_zh)
        input_ids_en = inputs_en["input_ids"].squeeze(0).tolist()
        tokens_en = tokenizer.convert_ids_to_tokens(input_ids_en)
        input_ids_zh = inputs_zh["input_ids"].squeeze(0).tolist()
        tokens_zh = zh_decode(input_ids_zh)
        embedding_out_en = outputs_en.hidden_states[0].cpu()
        embedding_out_zh = outputs_zh.hidden_states[0].cpu()
        hidden_states_en = [h.cpu() for h in outputs_en.hidden_states[:]]
        hidden_states_zh = [h.cpu() for h in outputs_zh.hidden_states[:]]
        attentions_en = [a[0].cpu() for a in outputs_en.attentions]
        attentions_zh = [a[0].cpu() for a in outputs_zh.attentions]
        logits_en = outputs_en.logits[0].cpu()
        logits_zh = outputs_zh.logits[0].cpu()
        generated_ids_en = model.generate(
            inputs_en.input_ids,
            max_new_tokens=512,
            temperature=0.1,
            top_p=0.1,
            do_sample=False,
            num_beams=1,
            repetition_penalty=1.0
        )
        generated_ids_zh = model.generate(
            inputs_zh.input_ids,
            max_new_tokens=512,
            temperature=0.1,
            top_p=0.1,
            do_sample=False,
            num_beams=1,
            repetition_penalty=1.0
        )
        generated_ids_en = [
            output_ids[len(input_ids_en):] for input_ids_en, output_ids in zip(inputs_en.input_ids, generated_ids_en)
        ]
        generated_ids_zh = [
            output_ids[len(input_ids_zh):] for input_ids_zh, output_ids in zip(inputs_zh.input_ids, generated_ids_zh)
        ]
        predicted_answer_en = tokenizer.batch_decode(generated_ids_en, skip_special_tokens=True)[0].strip()
        predicted_answer_zh = tokenizer.batch_decode(generated_ids_zh, skip_special_tokens=True)[0].strip()
        print(f"English Predicted Answer: {predicted_answer_en}")
        print(f"Chinese Predicted Answer: {predicted_answer_zh}")
        is_correct_en, reason_en = check_answer_with_gpt(predicted_answer_en, ground_truth["en"], prompt_text_en, "en")
        is_correct_zh, reason_zh = check_answer_with_gpt(predicted_answer_zh, ground_truth["zh"], prompt_text_zh, "zh")
        print(f"English Answer Evaluation: [{1 if is_correct_en else 0}] {reason_en}")
        print(f"Chinese Answer Evaluation: [{1 if is_correct_zh else 0}] {reason_zh}")
        all_records.append({
            "id": sample_id,
            "prompt_en": prompt_text_en,
            "prompt_zh": prompt_text_zh,
            "input_ids_en": input_ids_en,
            "tokens_en": tokens_en,
            "input_ids_zh": input_ids_zh,
            "tokens_zh": tokens_zh,
            "embedding_out_en": embedding_out_en,
            "embedding_out_zh": embedding_out_zh,
            "hidden_states_en": hidden_states_en,
            "hidden_states_zh": hidden_states_zh,
            "attentions_en": attentions_en,
            "attentions_zh": attentions_zh,
            "logits_en": logits_en,
            "logits_zh": logits_zh,
            "predicted_answer_en": predicted_answer_en,
            "predicted_answer_zh": predicted_answer_zh,
            "ground_truth": ground_truth,
            "is_correct_en": is_correct_en,
            "is_correct_zh": is_correct_zh,
            "judgment_reason_en": reason_en,
            "judgment_reason_zh": reason_zh
        })
    print(f"Saving {len(all_records)} records to {output_path} ...")
    torch.save(all_records, output_path)
    correct_en = sum(1 for record in all_records if record["is_correct_en"])
    correct_zh = sum(1 for record in all_records if record["is_correct_zh"])
    total = len(all_records)
    stats = {
        "file_name": data_file.name,
        "total_samples": total,
        "correct_en": correct_en,
        "correct_zh": correct_zh,
        "accuracy_en": correct_en/total*100,
        "accuracy_zh": correct_zh/total*100
    }
    stats_file = OUTPUT_DIR / f"{data_file.stem}_stats.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"\nStats ({data_file.stem}):")
    print(f"EN: {correct_en}/{total} = {correct_en/total*100:.2f}%")
    print(f"ZH: {correct_zh}/{total} = {correct_zh/total*100:.2f}%")
    print(f"Stats saved to: {stats_file}")
    nature_res = []
    for i, output in enumerate(all_records, 1):
        new_irem = {}
        new_irem["id"] = output["id"]
        new_irem["prompt_en"] = output["prompt_en"]
        new_irem["predicted_answer_en"] = output["predicted_answer_en"]
        new_irem["ground_truth_en"] = output["ground_truth"]["en"]
        new_irem["is_correct_en"] = output["is_correct_en"]
        new_irem["judgment_reason_en"] = output["judgment_reason_en"]
        new_irem["prompt_zh"] = output["prompt_zh"]
        new_irem["predicted_answer_zh"] = output["predicted_answer_zh"]
        new_irem["ground_truth_zh"] = output["ground_truth"]["zh"]
        new_irem["is_correct_zh"] = output["is_correct_zh"]
        new_irem["judgment_reason_zh"] = output["judgment_reason_zh"]
        nature_res.append(new_irem)
    output_jsonl = OUTPUT_DIR / f"{data_file.stem}_results.jsonl"
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for item in nature_res:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"All results saved for {data_file.stem}!")
    print(f"Natural language outputs saved to {output_jsonl}!")
all_stats = []
for data_file in jsonl_files:
    stats_file = OUTPUT_DIR / f"{data_file.stem}_stats.json"
    if stats_file.exists():
        with open(stats_file, "r", encoding="utf-8") as f:
            all_stats.append(json.load(f))
summary_file = OUTPUT_DIR / "summary_stats.json"
with open(summary_file, "w", encoding="utf-8") as f:
    json.dump(all_stats, f, ensure_ascii=False, indent=2)
print("\nAll files processed!")
print(f"Summary saved to: {summary_file}") 