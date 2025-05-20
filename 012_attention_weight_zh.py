import torch, numpy as np, tqdm, json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
# 设定环境
torch.cuda.set_device(0)
torch.cuda.empty_cache()

# 配置
MODEL_NAME = "Qwen/Qwen1.5-1.8B-Chat"
DATA_DIR = Path("data/clean_causal_data_v2")
FORWARD_DIR = Path("outputs/qwen15_18b_00_v2")
SAVE_DIR = Path("outputs/try/attention_weights/attention_weights_zh")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="cuda:0",
    torch_dtype=torch.float16,
    output_hidden_states=True,
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model.eval()

def zh_decode(input_ids):
    """将input_ids转换为中文tokens"""
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    for i in range(len(tokens)):
        tokens[i] = tokenizer.decode(input_ids[i], skip_special_tokens=False)
    return tokens

def find_token_indices_for_word(prompt: str, target_word: str, component: str):
    """
    找到目标词在 prompt 中被 tokenizer 切分后的 token 索引。
    如果 component 是 question_subject 或 question_verb，则找第二次出现的位置。
    """
    encoding = tokenizer(prompt, return_offsets_mapping=True, return_tensors="pt", add_special_tokens=False)
    tokens = zh_decode(encoding.input_ids[0])
    offsets = encoding.offset_mapping[0].tolist()

    # 找所有匹配位置
    all_occurrences = [m.start() for m in re.finditer(re.escape(target_word), prompt)]

    if not all_occurrences:
        return [], []  # 没找到
    elif component in {"question_subject", "question_verb"}:
        if len(all_occurrences) < 2:
            start_idx = all_occurrences[0]
        else:
            start_idx = all_occurrences[1]
    else:
        start_idx = all_occurrences[0]

    end_idx = start_idx + len(target_word)

    # 找被 tokenizer 对应的 token indices
    target_token_indices = [
        i for i, (s, e) in enumerate(offsets)
        if not (e <= start_idx or s >= end_idx)
    ]

    return target_token_indices, [tokens[i] for i in target_token_indices]

def compute_component_attention(attentions, token_indices, num_heads):
    """计算每个组件在每个head上的attention占比（考虑自回归结构）"""
    # attentions: [num_layers, num_heads, seq_len, seq_len]
    # 返回 shape: [num_layers, num_heads]
    attention_ratios = np.zeros((len(attentions), num_heads))

    for layer_idx, layer_attention in enumerate(attentions):
        for head_idx in range(num_heads):
            head_attention = layer_attention[head_idx]  # [seq_len, seq_len]
            seq_len = head_attention.size(0)

            component_ratios = []

            for idx in token_indices:
                # 有效 query 是这个 token 之后的位置
                valid_queries = list(range(idx, seq_len))

                # 对每个token，计算这些 query 给它分配的注意力总和
                attn_to_token = head_attention[valid_queries, idx].sum().item()
                attn_total_from_queries = head_attention[valid_queries].sum().item()

                ratio = attn_to_token / attn_total_from_queries if attn_total_from_queries > 0 else 0.0
                component_ratios.append(ratio)

            # 多 token 平均
            avg_ratio = sum(component_ratios) / len(component_ratios) if component_ratios else 0.0
            attention_ratios[layer_idx, head_idx] = avg_ratio

    return attention_ratios

def load_theme_data(theme_name: str):
    """加载单个主题的数据"""
    # 加载原始数据
    jsonl_path = DATA_DIR / f"{theme_name}.jsonl"
    raw_data = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            raw_data.append(json.loads(line))

    # 加载前向结果
    pt_path = FORWARD_DIR / f"{theme_name}.pt"
    forward_data = torch.load(pt_path, map_location="cpu")
    
    # 确保forward_data是列表
    if not isinstance(forward_data, list):
        forward_data = [forward_data]

    # 合并数据
    for raw_sample, forward_sample in zip(raw_data, forward_data):
        if raw_sample["id"] != forward_sample["id"]:
            continue
        raw_sample["is_correct_zh"] = forward_sample.get("is_correct_zh", False)
        raw_sample["input_ids_zh"] = forward_sample.get("input_ids_zh", None)
        raw_sample["attentions_zh"] = forward_sample.get("attentions_zh", None)

    return raw_data

def process_sample(model, sample: dict):
    """处理单个样本，计算所有组件的attention权重"""
    # 构造正确的消息格式
    messages = [
        {"role": "system", "content": "只用一句话回答。"},
        {"role": "user", "content": "\n".join(sample["zh"])}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # 使用新的prompt生成input_ids
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda:0")
    tokens_out = zh_decode(input_ids[0])

    labels = sample["labels"]["zh"]
    # 添加标记词组件
    labels["connective_once"] = "一旦"
    labels["connective_then"] = "然后"
    labels["connective_if"] = "如果"
    labels["connective_therefore"] = "因此"
    labels["connective_final_result"] = "最终结果"
    labels["question_subject"] = labels["cause_subject"]
    labels["question_verb"] = labels["cause_verb"]

    # 获取attention权重
    attentions_zh = sample["attentions_zh"]
    if attentions_zh is None:
        return None

    # 计算每个组件的attention权重
    component_attentions = {}
    component_tokens = {}

    # 获取attention head的数量
    num_heads = attentions_zh[0].shape[0]  # 使用第一个attention层的head数量

    components = [
        "cause_subject", "cause_verb",
        "intermediate_subject", "intermediate_verb",
        "final_subject", "final_verb",
        "question_subject", "question_verb",
        "connective_once", "connective_then",
        "connective_if", "connective_therefore",
        "connective_final_result"
    ]

    for component in components:
        value = labels[component]
        token_indices, tokens = find_token_indices_for_word(prompt, value, component)
        tokens = [tokens_out[i] for i in token_indices]
        if not token_indices:
            continue
        attention_sums = compute_component_attention(attentions_zh, token_indices, num_heads)
        component_attentions[component] = attention_sums.tolist() # [num_layers, num_heads]
        component_tokens[component] = tokens

    return {
        "sample_id": sample.get("id", None),
        "attentions": component_attentions,
        "tokens": component_tokens,
        "prompt": prompt,
        "labels": labels,
        "is_correct_zh": sample.get("is_correct_zh", False)
    }

def process_theme(theme_name: str):
    """处理单个主题的数据"""
    theme_data = load_theme_data(theme_name)

    # 创建保存目录
    correct_dir = SAVE_DIR / "correct"
    incorrect_dir = SAVE_DIR / "incorrect"
    correct_dir.mkdir(parents=True, exist_ok=True)
    incorrect_dir.mkdir(parents=True, exist_ok=True)

    correct_results = []
    incorrect_results = []
    error_count = 0
    total_samples = len(theme_data)

    for sample in tqdm.tqdm(theme_data):
        try:
            result = process_sample(model, sample)
            if result is not None:
                if result["is_correct_zh"]:
                    correct_results.append(result)
                else:
                    incorrect_results.append(result)
        except Exception as e:
            error_count += 1
            continue
        finally:
            torch.cuda.empty_cache()

    # 保存正确样本的结果
    correct_file = correct_dir / f"attention_{theme_name}.json"
    with open(correct_file, "w", encoding="utf-8") as f:
        json.dump(correct_results, f, ensure_ascii=False, indent=2)

    # 保存错误样本的结果
    incorrect_file = incorrect_dir / f"attention_{theme_name}.json"
    with open(incorrect_file, "w", encoding="utf-8") as f:
        json.dump(incorrect_results, f, ensure_ascii=False, indent=2)

def main():
    """主函数"""
    # 获取所有主题名称
    theme_names = [f.stem for f in DATA_DIR.glob("*.jsonl")]
    print(f"Found {len(theme_names)} themes: {theme_names}")
    
    # 处理所有主题
    for theme_name in theme_names:
        print(f"\nProcessing theme: {theme_name}")
        process_theme(theme_name)

    print("✅ Attention weight analysis 完成！")

if __name__ == "__main__":
    main()