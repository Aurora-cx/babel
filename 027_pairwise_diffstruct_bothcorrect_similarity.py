import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# 配置
DATA_DIRS = {
    "en": Path("data/clean_causal_data_v2"),
    "zh": Path("data/clean_causal_data_v2"),
    "enpara": Path("data/paraphrased_causal_data_v2"),
    "zhpara": Path("data/paraphrased_causal_data_v2")
}
OUTPUT_DIRS = {
    "en": Path("outputs/qwen15_18b_00_v2"),
    "zh": Path("outputs/qwen15_18b_00_v2"),
    "enpara": Path("outputs/qwen15_18b_para_v2_00"),
    "zhpara": Path("outputs/qwen15_18b_para_v2_00")
}
SAVE_DIR = Path("outputs/period_pairwise_similarity")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# period前token定位
def find_period_position(tokens, is_chinese=False):
    period = '。\n' if is_chinese else '.Ċ'
    for i, token in enumerate(tokens):
        if token == period:
            return i - 1
    return -1

def main():
    theme_names = [f.stem for f in DATA_DIRS["en"].glob("*.jsonl")]
    pair_types = [
        ("en", "enpara"),
        ("zh", "zhpara"),
        ("en", "zh"),
        ("enpara", "zhpara")
    ]
    pair_labels = ["EN-ENPARA", "ZH-ZHPARA", "EN-ZH", "ENPARA-ZHPARA"]
    is_chinese_map = {"en": False, "zh": True, "enpara": False, "zhpara": True}
    all_layer_num = None
    # 存储每组配对的所有样本的相似度
    sim_dict = {k: [] for k in pair_labels}
    for theme in tqdm(theme_names, desc="主题遍历"):
        # 加载四个结构的pt
        data = {}
        for struct in ["en", "zh", "enpara", "zhpara"]:
            pt_path = OUTPUT_DIRS[struct] / f"{theme}.pt"
            if not pt_path.exists():
                break
            data[struct] = torch.load(pt_path)
        if len(data) < 4:
            continue
        sample_num = len(data["en"])
        for i in range(sample_num):
            # 取出四个结构的样本
            sample = {k: data[k][i] for k in data}
            # period前token位置
            pos = {k: find_period_position(sample[k]["tokens_zh"] if is_chinese_map[k] else sample[k]["tokens_en"], is_chinese_map[k]) for k in data}
            if any(p == -1 for p in pos.values()):
                continue
            # 取hidden_states
            hiddens = {k: sample[k]["hidden_states_zh"] if is_chinese_map[k] else sample[k]["hidden_states_en"] for k in data}
            # 取is_correct
            is_correct = {k: sample[k]["is_correct_zh"] if is_chinese_map[k] else sample[k]["is_correct_en"] for k in data}
            # 每组配对
            for (k1, k2), label in zip(pair_types, pair_labels):
                # 两个都正确
                if not (is_correct[k1] and is_correct[k2]):
                    continue
                # 逐层取period前hidden
                vecs1 = [h[0, pos[k1]].numpy() for h in hiddens[k1]]
                vecs2 = [h[0, pos[k2]].numpy() for h in hiddens[k2]]
                if all_layer_num is None:
                    all_layer_num = len(vecs1)
                # 逐层cosine similarity
                sim = [cosine_similarity(vecs1[i].reshape(1, -1), vecs2[i].reshape(1, -1))[0,0] for i in range(len(vecs1))]
                sim_dict[label].append(sim)
    # 统计平均
    avg_sim = {label: np.mean(sim_dict[label], axis=0) if sim_dict[label] else np.zeros(all_layer_num) for label in pair_labels}


if __name__ == "__main__":
    main() 