import pandas as pd
import numpy as np
from collections import defaultdict
import networkx as nx
from scipy.stats import norm

# 数据加载
data = {
    "编号": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15"],
    "色泽": ["青绿", "乌黑", "浅白", "青绿", "浅白", "乌黑", "青绿", "浅白", "乌黑", "浅白", "青绿", "乌黑", "青绿", "浅白", "乌黑"],
    "根蒂": ["蜷缩", "蜷缩", "蜷缩", "蜷缩", "稍蜷", "稍蜷", "硬挺", "硬挺", "蜷缩", "蜷缩", "稍蜷", "稍蜷", "蜷缩", "蜷缩", "蜷缩"],
    "敲声": ["浊响", "沉闷", "清脆", "浊响", "浊响", "浊响", "清脆", "浊响", "浊响", "沉闷", "沉闷", "浊响", "浊响", "浊响", "浊响"],
    "纹理": ["清晰", "清晰", "清晰", "清晰", "清晰", "稍糊", "模糊", "模糊", "稍糊", "稍糊", "模糊", "稍糊", "稍糊", "模糊", "清晰"],
    "脐部": ["凹陷", "凹陷", "凹陷", "稍凹", "稍凹", "稍凹", "平坦", "平坦", "稍凹", "凹陷", "凹陷", "稍凹", "稍凹", "稍凹", "凹陷"],
    "触感": ["硬滑", "硬滑", "硬滑", "软粘", "硬滑", "硬滑", "软粘", "硬滑", "软粘", "硬滑", "硬滑", "软粘", "硬滑", "硬滑", "软粘"],
    "密度": [0.697, 0.774, 0.634, 0.608, 0.556, 0.403, 0.481, 0.437, 0.666, 0.243, 0.245, 0.343, 0.639, 0.657, 0.360],
    "含糖量": [0.460, 0.376, 0.264, 0.318, 0.215, 0.237, 0.149, 0.211, 0.091, 0.267, 0.057, 0.099, 0.161, 0.198, 0.370],
    "好瓜": ["是", "是", "是", "是", "否", "否", "否", "否", "否", "否", "否", "否", "是", "是", "否"]
}
df = pd.DataFrame(data)

# 特征列表
features = ["色泽", "根蒂", "敲声", "纹理", "脐部", "触感", "密度", "含糖量"]
target = "好瓜"

# 计算条件互信息
def calculate_conditional_mutual_info(df, feature1, feature2, target):
    joint_probs = pd.crosstab([df[feature1], df[feature2]], df[target], normalize=True)
    feature1_probs = df[feature1].value_counts(normalize=True)
    feature2_probs = df[feature2].value_counts(normalize=True)
    target_probs = df[target].value_counts(normalize=True)

    mutual_info = 0
    for (f1, f2), _ in joint_probs.iterrows():
        for t in target_probs.index:
            p_joint = joint_probs.loc[(f1, f2), t]
            if p_joint > 0:
                mutual_info += p_joint * np.log(p_joint / (feature1_probs[f1] * feature2_probs[f2] * target_probs[t]))
    return mutual_info

# 构建加权图
edges = []
for i, f1 in enumerate(features[:-2]):  # 离散变量
    for f2 in features[i+1:-2]:
        cmi = calculate_conditional_mutual_info(df, f1, f2, target)
        edges.append((f1, f2, cmi))

graph = nx.Graph()
graph.add_weighted_edges_from(edges)

# 最大生成树
mst = nx.maximum_spanning_tree(graph)

# 确定TAN结构
tan_structure = {}
root_feature = "色泽"
for edge in mst.edges:
    tan_structure[edge[1]] = edge[0] if edge[0] != root_feature else None

# 计算先验概率
def calculate_prior_prob(df, target):
    return df[target].value_counts(normalize=True).to_dict()

# 计算条件概率
def calculate_conditional_prob(df, feature, target, parent=None, smooth=1):
    if feature in ["密度", "含糖量"]:  # 连续变量
        cond_prob = {}
        for t in df[target].unique():
            sub_df = df[df[target] == t]
            mean = sub_df[feature].mean()
            std = sub_df[feature].std()
            cond_prob[t] = {"mean": mean, "std": std}
        return cond_prob
    else:  # 离散变量
        conditional_prob = defaultdict(lambda: defaultdict(float))
        target_values = df[target].unique()

        if parent:
            for t in target_values:
                sub_df = df[df[target] == t]
                parent_values = sub_df[parent].unique()

                for p in parent_values:
                    sub_sub_df = sub_df[sub_df[parent] == p]
                    feature_values = sub_sub_df[feature].value_counts()
                    total = len(sub_sub_df) + smooth * len(df[feature].unique())

                    for fv in df[feature].unique():
                        conditional_prob[(fv, t, p)] = (feature_values.get(fv, 0) + smooth) / total
        else:
            for t in target_values:
                sub_df = df[df[target] == t]
                feature_values = sub_df[feature].value_counts()
                total = len(sub_df) + smooth * len(df[feature].unique())

                for fv in df[feature].unique():
                    conditional_prob[(fv, t)] = (feature_values.get(fv, 0) + smooth) / total

        return conditional_prob

# 模型训练
model = {"prior": calculate_prior_prob(df, target)}
for feature in features:
    parent = tan_structure.get(feature, None)
    model[feature] = calculate_conditional_prob(df, feature, target, parent)

# 使用TAN分类
def classify_tan(sample, model, tan_structure, target_values):
    posterior_probs = {}

    for t in target_values:
        prob = model["prior"][t]

        for feature in features:
            feature_value = sample[feature]

            if feature in ["密度", "含糖量"]:  # 连续变量
                mean = model[feature][t]["mean"]
                std = model[feature][t]["std"]
                prob *= norm.pdf(feature_value, loc=mean, scale=std)
            else:  # 离散变量
                parent = tan_structure.get(feature, None)
                if parent:
                    parent_value = sample[parent]
                    prob *= model[feature].get((feature_value, t, parent_value), 0)
                else:
                    prob *= model[feature].get((feature_value, t), 0)

        posterior_probs[t] = prob

    return max(posterior_probs, key=posterior_probs.get)

# 测试样本
test_samples = [
    {"色泽": "青绿", "根蒂": "蜷缩", "敲声": "浊响", "纹理": "清晰", "脐部": "凹陷", "触感": "硬滑", "密度": 0.697, "含糖量": 0.460},
    {"色泽": "浅白", "根蒂": "蜷缩", "敲声": "沉闷", "纹理": "模糊", "脐部": "稍凹", "触感": "软粘", "密度": 0.463, "含糖量": 0.135},
    {"色泽": "乌黑", "根蒂": "稍蜷", "敲声": "清脆", "纹理": "模糊", "脐部": "凹陷", "触感": "硬滑", "密度": 0.428, "含糖量": 0.208}
]

# 分类
for i, sample in enumerate(test_samples):
    prediction = classify_tan(sample, model, tan_structure, target_values=["是", "否"])
    print(f"测{i+1} 是否为好瓜: {prediction}")
