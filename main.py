import random
import pandas as pd
import numpy as np
import os
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from model.cluster import KMeans, DBSCAN, Clique, HierarchicalCluster


def drop_irregular_columns(data_df):
    # 需要舍弃id和readmitted类型
    must_drops = ['encounter_id', 'patient_nbr', 'readmitted']
    data_df = data_df.drop(columns=must_drops)
    # 舍弃一些现在处理不了的列
    drops = []
    for col in data_df.columns.values:
        accept_col = True
        for v in data_df[col]:
            try:
                _ = int(v)
            except Exception:
                accept_col = False
                break
        if not accept_col:
            drops.append(col)
    final_df = data_df.drop(columns=drops)
    # print(final_df)
    print(f"[0] Use these columns:{final_df.columns.values}")
    return final_df


def set_seeds(seed):
    # 随机种子
    random.seed(seed)
    np.random.seed(seed)


# 用于聚类模型
def main(args):
    std = StandardScaler()
    # 加载数据
    data_name = args['data_name']
    _data_x = pd.read_csv('data/' + data_name + '.csv', encoding='utf-8')
    set_seeds(args['seed'])
    if data_name == 'diabetic_data':
        # 　如果没有preprocess,　则drop标签列, 归一化
        _data_x = drop_irregular_columns(_data_x)
        _data_x = pd.DataFrame(std.fit_transform(_data_x))
    # 是否做one-hot encoding
    if args['data_reduction'] == 'get_dummies':
        _data_x = pd.get_dummies(_data_x)
    # 选模型
    if args['model_name'] == 'KMeans':
        clf = KMeans(args['k'], args['max_iter'], args['data_reduction'])
    elif args['model_name'] == 'Clique':
        clf = Clique()
    elif args['model_name'] == 'DBSCAN':
        clf = DBSCAN()
    elif args['model_name'] == 'Hierarchical':
        clf = HierarchicalCluster(args['k'])
    else:
        clf = None
    # 聚类
    node_id2label = clf.predict(_data_x)
    # 评价
    evaluate(node_id2label, pd.get_dummies(_data_x))


def evaluate(node_id2label, _data_x):
    df = pd.read_csv("./data/label.csv")
    labels_true = list(df["readmitted"])
    #  找数据对齐的标签
    labels = node_id2label
    # print(f"Homogeneity: {metrics.homogeneity_score(labels_true, labels):.5f}")
    # print(f"Completeness: {metrics.completeness_score(labels_true, labels):.5f}")
    # print(f"V-measure: {metrics.v_measure_score(labels_true, labels):.5f}")
    # print(f"Adjusted Rand Index: {metrics.adjusted_rand_score(labels_true, labels):.5f}")
    # print(f"Adjusted Mutual Information: {metrics.adjusted_mutual_info_score(labels_true, labels):.5f}")
    print(f"Silhouette Coefficient: {metrics.silhouette_score(_data_x, labels[0]):.5f}")
    print(f"Calinski Harabasz Index: {metrics.calinski_harabasz_score(_data_x, labels[0]):.5f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    # model and data
    parser.add_argument('--model_name', type=str, default='KMeans',
                        choices=['KMeans', 'Clique', 'DBSCAN', 'Hierarchical'])
    parser.add_argument('--data_name', type=str, default='processed_data',
                        choices=['diabetic_data', 'processed_data'])
    parser.add_argument('--data_reduction', type=str, default='None',
                        choices=['None', 'get_dummies'])
    # parser.add_argument('--train_size', type=float, default=0.8)
    # hyper parameters
    parser.add_argument('--seed', type=int, default=1453, help="random seed")
    parser.add_argument('--k', type=int, default=5, help="k for k-means")
    parser.add_argument('--max_iter', type=int, default=128, help="max iteration for k-means")
    args = parser.parse_args()
    args = vars(args)
    print(main(args))
