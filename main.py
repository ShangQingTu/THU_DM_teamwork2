import pandas as pd
import numpy as np
import os
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler
from model.cluster import KMeans, DBSCAN, Clique


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


# 用于聚类模型
def main(args):
    std = StandardScaler()
    # 加载数据,　drop标签列, 归一化
    data_name = args['data_name']
    data = pd.read_csv('data/' + data_name + '.csv', encoding='utf-8')
    _data_x = drop_irregular_columns(data)
    data_x = pd.DataFrame(std.fit_transform(_data_x))
    # 选模型
    if args['model_name'] == 'KMeans':
        clf = KMeans()
    elif args['model_name'] == 'Clique':
        clf = Clique()
    elif args['model_name'] == 'DBSCAN':
        clf = DBSCAN()
    else:
        clf = None
    # 聚类
    cluster_result = clf.predict(data_x)
    # print(accuracy_score(Y_test, Y_result), roc_auc_score(Y_test, Y_result), f1_score(Y_test, Y_result))
    # import pdb;pdb.set_trace()
    return cluster_result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    # model and data
    parser.add_argument('--model_name', type=str, default='KMeans',
                        choices=['KMeans', 'Clique', 'DBSCAN'])
    parser.add_argument('--data_name', type=str, default='diabetic_data',
                        choices=['diabetic_data', 'processed_data'])
    # parser.add_argument('--train_size', type=float, default=0.8)
    # hyper parameters
    parser.add_argument('--k', type=int, default=3, help="k for k-means")
    args = parser.parse_args()
    args = vars(args)
    print(main(args))
