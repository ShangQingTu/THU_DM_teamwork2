import pandas as pd
import numpy as np
import random

nominal_cols = ["race", "gender", "admission_type_id", "discharge_disposition_id", "admission_source_id", "change",
                "diabetesMed", "metformin", "glimepiride", "glipizide", "glyburide", "pioglitazone",
                "rosiglitazone", "insulin"]


class Calculator:
    def __init__(self, df, k):
        self.k = k
        self.numerical_ids = []
        self.nominal_ids = []
        # 记录numerical和nominal属性的id
        for i, col in enumerate(df.columns.values):
            if col in nominal_cols:
                self.nominal_ids.append(i)
            else:
                self.numerical_ids.append(i)

    def get_matrix_split(self, u, node, i):
        # TODO
        return 0

    def get_matrix_merge(self, u, node):
        # TODO
        return 0

    def cal_distance(self, node, center):
        # 　对于连续有序的numerical属性, 用闵可夫斯基距离
        minkowski = np.sum(np.square(node[self.numerical_ids] - center[self.numerical_ids]))
        # 　对于离散无序的nominal属性, 用VDM(Value Difference Metric)
        vdm = 0
        for u in self.nominal_ids:
            for i in range(self.k):
                m_uai = self.get_matrix_split(u, node, i)
                m_ubi = self.get_matrix_split(u, node, i)
                m_ua = self.get_matrix_merge(u, center)
                m_ub = self.get_matrix_merge(u, center)
                vdm += (m_uai / m_ua - m_ubi / m_ub) ** 2

        return np.sqrt(minkowski + vdm)

    def cal_variance(self, cluster_dict, center):
        v_sum = 0
        for i in range(len(center)):
            cluster = cluster_dict[i]
            for j in cluster:
                v_sum += self.cal_distance(j, center[i])
        return v_sum


class KMeans:
    # partitioning-based
    def __init__(self, k, max_iter, min_variance=0.1):
        self.k = k
        self.max_iter = max_iter
        self.min_variance = min_variance
        self.calculator = None

    def random_center(self, data):
        data = list(data)
        return random.sample(data, self.k)

    def get_cluster(self, data, center):
        cluster_dict = dict()
        for node in data:
            # 初始化
            cluster_class = -1
            min_distance = float('inf')
            for i in range(self.k):
                dist = self.calculator.cal_distance(node, center[i])
                if dist < min_distance:
                    # 分配到最近的center
                    cluster_class = i
                    min_distance = dist
            if cluster_class not in cluster_dict.keys():
                cluster_dict[cluster_class] = []
            cluster_dict[cluster_class].append(node)
        return cluster_dict

    def get_avg_center(self, cluster_dict):
        new_center = []
        for i in range(self.k):
            # TODO
            # 对于类别属性,选出现次数最多的 而非简单平均
            center = np.mean(cluster_dict[i], axis=0)
            new_center.append(center)
        return new_center

    def predict(self, df):
        # 数据准备
        self.calculator = Calculator(df, self.k)
        data = df.to_numpy()
        # 初始随机选中心
        center = self.random_center(data)
        cluster_dict = self.get_cluster(data, center)
        new_variance = self.calculator.cal_variance(cluster_dict, center)
        old_variance = 1
        _iter = 0
        # 直到整体的中心到点的距离variance小于一个min_variance值 或者　迭代到max_iter次
        while abs(old_variance - new_variance) > self.min_variance:
            if _iter >= self.max_iter:
                break
            # 重新选平均值点作为簇中心
            center = self.get_avg_center(cluster_dict)
            cluster_dict = self.get_cluster(data, center)
            old_variance = new_variance
            new_variance = self.calculator.cal_variance(cluster_dict, center)
            _iter += 1
        print("Iteration of KMeans is, ", _iter)
        return cluster_dict, center


class Clique:
    # grid-based
    def __init__(self):
        pass

    def predict(self, df):
        pass


class DBSCAN:
    # Density-based
    def __init__(self):
        pass

    def predict(self, df):
        pass


if __name__ == '__main__':
    df_data = pd.DataFrame([[1, 1, 1], [2, 2, 2], [1, 2, 1], [9, 8, 7], [7, 8, 9], [8, 9, 7]], columns=['a', 'b', 'c'])
    print(df_data)
    clf = KMeans(2, 114)
    cluster_dict1, center1 = clf.predict(df_data)
    print(cluster_dict1)
    print("center:")
    print(center1)
