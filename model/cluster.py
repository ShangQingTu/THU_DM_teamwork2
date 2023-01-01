import pandas as pd
import numpy as np
import random

nominal_cols = ["race", "gender", "admission_type_id", "discharge_disposition_id", "admission_source_id", "change",
                "diabetesMed", "metformin", "glimepiride", "glipizide", "glyburide", "pioglitazone",
                "rosiglitazone", "insulin"]


class Calculator:
    def __init__(self, df, k, data):
        self.k = k
        self.numerical_ids = []
        self.nominal_ids = []
        # 记录numerical和nominal属性的id
        for i, col in enumerate(df.columns.values):
            if col in nominal_cols:
                self.nominal_ids.append(i)
            else:
                self.numerical_ids.append(i)
        self.id2count_split = {}  # key: cluster_id, value: a dict like self.id2count_merge
        self.id2count_merge = {}  # key: column_id, value: Dict of Counter, Counter: {data's value :count}
        self.init_count(data)
        # 　假设一开始都在一个cluster上
        self.update_count({0: data})

    def init_count(self, data):
        # 初始化总体的counter
        for col_id in self.nominal_ids:
            unique, counts = np.unique(data[:, col_id], return_counts=True)
            count_merge = {}
            for i, key in enumerate(unique):
                count_merge[key] = counts[i]
            self.id2count_merge[col_id] = count_merge

    def update_count(self, cluster_dict):
        # 每轮更新每个cluster上的counter
        for cluster_id, _data in cluster_dict.items():
            id2count_merge = {}
            for col_id in self.nominal_ids:
                unique, counts = np.unique(_data[:, col_id], return_counts=True)
                count_merge = {}
                for i, key in enumerate(unique):
                    count_merge[key] = counts[i]
                id2count_merge[col_id] = count_merge
            self.id2count_split[cluster_id] = id2count_merge

    def get_matrix_split(self, u, node, i):
        return self.id2count_split[i][u][node[u]]

    def get_matrix_merge(self, u, node):
        return self.id2count_merge[u][node[u]]

    def cal_distance(self, node, center):
        # 　对于连续有序的numerical属性, 用闵可夫斯基距离
        minkowski = np.sum(np.square(node[self.numerical_ids] - center[self.numerical_ids]))
        # 　对于离散无序的nominal属性, 用VDM(Value Difference Metric)
        vdm = 0
        for u in self.nominal_ids:
            for i in range(self.k):
                m_uai = self.get_matrix_split(u, node, i)
                m_ubi = self.get_matrix_split(u, center, i)
                m_ua = self.get_matrix_merge(u, node)
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
        # 更新counter
        np_cluster_dict = {}
        for k, v in cluster_dict:
            np_data = np.asarray(v)
            np_cluster_dict[k] = np_data
        self.calculator.update_count(np_cluster_dict)
        return cluster_dict, np_cluster_dict

    def get_avg_center(self, np_cluster_dict):
        new_center = []
        for i in range(self.k):
            # 初始化一个点
            center = np_cluster_dict[i][0]
            # 算平均
            cluster_nodes = np_cluster_dict[i]
            avg = np.mean(cluster_nodes[:, self.calculator.numerical_ids], axis=0)
            center[self.calculator.numerical_ids] = avg
            # 对于类别属性,选出现次数最多的 而非简单平均 (TODO　这种做法有待商榷)
            for col_id in self.calculator.nominal_ids:
                # np.unique默认是降序
                unique = np.unique(cluster_nodes[:, col_id])
                center[col_id] = unique[0]
            new_center.append(center)
        return new_center

    def predict(self, df):
        # 数据准备
        data = df.to_numpy()
        self.calculator = Calculator(df, self.k, data)
        # 初始随机选中心
        center = self.random_center(data)
        cluster_dict, np_cluster_dict = self.get_cluster(data, center)
        new_variance = self.calculator.cal_variance(cluster_dict, center)
        old_variance = 1
        _iter = 0
        # 直到整体的中心到点的距离variance小于一个min_variance值 或者　迭代到max_iter次
        while abs(old_variance - new_variance) > self.min_variance:
            if _iter >= self.max_iter:
                break
            # 重新选平均值点作为簇中心
            center = self.get_avg_center(np_cluster_dict)
            cluster_dict, np_cluster_dict = self.get_cluster(data, center)
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
    df_data = pd.DataFrame(
        [[1, 1, 'a', 'c'], [2, 2, 'a', 'd'], [1, 2, 'a', 'a'], [9, 8, 'a', 'a'], [7, 8, 'b', 'a'], [8, 9, 'b', 'a']],
        columns=['a', 'b', 'insulin', 'glimepiride'])
    print(df_data)
    clf = KMeans(2, 114)
    cluster_dict1, center1 = clf.predict(df_data)
    print(cluster_dict1)
    print("center:")
    print(center1)
