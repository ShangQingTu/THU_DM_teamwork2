import pandas as pd
import numpy as np
from tqdm import tqdm
import random
import time

nominal_cols = ["race", "gender", "admission_type_id", "discharge_disposition_id", "admission_source_id", "change",
                "diabetesMed", "metformin", "glimepiride", "glipizide", "glyburide", "pioglitazone",
                "rosiglitazone", "insulin"]


class Calculator:
    def __init__(self, df, k, data, only_numeric=False):
        self.only_numeric = only_numeric
        self.k = k
        self.numerical_ids = []
        self.nominal_ids = []
        # 记录numerical和nominal属性的id
        for i, col in enumerate(df.columns.values):
            if col in nominal_cols:
                if only_numeric:
                    self.numerical_ids.append(i)
                else:
                    self.nominal_ids.append(i)
            else:
                self.numerical_ids.append(i)
        self.id2count_split = {}  # key: cluster_id, value: a dict like self.id2count_merge
        self.id2count_merge = {}  # key: column_id, value: Dict of Counter, Counter: {data's value :count}
        self.init_count(data)

    def init_count(self, data):
        # 初始化总体的counter
        for col_id in self.nominal_ids:
            unique, counts = np.unique(data[:, col_id], return_counts=True)
            count_merge = {}
            for i, key in enumerate(unique):
                count_merge[key] = counts[i]
            self.id2count_merge[col_id] = count_merge

    def update_count(self, cluster_dict):
        if self.only_numeric:
            return
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
        try:
            return self.id2count_split[i][u][node[u]]
        except KeyError:
            return 0

    def get_matrix_merge(self, u, node):
        return self.id2count_merge[u][node[u]]

    def cal_distance(self, node, center):
        # 　对于连续有序的numerical属性, 用闵可夫斯基距离
        minkowski = np.sum(np.square(node[self.numerical_ids] - center[self.numerical_ids]))
        if self.only_numeric:
            return np.sqrt(minkowski)
        # 　对于离散无序的nominal属性, 用VDM(Value Difference Metric)
        vdm = 0
        for u in self.nominal_ids:
            m_ua = self.get_matrix_merge(u, node)
            m_ub = self.get_matrix_merge(u, center)
            for i in range(len(self.id2count_split)):
                m_uai = self.get_matrix_split(u, node, i)
                m_ubi = self.get_matrix_split(u, center, i)
                vdm += (m_uai / m_ua - m_ubi / m_ub) ** 2

        return np.sqrt(minkowski + vdm)

    def cal_variance(self, cluster_dict, center):
        v_sum = 0
        for i in range(len(center)):
            cluster = cluster_dict[i]
            for j in cluster:
                v_sum += self.cal_distance(j, center[i])
        return v_sum

class GowerCalculator(Calculator):
    def __init__(self, df):
        self.numerical_ids = []
        self.nominal_ids = []
        # 记录numerical和nominal属性的id
        for i, col in enumerate(df.columns.values):
            if col in nominal_cols:
                self.nominal_ids.append(i)
            else:
                self.numerical_ids.append(i)
        # cat_features = [c in nominal_cols for c in df.columns]
        # cat_features = np.bool8(cat_features)
        # self.nominal_ids = cat_features
        # self.numerical_ids = ~cat_features
        num_data = df.iloc[:, self.numerical_ids].values
        # self.nominal_ids = np.concatenate([self.nominal_ids,[False]])
        # self.numerical_ids = np.concatenate([self.numerical_ids,[False]])
        self.numerical_range = num_data.max(0) - num_data.min(0)
        self.numerical_range[self.numerical_range == 0] = 1
        del num_data

    def cal_distance(self, x, y):
        # x, y = x[:-1], y[:-1]
        num_dist = (1 - np.abs(x[self.numerical_ids] - y[self.numerical_ids]) / self.numerical_range).sum()
        if only_numeric:
            cat_dist = 0
            dim = len(self.numerical_ids)
        else:
            cat_dist = (x[self.nominal_ids] == y[self.nominal_ids]).sum()
            dim = len(self.nominal_ids) + len(self.numerical_ids)
        return 1 - (num_dist + cat_dist) / dim
    
    def update_count(self, cluster_dict):
        pass

class KMeans:
    # partitioning-based
    def __init__(self, k, max_iter, data_reduction, min_variance=0.1):
        self.k = k
        self.max_iter = max_iter
        self.data_reduction = data_reduction
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
        for k, v in cluster_dict.items():
            np_data = np.asarray(v)
            np_cluster_dict[k] = np_data
        self.calculator.update_count(np_cluster_dict)
        return cluster_dict, np_cluster_dict

    def get_avg_center(self, np_cluster_dict):
        new_centers = []
        for i in range(self.k):
            # 初始化一个点
            center_list = np_cluster_dict[i][0]
            # 算平均
            cluster_nodes = np_cluster_dict[i]
            avg = np.mean(cluster_nodes[:, self.calculator.numerical_ids], axis=0)
            center_list[self.calculator.numerical_ids] = avg
            # 对于类别属性,选出现次数最多的 而非简单平均 (这种做法有待商榷)
            for col_id in self.calculator.nominal_ids:
                # np.unique默认是降序
                unique = np.unique(cluster_nodes[:, col_id])
                center_list[col_id] = unique[0]
            new_centers.append(center_list)
        return new_centers

    def predict(self, df):
        # 数据准备
        if isinstance(df, pd.DataFrame):
            # df = df.drop(columns=["Unnamed: 0"])
            data = df.to_numpy()
        elif isinstance(df, np.ndarray):
            data = df
        ids = [[i] for i in range(len(data))]
        ids_np = np.asarray(ids)
        # print(ids_np.shape)
        # print(data.shape)
        data = np.concatenate([data, ids_np], axis=1)
        print(data.shape)
        if self.data_reduction == "get_dummies":
            self.calculator = Calculator(df, self.k, data, only_numeric=True)
        else:
            self.calculator = Calculator(df, self.k, data, only_numeric=False)
        # 　假设一开始都在一个cluster上
        self.calculator.update_count({0: data})
        # 初始随机选中心
        center_list = self.random_center(data)
        cluster_dict, np_cluster_dict = self.get_cluster(data, center_list)
        new_variance = self.calculator.cal_variance(cluster_dict, center_list)
        old_variance = 1
        _iter = 0
        # 直到整体的中心到点的距离variance小于一个min_variance值 或者　迭代到max_iter次
        start_time = time.time()
        while abs(old_variance - new_variance) > self.min_variance:
            end_time = time.time()
            print(f"Iter {_iter}, Cost {end_time - start_time}, Loss {old_variance/len(df)}")
            if _iter >= self.max_iter:
                break
            # 重新选平均值点作为簇中心
            center_list = self.get_avg_center(np_cluster_dict)
            cluster_dict, np_cluster_dict = self.get_cluster(data, center_list)
            old_variance = new_variance
            new_variance = self.calculator.cal_variance(cluster_dict, center_list)
            _iter += 1
        print("End Iteration of KMeans is, ", _iter)
        node_id2label = [0] * len(data)
        for label, nodes in cluster_dict.items():
            for node in nodes:
                node_id = node[-1]
                node_id2label[node_id] = label
        return node_id2label, center_list


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


class FastCalculator(Calculator):
    """
    专门给HierarchicalCluster设计的 FastCalculator
    """

    def __init__(self, df, k, data, real_fast=True):
        self.real_fast = real_fast
        self.id2count_merge_current = {}  # 记录当前时刻的$m_{u,a}^*$
        self.id2count_reverse_square = {}  # 记录 $ \frac{1}{m_{u,a}^2} $
        self.vdm_log = {}  # 记录 $  \frac{ m_{u,a}^{*}  }{m_{u,a}^2} $
        super().__init__(df, k, data)

    def init_count(self, data):
        super().init_count(data)
        # 初始化 id2count_merge_current 与 id2count_reverse_square
        self.id2count_merge_current = self.id2count_merge
        for u, counter in self.id2count_merge.items():
            reverse_square_counter = {}
            for k, c in counter.items():
                reverse_square_counter[k] = 1 / (c * c)
            self.id2count_reverse_square[u] = reverse_square_counter
        self.update_vdm()

    def update_vdm(self):
        # 预先计算　 $  \frac{ m_{u,a}^{*}  }{m_{u,a}^2} $
        for u, counter in self.id2count_merge_current.items():
            vdm_log = {}
            for k, c in counter.items():
                vdm_log[k] = c * self.id2count_reverse_square[u][k]
            self.vdm_log[u] = vdm_log

    def decrease_count(self, node1, node2, new_node):
        # 对于Hierarchical聚类,任何时候每个cluster都是一个合并好的簇,其属性都只有1种取值
        # k在若干次合并后会逐渐减少, id2count_merge_current　需要更新
        vec1 = node1.vec
        vec2 = node2.vec
        new_vec = new_node.vec
        for u, counter in self.id2count_merge_current.items():
            # 新簇的value
            new_value = new_vec[u]
            counter[new_value] += 1
            # 合并掉的2个簇的value
            missing_value1 = vec1[u]
            counter[missing_value1] -= 1
            missing_value2 = vec2[u]
            counter[missing_value2] -= 1

        self.update_vdm()

    def cal_distance(self, node, center):
        # 　对于连续有序的numerical属性, 用闵可夫斯基距离
        minkowski = np.sum(np.square(node[self.numerical_ids] - center[self.numerical_ids]))
        if self.real_fast:
            return np.sqrt(minkowski)
        # 　对于离散无序的nominal属性, 用VDM(Value Difference Metric)
        vdm = 0
        for u in self.nominal_ids:
            a = node[u]
            b = center[u]
            if a == b:
                # $ VDM_p(a,b) $ 　　= 0
                continue
            else:
                vdm_log = self.vdm_log[u]
                vdm += vdm_log[a] + vdm_log[b]

        return np.sqrt(minkowski + vdm)


class ClusterNode:
    def __init__(self, vec, left=None, right=None, distance=-1, _id=None, count=1):
        """
        :param vec: 保存两个数据聚类后形成新的中心
         :param left: 左节点
         :param right:  右节点
         :param distance: 两个节点的距离
         :param _id: 用来标记哪些节点是计算过的
         :param count: 这个节点的叶子节点个数
        """
        self.vec = vec
        self.left = left
        self.right = right
        self.distance = distance
        self.id = _id
        self.count = count


class HierarchicalCluster:
    def __init__(self, k=1):
        self.k = k
        self.calculator = None
        self.labels = None
        self.nodes = None

    def predict(self, df):
        # 数据准备
        data = df.to_numpy()
        self.calculator = FastCalculator(df, self.k, data)
        nodes = [ClusterNode(vec=v, _id=i) for i, v in enumerate(data)]
        distances = {}
        # 特征的维度
        point_num, future_num = np.shape(data)
        self.labels = [-1] * point_num
        current_clust_id = -1
        start_time = time.time()
        while len(nodes) > self.k:
            end_time = time.time()
            print(f"Iter {len(df) - len(nodes)}, Cost {end_time - start_time}")
            min_dist = float('inf')
            nodes_len = len(nodes)
            closest_part = None  # 表示最相似的两个聚类
            for i in tqdm(range(nodes_len - 1)):
                for j in range(i + 1, nodes_len):
                    # 为了不重复计算距离，保存在字典内
                    d_key = (nodes[i].id, nodes[j].id)
                    if d_key not in distances:
                        distances[d_key] = self.calculator.cal_distance(nodes[i].vec, nodes[j].vec)
                    d = distances[d_key]
                    if d < min_dist:
                        min_dist = d
                        closest_part = (i, j)
            # 合并两个聚类
            part1, part2 = closest_part
            node1, node2 = nodes[part1], nodes[part2]
            new_vec = self.combine(node1, node2)
            new_node = ClusterNode(vec=new_vec,
                                   left=node1,
                                   right=node2,
                                   distance=min_dist,
                                   _id=current_clust_id,
                                   count=node1.count + node2.count)
            if not self.calculator.real_fast:
                self.calculator.decrease_count(node1, node2, new_node)
            current_clust_id -= 1
            del nodes[part2], nodes[part1]  # 一定要先del索引较大的
            nodes.append(new_node)
        self.nodes = nodes
        self.calc_label()
        return self.labels, None

    def combine(self, node1, node2):
        """
        合并2个簇
        """
        vec1 = node1.vec
        vec2 = node2.vec
        new_vec = vec1
        sum_count = node1.count + node2.count
        # 对于numerical属性,加权合并
        new_vec[self.calculator.numerical_ids] = (vec1[self.calculator.numerical_ids] * node1.count + vec2[
            self.calculator.numerical_ids] * node2.count) / sum_count
        # 对于nominal属性,按大者合并
        # 这种合并能算得快些, 但有待商榷
        if not self.calculator.real_fast:
            for u in self.calculator.nominal_ids:
                if vec1[u] != vec2[u]:
                    if node2.count > node1.count:
                        new_vec[u] = vec2[u]
        return new_vec

    def calc_label(self):
        """
        调取聚类的结果
        """
        for i, node in enumerate(self.nodes):
            # 将节点的所有叶子节点都分类
            self.leaf_traversal(node, i)

    def leaf_traversal(self, node: ClusterNode, label):
        """
        递归遍历叶子节点
        """
        if node.left is None and node.right is None:
            self.labels[node.id] = label
        if node.left:
            self.leaf_traversal(node.left, label)
        if node.right:
            self.leaf_traversal(node.right, label)


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
