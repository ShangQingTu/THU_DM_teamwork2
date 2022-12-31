# THU_DM_teamwork2
Code for the Hospital Readmission Clustering Analysis task.





# Model

在./model/cluster.py中实现了2个聚类方法：

- 基于partitioning的K-Means
- ~~基于Density的DBSCAN~~
- 基于网格的聚类算法Clique

## Evaluation

| 评价指标           | K-Means       | Clique                 |
| ------------------ | ------------- | ---------------------- |
| 算法性能           | local optimum | simple but low quality |
| 稳定度             |               |                        |
| 复杂度             | **$O(tkn)$**  | **$O(n^2)$** (worst)   |
| Hopkins  Statistic |               |                        |
| 紧凑度             |               |                        |
| 分离度             |               |                        |

其中，t是迭代轮数，k是k-means的簇数，n是样本数量

还有一些算出的数值指标:

- 聚类趋势
  - Hopkins  Statistic

- 聚类质量
  - 紧凑度
  - 分离度
