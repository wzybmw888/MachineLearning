import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.neighbors import KDTree, BallTree

# 生成随机数据集
X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)

# 定义最近邻数量
k = 5

# 构建KD树和Ball树，并计算查询时间
kdt = KDTree(X)
btree = BallTree(X)

kdt_query_times = []
btree_query_times = []
bf_query_times = []

for i in range(100):
    query_point = np.random.randn(1, 10)

    start_time = time.time()
    kdt.query(query_point, k=k)
    kdt_query_times.append(time.time() - start_time)

    start_time = time.time()
    btree.query(query_point, k=k)
    btree_query_times.append(time.time() - start_time)

    start_time = time.time()
    distances = []
    for j in range(X.shape[0]):
        dist = np.linalg.norm(query_point - X[j])
        distances.append((dist, y[j]))
    distances.sort()
    top_k = distances[:k]
    bf_query_times.append(time.time() - start_time)

# 绘制时间对比图
plt.figure(figsize=(8, 6))
plt.plot(kdt_query_times, label='KDTree')
plt.plot(btree_query_times, label='BallTree')
plt.plot(bf_query_times, label='Brute Force')
plt.xlabel('Query Index')
plt.ylabel('Query Time (s)')
plt.title(f'Time Comparison of KDTree, BallTree and Brute Force KNN (k={k})')
plt.legend()
plt.show()