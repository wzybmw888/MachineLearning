#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 在这里写下你的代码
import numpy as np


# 将1024维特征降维到64
def reduce_dim(feature, Dim=64):
    X = feature.reshape((32, 32))
    x = np.zeros(Dim)
    for i_r in range(8):
        for i_c in range(8):
            patch = X[i_r * 4:i_r * 4 + 4, i_c * 4:i_c * 4 + 4]
            x[i_r * 8 + i_c] = patch.sum()
    return x


# 从文本文件读取1024维特征
# 同时返回降维后的特征及标签
def read_from_txt(file_name, Dim1=1024, Dim2=64):
    fid = open(file_name, 'r')
    datas = fid.readlines()
    X_1024 = np.zeros((len(datas), Dim1), dtype=np.float64)
    X_64 = np.zeros((len(datas), Dim2), dtype=np.float64)
    Y = np.zeros(len(datas), dtype=np.uint8)
    for idx, data in enumerate(datas):
        img_1024 = [np.float64(d) for d in data[:Dim1]]
        img_1024 = np.array(img_1024)
        X_1024[idx, :] = img_1024
        X_64[idx, :] = reduce_dim(img_1024)
        Y[idx] = np.uint8(data[Dim1])
    fid.close()
    return X_1024, X_64, Y


# In[2]:


X_train_1024, X_train_64, Y_train = read_from_txt(
    r'UCI_digits.train',
    Dim1=1024, Dim2=64)
# 读取测试集
X_test_1024, X_test_64, Y_test = read_from_txt(
    r'UCI_digits.test',
    Dim1=1024, Dim2=64)

# 把数据合在一起
X_1024 = np.concatenate((X_train_1024, X_test_1024))
X_64 = np.concatenate((X_train_64, X_test_64))
Y = np.concatenate((Y_train, Y_test))

# In[3]:


from sklearn.neighbors import KNeighborsClassifier, KDTree, BallTree
from sklearn.metrics import accuracy_score
import numpy as np
import time
import matplotlib.pyplot as plt


def knn_train_rate_compare(X=X_64, Y=Y, feature_n=None):
    if feature_n:
        selected_features = np.random.choice(range(64), feature_n, replace=False)  # 随机选择 16 个特征
        X = X[:, selected_features]  # 构造新的特征矩阵，仅包含选中的 16 个特征

    # 随机选择500个样本作为测试集
    test_indices = np.random.choice(len(X), size=500, replace=False)
    X_test = X[test_indices]
    y_test = Y[test_indices]

    # 记录查询时间
    brute_times = []
    kd_times = []
    ball_times = []

    # 选择不同的训练样本数量
    num_samples = [50, 100, 400, 800, 1600, 3200, 5000]

    # 对每种训练样本数量，分别使用暴力搜索、KDTree和BallTree三种算法进行查询，并记录查询时间
    for n in num_samples:
        # 随机选择n个样本作为训练集
        train_indices = np.random.choice(len(X), size=n, replace=False)
        X_train = X[train_indices]
        y_train = Y[train_indices]

        # 使用暴力搜索算法
        start = time.time()
        knn = KNeighborsClassifier(n_neighbors=5, algorithm='brute')
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        brute_times.append(time.time() - start)

        # 使用KDTree算法
        start = time.time()
        knn = KNeighborsClassifier(n_neighbors=5, algorithm='kd_tree')
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        kd_times.append(time.time() - start)

        # 使用BallTree算法
        start = time.time()
        knn = KNeighborsClassifier(n_neighbors=5, algorithm='ball_tree')
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        ball_times.append(time.time() - start)

    plt.plot(num_samples, brute_times, color='red', linestyle='-', linewidth=2, marker='o', label='Brute Force')
    plt.plot(num_samples, kd_times, color='green', linestyle=':', linewidth=2, marker='*', markersize=8, label='KDTree')
    plt.plot(num_samples, ball_times, color='blue', linestyle='--', linewidth=2, marker='^', markersize=6,
             markerfacecolor='none', markeredgewidth=1, label='BallTree')
    plt.legend(loc='upper left', fontsize=12)
    if feature_n:
        plt.xlabel(f'Number of training samples with shape {feature_n}', fontsize=14)
    plt.ylabel('Train time', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True)
    plt.show()


# In[4]:


knn_train_rate_compare(feature_n=16)

# In[5]:


knn_train_rate_compare(feature_n=64)


# In[8]:


def knn_search_rate_compare(X=X_64, Y=Y, feature_n=None):
    if feature_n:
        selected_features = np.random.choice(range(64), feature_n, replace=False)  # 随机选择 n个特征
        X = X[:, selected_features]  # 构造新的特征矩阵，仅包含选中的n个特征
    n_query = 500
    query_points = np.random.choice(len(X), 500)

    # 定义训练集大小的不同取值
    train_sizes = [10, 50, 100, 200, 300, 400, 500, 1000]

    # 定义查询时间列表
    brute_times = []
    kd_times = []
    ball_times = []

    # 迭代计算不同大小的训练集的查询时间
    for train_size in train_sizes:
        X_train = X[:train_size]

        # 暴力搜索
        start = time.time()
        distances, indices = [], []
        for query_point in X[query_points]:
            dist = np.linalg.norm(X_train - query_point, axis=1)
            idx = np.argsort(dist)
            distances.append(dist[idx])
            indices.append(idx)
        end = time.time()
        brute_times.append((end - start) / n_query)

        # KD树
        start = time.time()
        kdt = KDTree(X_train)
        kdt.query(X[query_points], k=1, return_distance=False)
        end = time.time()
        kd_times.append((end - start) / n_query)

        # Ball树
        start = time.time()
        bdt = BallTree(X_train)
        bdt.query(X[query_points], k=1, return_distance=False)
        end = time.time()
        ball_times.append((end - start) / n_query)

    # 绘制查询时间与训练集大小的关系图
    plt.plot(train_sizes, brute_times, color='red', linestyle='-', linewidth=2, marker='o', label='Brute Force')
    plt.plot(train_sizes, kd_times, color='green', linestyle=':', linewidth=2, marker='*', markersize=8, label='KDTree')
    plt.plot(train_sizes, ball_times, color='blue', linestyle='--', linewidth=2, marker='^', markersize=6,
             markerfacecolor='none', markeredgewidth=1, label='BallTree')
    plt.legend(loc='upper left', fontsize=12)
    if feature_n:
        plt.xlabel(f'Number of training samples with shape {feature_n}', fontsize=14)
    plt.ylabel('Search time', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True)
    plt.show()


# In[9]:


knn_search_rate_compare(feature_n=64)

# In[10]:


knn_search_rate_compare(feature_n=16)

# In[ ]: