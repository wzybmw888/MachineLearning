# 欧几里得距离（Euclidean Distance）：
from sklearn.metrics import pairwise_distances

X = [[0, 1], [1, 1]]
Y = [[1, 2], [2, 2]]
distances = pairwise_distances(X, Y, metric='euclidean')
print(distances)

# 曼哈顿距离（Manhattan Distance）：
distances = pairwise_distances(X, Y, metric='manhattan')
print(distances)

# 切比雪夫距离（Chebyshev Distance）：
distances = pairwise_distances(X, Y, metric='chebyshev')
print(distances)

# 余弦相似度（Cosine Similarity）：
distances = pairwise_distances(X, Y, metric='cosine')
print(distances)