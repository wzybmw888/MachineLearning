from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.metrics import accuracy_score

# 加载数据集
iris = load_iris()
X = iris.data
y = iris.target

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# 定义限定半径最近邻算法模型
rnn = RadiusNeighborsClassifier(radius=0.5,metric='minkowski',p=2)

# 训练模型
rnn.fit(X_train, y_train)

# 进行预测
y_pred = rnn.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)