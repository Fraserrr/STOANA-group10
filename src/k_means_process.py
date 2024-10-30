import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 读取CSV文件
df = pd.read_csv('../data/rawdata.csv', encoding='utf-8')

# 选择财务指标
selected_columns = [
    '股票号', '年份',
    '【资产负债比    (％)】',  # 离散度高
    '【总资产      (万元)】',
    '【流动资产    (万元)】',
    '【长期投资    (万元)】',
    '【固定资产    (万元)】',
    '【无形其他资产(万元)】',
    '【股东权益比率  (％)】',
    '【股东权益增长率(％)】',
    '【股东权益    (万元)】',
    '【主营收入增长率(％)】',  # 以下为离散度较大的数据
    '【主营业务利润率(％)】',
    '【总资产增长率  (％)】'
]
# 提取并清理数据
data = df[selected_columns].copy()
data.columns = ['股票号', '年份',
                '资产负债比',
                '总资产', '流动资产', '长期投资', '固定资产', '无形其他资产',
                '股东权益比率', '股东权益增长率', '股东权益',
                '主营收入增长率', '主营业务利润率', '总资产增长率']
# 选择某一年度
year_data = data[data['年份'] == 2003]

# 选出需要用于聚类的数据维度
X_prepared = year_data[['资产负债比', '总资产增长率']]


# 计算欧氏距离
def distCal(vecA, vecB):
    return np.sqrt(np.sum(np.power(vecA - vecB, 2)))
    # vecA, vecB都为数组的形式，类似于[1 2]
    # power(x1, x2)数组的元素分别求n次方。x2可以是数字，也可以是数组，但是x1和x2的列数要相同。


# 为给定数据集构建一个随机质心矩阵
def randCent(dataSet, k):
    n = np.shape(dataSet)[1]  # shape函数的功能是读取矩阵的长度，比如shape[0]就是读取矩阵第一维度的长度。
    # shape[0]返回的是有多少行，shape[1]返回的是每一行有几个数据
    centroids = np.asmatrix(np.zeros((k, n)))  # 生成k*n的矩阵，用0初始化
    for j in range(n):  # 每一列的范围不一样
        minJ = min(dataSet[:, j])
        rangeJ = float(max(dataSet[:, j]) - minJ)
        centroids[:, j] = minJ + rangeJ * np.random.rand(k, 1)  # random.rand(k, 1)生成k行1列的随机数组
    return centroids


def kMeans(dataSet, k):
    m = np.shape(dataSet)[0]  # 一共有m行数据
    clusterAssment = np.asmatrix(np.zeros((m, 2)))  # 生成m*2的0矩阵，第一列存放该点所在簇的索引，第二列存放该点与簇质心的距离
    centroids = randCent(dataSet, k)  # 调用函数，构建质心矩阵
    clusterChanged = True  # 创建一个标志变量，若为真，则继续迭代
    count = 0
    while clusterChanged:
        count += 1
        clusterChanged = False
        # 寻找最近的质心
        for i in range(m):  # 对于每一行的数据
            minIndex = -1  # 在未迭代之前，将簇的索引设为-1
            minDist = np.inf  # 在未迭代之前，将最小距离设为无穷大
            for j in range(k):  # 对于当前每一个质心
                distJI = distCal(centroids[j, :], dataSet[i, :])  # 计算第i个数据与第j个质心之间的距离
                if distJI < minDist:
                    minIndex = j  # 若i与j之间距离小于目前的最小值，则更新当前点的索引为j
                    minDist = distJI  # 若i与j之间距离小于目前的最小值，则更新当前最小值
            if clusterAssment[i, 0] != minIndex:  # 如果第i行的第一个值（簇的索引）不为当前的minIndex，说明簇发生了改变
                clusterChanged = True  # 继续迭代
            clusterAssment[i, :] = int(minIndex), minDist ** 2
        # print(centroids)
        # 更新质心的位置
        for cent in range(k):
            category = np.nonzero(clusterAssment[:, 0].A == cent)  # 得到簇索引为cent的值的位置
            ptsInClust = dataSet[category[0]]
            centroids[cent, :] = np.mean(ptsInClust, axis=0)  # axis=0表示沿着矩阵列的方向进行均值计算
    print('*********************************************************************************')
    print('经过%d次迭代，最终的质心坐标为' % count)
    print(centroids)
    print('*********************************************************************************')
    print('打印所有点的索引及距离')
    print(clusterAssment)
    return centroids, clusterAssment


# 肘部法则确定K值
def elbow_method_show(X):
    distortions = []
    K = range(1, 13)  # 测试K值1~12

    for k in K:
        kmeanModel = KMeans(n_clusters=k, random_state=0).fit(X)
        distortions.append(kmeanModel.inertia_)

    # 绘制肘部图
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.show()


# 轮廓系数确定K值
def silhouette_method_show(X):
    silhouette_scores = []
    K = range(2, 13)  # 测试K值2~12，轮廓系数需要至少两个聚类

    for k in K:
        kmeanModel = KMeans(n_clusters=k, random_state=0).fit(X)
        labels = kmeanModel.labels_
        silhouette_scores.append(silhouette_score(X, labels))

    # 绘制轮廓系数图
    plt.plot(K, silhouette_scores, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Silhouette Score')
    plt.title('The Silhouette Method showing the optimal k')
    plt.show()


# 使用自定义K-Means函数进行聚类
def k_means_sample_show(X, k_value):
    # 读取列名用于自动设置标签
    x_label, y_label = X.columns[0], X.columns[1]

    # 将数据转为 NumPy 数组进行计算
    X_array = X.to_numpy()

    # 自定义 kMeans 函数调用
    centroids, clusterAssment = kMeans(X_array, k_value)

    # 可视化聚类结果
    plt.figure(figsize=(16, 12))
    # 提取每个簇的数据并绘制
    k = centroids.shape[0]
    for i in range(k):
        # 获取属于当前簇的所有数据点
        points_in_cluster = X_array[np.nonzero(clusterAssment[:, 0].A == i)[0]]
        plt.scatter(points_in_cluster[:, 0], points_in_cluster[:, 1], s=30, label=f'Cluster {i + 1}')

    # 绘制质心
    plt.scatter(centroids[:, 0].flatten().tolist(), centroids[:, 1].flatten().tolist(),
                c="red", s=400, alpha=0.6)

    plt.title('KMeans Clustering')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    # plt.legend()
    # plt.grid(True)
    plt.show()


# 使用确定的K值进行二维聚类（结果最稳定的函数）
def k_means_2d_show(X, k_value):
    # 读取列名用于自动设置标签
    x_label, y_label = X.columns[0], X.columns[1]

    # 将数据转为 NumPy 数组进行计算
    X_array = X.to_numpy()

    kmeans = KMeans(n_clusters=k_value, random_state=0)
    kmeans.fit(X_array)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    # 可视化聚类结果
    plt.scatter(X_array[:, 0], X_array[:, 1],
                c=labels,
                cmap='viridis',
                alpha=0.8
                )
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=300, alpha=0.5)
    plt.title('K-means Clustering')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


# 三维聚类
def k_means_3d_show(X, k_value):
    # 读取列名用于自动设置标签
    x_label, y_label, z_label = X.columns[0], X.columns[1], X.columns[2]

    # 将数据转为 NumPy 数组进行计算
    X_array = X.to_numpy()

    kmeans = KMeans(n_clusters=k_value, random_state=0)
    kmeans.fit(X_array)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    # 可视化聚类结果 (三维)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_array[:, 0], X_array[:, 1], X_array[:, 2], c=labels, cmap='viridis', alpha=0.6)
    ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], c='red', s=300, alpha=0.5)
    ax.set_title('K-means Clustering')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)
    plt.show()


# 使用肘部法则和轮廓系数测试
elbow_method_show(X_prepared)
silhouette_method_show(X_prepared)

# 最佳K值调用
k_means_2d_show(X_prepared, 7)
