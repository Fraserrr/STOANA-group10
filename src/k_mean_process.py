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


# 使用确定的K值进行二维聚类
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
    ax.set_xlabel('x_label')
    ax.set_ylabel('y_label')
    ax.set_zlabel('z_label')
    plt.show()


# 使用肘部法则和轮廓系数测试
elbow_method_show(X_prepared)
silhouette_method_show(X_prepared)

# 最佳K值调用
k_means_2d_show(X_prepared, 7)
