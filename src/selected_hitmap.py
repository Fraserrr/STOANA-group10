import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 读取CSV文件
df = pd.read_csv('../data/rawdata.csv', encoding='utf-8')

# 选择财务指标
selected_columns = [
    '股票号', '年份',
    '【每股收益      (元)】',
    '【净资产收益率  (％)】',
    '【净利润      (万元)】',
    '【资产负债比    (％)】',  # 离散度高
    '【总资产      (万元)】',
    '【流动资产    (万元)】',
    '【长期投资    (万元)】',
    '【固定资产    (万元)】',
    '【无形其他资产(万元)】',
    '【股东权益比率  (％)】',
    '【股东权益增长率(％)】',
    '【股东权益    (万元)】',
    '【主营收入增长率(％)】',  # 离散度较大的数据
    '【主营业务利润率(％)】',
    '【总资产增长率  (％)】'
]

# 提取并清理数据
data = df[selected_columns].copy()
data.columns = ['股票号', '年份',
                '每股收益', '净资产收益率', '净利润',
                '资产负债比',
                '总资产', '流动资产', '长期投资', '固定资产', '无形其他资产',
                '股东权益比率', '股东权益增长率', '股东权益',
                '主营收入增长率', '主营业务利润率', '总资产增长率']

# 选择某一年度
year = 2000
year_data = data[data['年份'] == year]

# 需要计算相关性的字段
label_columns = [
     '每股收益', '净资产收益率', '净利润',
     '资产负债比',
     '总资产', '流动资产', '长期投资', '固定资产', '无形其他资产',
     '股东权益比率', '股东权益增长率', '股东权益',
     '主营收入增长率', '主营业务利润率', '总资产增长率'
]

correlation_selected_data = year_data[label_columns].copy()

# 数据清洗
for column in correlation_selected_data.columns:
    q1 = correlation_selected_data[column].quantile(0.20)
    q3 = correlation_selected_data[column].quantile(0.80)
    mask = (correlation_selected_data[column] - q1 > 0.38) | (q3 - correlation_selected_data[column] > 0.38)

    if mask.any():
        mean_val = correlation_selected_data[~mask][column].mean()
        correlation_selected_data.loc[mask, column] = mean_val


# 设置字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def plot_correlation_heatmap(data):
    # 计算相关性矩阵
    corr_matrix = data.corr()

    # 设置图形大小
    plt.figure(figsize=(12, 9))

    # 绘制热力图
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        cbar=True,
        xticklabels=label_columns,  # 显示横轴标签
        yticklabels=label_columns,  # 显示纵轴标签
        square=True  # 正方形单元格
    )

    # 设置标题
    plt.title("关键财务指标相关性热力图")
    plt.xticks(rotation=45)  # 调整横轴标签旋转角度，避免重叠
    plt.yticks(rotation=0)  # 保持纵轴标签水平显示
    plt.tight_layout()  # 调整图像布局，避免标签重叠
    plt.show()


plot_correlation_heatmap(correlation_selected_data)
