import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# 读取CSV文件
df = pd.read_csv('../data/rawdata.csv', encoding='utf-8')

# 选择财务指标
selected_columns = [
    '股票号', '年份',
    '【每股收益      (元)】',
    '【每股净资产    (元)】',
    '【每股收益增长率(％)】',
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
                '每股收益', '每股净资产', '每股收益增长率', '净利润',
                '资产负债比',
                '总资产', '流动资产', '长期投资', '固定资产', '无形其他资产',
                '股东权益比率', '股东权益增长率', '股东权益',
                '主营收入增长率', '主营业务利润率', '总资产增长率'
                ]

# 需要做时间序列分析的字段
selected_column_name = '总资产增长率'


# 绘制所有股票的某一列数据的时间序列。
def plot_time_series_selected(column_name):
    unique_stock_ids = data['股票号'].unique()  # 获取所有唯一的股票号
    unique_years = sorted(data['年份'].unique())  # 获取所有唯一的年份，并排序

    # 选择 4 个离散横坐标点
    selected_years = unique_years[::max(1, len(unique_years) // 3)][:4]

    plt.figure(figsize=(24, 12))  # 设置图形大小

    # 遍历所有的股票号
    for stock_id in unique_stock_ids:
        stock_data = data[data['股票号'] == stock_id]  # 获取对应股票的数据
        plt.plot(stock_data['年份'], stock_data[column_name], label=f'股票号 {stock_id}')

    # 设置图的标题和坐标轴标签
    plt.title(f'所有股票的 {column_name} 时间序列分析')
    plt.xlabel('年份')
    plt.ylabel(column_name)
    plt.xticks(selected_years)  # 设置横轴为选中的 4 个离散点

    # plt.legend()  # 显示图例（图例过大可能影响视觉）
    plt.show()  # 显示图形


# 调用函数
plot_time_series_selected(selected_column_name)
