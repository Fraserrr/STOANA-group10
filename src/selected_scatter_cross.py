import pandas as pd
import matplotlib.pyplot as plt

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

'''
参数配置项说明
alpha: 不透明度
cmap: 颜色映射方案
s: 气泡大小字段
c: 气泡颜色字段
'''


def plot_2_dim_scatter(year):
    # 绘制二维散点图
    year_data = data[data['年份'] == year]
    plt.figure(figsize=(24, 12))
    plt.scatter(year_data['资产负债比'], year_data['总资产增长率'],
                alpha=0.8
                )
    plt.xlabel('资产负债比')
    plt.ylabel('总资产增长率')
    plt.title(f'{year}年资产负债比-总资产增长率分析')
    plt.show()


def plot_3_dim_scatter(year):
    year_data = data[data['年份'] == year]
    plt.figure(figsize=(24, 12))
    plt.scatter(
        year_data['每股净资产'], year_data['每股收益'],
        s=year_data['净利润'] * 300,  # 气泡大小
        alpha=0.6
    )
    plt.xlabel('每股净资产')
    plt.ylabel('每股收益')
    plt.title(f'{year}年每股净资产-每股收益-净利润分析')
    plt.show()


def plot_4_dim_scatter(year):
    year_data = data[data['年份'] == year]
    plt.figure(figsize=(48, 48))
    bubble = plt.scatter(
        year_data['每股净资产'], year_data['每股收益'],
        s=year_data['净利润'] * 100,  # 气泡大小
        c=year_data['每股收益增长率'],  # 气泡颜色
        cmap='viridis',  # 颜色映射方案
        alpha=0.8
    )
    plt.colorbar(bubble, label='每股收益增长率')  # 颜色图例
    plt.xlabel('每股净资产')
    plt.ylabel('每股收益')
    plt.title(f'{year}年每股净资产-每股收益-净利润-每股收益增长率分析')
    plt.show()


# 指定年份，调用函数绘图
plot_2_dim_scatter(2000)
plot_3_dim_scatter(2000)
plot_4_dim_scatter(2000)  # 运行可能稍慢，请耐心等待程序绘图
