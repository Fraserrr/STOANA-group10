import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取CSV文件
df = pd.read_csv('../data/rawdata.csv', encoding='utf-8')

# 选择财务指标
selected_columns = [
    '股票号', '年份',
    '【每股收益      (元)】',
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
                '每股收益',
                '资产负债比',
                '总资产', '流动资产', '长期投资', '固定资产', '无形其他资产',
                '股东权益比率', '股东权益增长率', '股东权益',
                '主营收入增长率', '主营业务利润率', '总资产增长率']

# 选择要绘图的数据列
plot_columns = ['资产负债比', '主营收入增长率', '主营业务利润率', '总资产增长率']


# 截面数据分析：选择某一年进行比较
def plot_box_cross_section(year):
    year_data = data[data['年份'] == year]
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=year_data[plot_columns])
    plt.title(f'{year} 年截面数据分析')
    plt.xticks(rotation=45)
    plt.legend()
    plt.show()


# 指定年份绘图
plot_box_cross_section(2000)
