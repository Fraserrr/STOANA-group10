import pandas as pd
import matplotlib.pyplot as plt

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

# 选择某一年度
year = 2003
year_data = data[data['年份'] == year]

# 选择三年的数据
years_to_extract = [2000, 2001, 2003]
multiyear_data = data[data['年份'].isin(years_to_extract)]

# 要绘制直方图的列
column_to_plot = '每股收益'
selected_data = year_data[column_to_plot]

# 年份堆叠直方图的列
multiyear_column_to_plot = '每股收益'
data_y0 = multiyear_data[multiyear_data['年份'] == years_to_extract[0]][multiyear_column_to_plot].dropna()
data_y1 = multiyear_data[multiyear_data['年份'] == years_to_extract[1]][multiyear_column_to_plot].dropna()
data_y2 = multiyear_data[multiyear_data['年份'] == years_to_extract[2]][multiyear_column_to_plot].dropna()


# 绘制直方图
def plot_hist():
    plt.figure(figsize=(10, 6))
    plt.hist(selected_data,
             bins=30,  # 区间数
             color='skyblue',  # 填充颜色
             edgecolor='black',  # 边框颜色
             alpha=0.8,  # 不透明度
             density=False,  # 显示概率密度(频率总和为1)而不是频次
             cumulative=False,  # 累积显示
             histtype='bar'
             # 'bar'（默认柱状图）、'step'（无填充的轮廓图）、'stepfilled'（填充的轮廓图）
             )
    # 设置标题和轴标签
    plt.title(f'{column_to_plot} - {year}年数据分布')
    plt.xlabel(column_to_plot)
    plt.ylabel('频数')
    # 显示图形
    plt.show()


# 绘制堆叠直方图
def plot_years_hist():
    plt.figure(figsize=(10, 6))
    plt.hist([data_y0, data_y1, data_y2],
             bins=30,
             stacked=True,
             color=['skyblue', 'salmon', 'lightgreen'],
             edgecolor='black',
             label=[f'{years_to_extract[0]}年', f'{years_to_extract[1]}年', f'{years_to_extract[2]}年']
             )
    # 添加标题和轴标签
    plt.title(f'{multiyear_column_to_plot} - 连续三年数据分布')
    plt.xlabel(multiyear_column_to_plot)
    plt.ylabel('频率')
    plt.legend()
    # 显示图形
    plt.show()


# 绘制一维直方图
plot_hist()
# 绘制堆叠直方图
plot_years_hist()
