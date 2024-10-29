import pandas as pd

# 读取 GBK 编码的 CSV 文件
df = pd.read_csv('sourcedata.csv', encoding='gbk')

# 将内容写入新文件，使用 UTF-8 编码
df.to_csv('rawdata.csv', encoding='utf-8', index=False)

print("GBK 文件编码已成功转换为 UTF-8！请查看新的 csv 文件是否生成")
