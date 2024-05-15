import pandas as pd

# 读取 CSV 文件
anom = pd.read_csv('./dataset/test.csv')

# 打印 CSV 文件的列名
print(anom.columns)

