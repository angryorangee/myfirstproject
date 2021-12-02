#从本地文件读取数据表并保存

import pandas as pd
import time
#计时开始
T1 = time.time()
df = pd.read_csv("Emission.csv")
#数据集标注列：期间、维度、指标
colnum = df.shape[1]
dict = {}
timedict = ["Year"]#时间列名需要先限定好

for (name,value) in df.iteritems():
    if name in timedict:
        dict[name] = "date"
    else:
        if isinstance(value[0],str):
            dict[name] = "dim"
        else:
            dict[name] = "measure"
#计时结束
T2 = time.time()

# print("数据读取耗时：%s 秒" %(T2-T1))



