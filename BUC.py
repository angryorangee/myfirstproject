import pandas as pd
import numpy as np
import load_data

def BUC(data,minsup):
    candidate = {}  # 空元组存放路径信息：路径信息仅由维度组成。
    data = load_data.df # 从取数模块获取数据
    # 遍历所有列仅保留维度列
    list = []
    for key, value in load_data.dict.items():
        if value is not "dim":
            list.append(key)
    df = data.drop(labels=list, axis=1)
    minsup = minsup * len(data)

    #遍历所有的列,并计数
    for i in df.columns:
        count = df.groupby([i]).count()  # 对当前维度计数
        space = count[count >= minsup].dropna()

        print(space,"\n--------------------",type(space))


BUC(load_data.df,0.03)
