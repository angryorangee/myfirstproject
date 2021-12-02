# 数据
import pandas as pd
import numpy as np
import scipy.stats
import sklearn as skl


"""
数据结构:
维度|指标|预测(基期)|实际（当期）
dimension|indicator|pred|actual
"""

# 构建测试数据
# 一个公司在上海、北京 通过四个渠道（苏宁、京东、淘宝、拼多多），销售三个产品（电脑、平板、手机）、以下是3季度的销售收入预实对比。
# 发现销售收入预测值为：100 但真实值只有50，财务部门需要做根因分析（找到造成未完成预算的罪魁祸首）。
# 财务部门需要：
# 1、找到最关键的维度
# 2、找到关键的成员
# 3、找到关键的组合例如（北京-平板）adtributor不可实现该功能,但可通过递归调用归因方法来实现该效果。
# 4、各个关键因素可量化排序
lists = [['region', '北京', 94, 47]
    , ['region', '上海', 6, 3],
         ['region01', '全国', 100, 50]
    , ['tunnel', '苏宁', 50, 24]
    , ['tunnel', '京东', 20, 21]
    , ['tunnel', '淘宝', 20, 4]
    , ['tunnel', '拼多多', 10, 1]
    , ['tunnel01', '苏东', 70, 50]
    , ['tunnel01', '拼淘', 30, 5]
    , ['device type', '电脑', 50, 49]
    , ['device type', '手机', 25, 1]
    , ['device type', '平板', 25, 0]
    , ['device type01', '桌面', 50, 49]
    , ['device type01', '移动', 50, 1]]
df = pd.DataFrame(lists, columns=['dimension', 'indicator', 'pred', 'actual'])

lists_2 = [['业务板块', '家用电器', 4850, 4500]
    , ['业务板块', '电动汽车', 3500, 2000]
    , ['业务板块', '金融业务', 1650, 1500]
    , ['区域', '中国', 5770, 4080]
    , ['区域', '北美', 2040, 2010]
    , ['区域', '欧洲', 990, 980]
    , ['区域', '拉美', 850, 670]
    , ['区域', '中东', 350, 260]]
df2 = pd.DataFrame(lists_2, columns=['dimension', 'indicator', 'pred', 'actual'])

"""
root cause analysis
"""


# JS散度
def JS_divergence(p, q):
    p = np.array(p)
    q = np.array(q)
    M = (p + q) / 2

    # 方法一：自定义函数
    js1 = 0.5 * np.sum(p * np.log(p / M)) + 0.5 * np.sum(q * np.log(q / M))
    # 方法二：调用scipy包
    js2 = 0.5 * scipy.stats.entropy(p, M) + 0.5 * scipy.stats.entropy(q, M)
    return round(float(js1), 4)


# 根因分析
def root_cause_analysis(df, num):
    # 第一步：计算某维度下某因素的真实值和预测值占总收入的差异性----Surprise
    group = df.groupby('dimension').sum()
    group = group.reset_index()
    group = group.rename(columns={'pred': 'pred_sum', 'actual': 'actual_sum'})
    dta = pd.merge(df, group, on='dimension', how='left')
    # print(dta)

    dta['actual_sum'] = dta['actual_sum'].apply(lambda x: round(x))
    dta['pred_sum'] = dta['pred_sum'].apply(lambda x: round(x))

    dta['p'] = dta['pred'] / dta['pred_sum']
    dta['q'] = dta['actual'] / dta['actual_sum']

    # 第一步：计算JS散度——surprise 惊奇度
    JS_list = []
    for i in dta['dimension'].tolist():
        ls1 = dta[dta['dimension'] == i]['p']
        ls2 = dta[dta['dimension'] == i]['q']
        JS = JS_divergence(ls1, ls2)
        JS_list.append(JS)
    dta['JS'] = JS_list

    # 第二步：计算某维度下某因素波动占总体波动的比率----Explanatory power
    dta['EP'] = (dta['actual'] - dta['pred']) / (dta['actual_sum'] - dta['pred_sum'])
    result = dta
    # 第三步：排序出结果
    result = result.sort_values(by=['JS', 'EP'], ascending=False).reset_index()
    root_cause = result.groupby('dimension').mean().sort_values(by=['JS'], ascending=False).reset_index().head(1)
    return result.head(num)


# 展示结论
root_cause_analysis(df, 14)
print(root_cause_analysis(df,14))