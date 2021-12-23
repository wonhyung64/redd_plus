#%%
import pandas as pd
import numpy as np
import openpyxl as xl
import os
from functools import reduce

#%%
# os.chdir("E:\Data\\redd_plus\survey")
os.chdir("/Users/wonhyung64/data/redd")

print("Current Working Directory : ", os.getcwd())

df = pd.read_excel("국내기업의 해외온실가스 감축사업 참여방안 설문조사 raw data_1.xlsx", header = 2)
df_ = pd.read_excel("국내기업의 해외온실가스 감축사업 참여방안 설문조사 raw data_2.xlsx", header = 2)

df = df.iloc[:,1:]

#%%
def make_columns_binary(df):
    cols = df.columns
    for x in cols:
        df[x] = df[x].replace('.*', 1, regex=True)
        df = df.fillna(0)
    
    return df
# %%
df_sec = df.iloc[:,0:26]
df_sec = make_columns_binary(df_sec)
df = df.iloc[:,26:]
df_sub = df.iloc[:,:5]
df_sub.columns = ['응답자 담당년수','직위','감축사업진행','추진기간','준비기간']
df_sub.iloc[:,2:3] = make_columns_binary(df_sub.iloc[:,2:3])

df = df.iloc[:,5:]
df_3 = df.iloc[:,:4]
df_3.columns = ['no.3 c1','no.3 c2','no.3 c3','no.3 c기타']
df_3.iloc[:,0:3] = make_columns_binary(df_3.iloc[:,0:3])

df = df.iloc[:,4:]
df_3_1 = df.iloc[:,0:1]
df_3_1.columns = ['no.3-1']

df_3_1['no.3-1'] = df_3_1['no.3-1'].map({'자발적 시장 거래 (VCS, GS)':1, 'CSR, ESG':2, '탄소중립':3, '양자협력 사업 참여':4})

df = df.iloc[:,1:]
df_4 = df.iloc[:,0:2]
df_4.columns = ['no.4','no.4 c기타']
df_4['no.4'] = df_4['no.4'].map({'CSR, ESG 부서':1, '배출권거래제 대응부서':2, '재무구매부서':3})

df = df.iloc[:,2:]
df_5 = df.iloc[:,0:1]
df_5.columns = ['no.5']
df_5['no.5'] = df_5['no.5'].map({'예':1, '아니요':2, '모르겠다':3})

df = df.iloc[:,1:]
df_6 = df.iloc[:,0:1]
df_6.columns = ['no.6']

df = df.iloc[:,1:]
df_7 = df.iloc[:,0:2]
df_7.columns = ['no.7','no.7 c기타']
df_7['no.7'] = df_7['no.7'].map({'추후 SDM* 전환 용이성을 위해':1, '사업 전력 국가라서':2})

df = df.iloc[:,2:]
df_8 = df.iloc[:,:6]
df_8.columns = ['no.8 c1','no.8 c2','no.8 c3','no.8 c4','no.8 c5','no.8 c기타']
df_8.iloc[:,0:5] = make_columns_binary(df_8.iloc[:,0:5])

df = df.iloc[:,6:]
df_9 = df.iloc[:,:6]

cols_tmp = df_9.columns
nw_cols = []
for x in cols_tmp:
    df_9[x] = df_9[x].map({'매우고려한다⑤':5, '고려한다④':4, '보통이다③':3, '고려하지않는다②':2, '전혀 고려하지 않는다':1})
    nw_cols.append('no.9_' + x)
df_9.columns = nw_cols

df = df.iloc[:,7:]
df_9_1 = df.iloc[:,:17]
df_9_1.columns = ['no.9-1 c1','no.9-1 c2','no.9-1 c3','no.9-1 c4','no.9-1 c5','no.9-1 c6','no.9-1 c7','no.9-1 c8',
                  'no.9-1 c9','no.9-1 c10','no.9-1 c11','no.9-1 c12','no.9-1 c13','no.9-1 c14','no.9-1 c15',
                  'no.9-1 c16','no.9-1 c17']
df_9_1 = make_columns_binary(df_9_1)

df = df.iloc[:,17:]
df_10 = df.iloc[:,:2]
df_10.columns = ['no.10','no.10 c기타']
df_10['no.10'] = df_10['no.10'].map({'직접투자':1,'간접투자(선도거래)':2,'펀드조성':3})

df_lst = [df_sec, df_sub ,df_3, df_3_1, df_4, df_5, df_6, df_7, df_8, df_9, df_9_1, df_10]
data1 =  reduce(lambda left, right: pd.concat([left,right],axis=1), df_lst)

#%%
df = df.iloc[:,2:]
df_1 = df.iloc[:,:7]
cols_tmp = df_1.columns
nw_cols = []
for x in cols_tmp:
    df_1[x] = df_1[x].map({'매우 그렇다':5, '그렇다':4, '보통이다':3, '그렇지 않다':2, '전혀 아니다':1})
    nw_cols.append('no.1_' + x)
df_1.columns = nw_cols

df = df.iloc[:,7:]
df_1_ = df.iloc[:,:4]
df_1_.columns = ['no.1-1','no.1-2','no.1-3','no.1-4']

df = df.iloc[:,4:]
df_2 = df.iloc[:,:5]
cols_tmp = df_2.columns

for x in cols_tmp:
    df_2[x] = df_2[x].map({'매우중요하다 ':5, '중요하다 ':4, '보통이다':3, '중요하지않다':2, '전혀중요하지않다':1})
df_2.columns = ['국제 개발 원조 효과', '국가감축목표(NDC) 달성 효과', '전지구적 감축노력', '비용효과성', '국내 감축기술 혁신']

df = df.iloc[:,5:]
df_3 = df.iloc[:,:4]

cols_tmp = df_3.columns
nw_cols = []
for x in cols_tmp:
    df_3[x] = df_3[x].map({'매우고려한다':5, '고려한다':4, '보통이다':3, '고려하지않는다':2, '아예 고려하지 않는다':1})
    nw_cols.append('no.3_' + x)
df_3.columns = nw_cols


df = df.iloc[:,4:]
df_4 = df.iloc[:,0:1]

df_4.columns = ['no.4']
df_4['no.4'] = df_4['no.4'].map({'예':1,'아니요':2,'모르겠다':3})

df = df.iloc[:,1:]
df_4_1 = df.iloc[:,:3]

cols_tmp = df_4_1.columns
nw_cols = []
for x in cols_tmp:
    df_4_1[x] = df_4_1[x].map({'매우필요하다':5, '필요하다':4, '보통이다':3, '필요하지않다':2, '전혀 필요하지 않다':1})
    nw_cols.append('no.4-1_' + x)
df_4_1.columns = nw_cols

df = df.iloc[:,3:]
df_4_2 = df.iloc[:,0:1]
df_4_2.columns = ['no.4-2']

df_lst = [df_1, df_1_, df_2, df_3, df_4, df_4_1, df_4_2]
data2 =  reduce(lambda left, right: pd.concat([left,right],axis=1), df_lst)

#%%
df = df.iloc[:,1:]
df_1 = df.iloc[:,:7]
cols_tmp = df_1.columns
nw_cols = []
for x in cols_tmp:
    df_1[x] = df_1[x].map({'잘 알고 있다':5, '알고 있다':4, '보통이다':3, '들어 보았다 ':2, '전혀 모른다 ':1})
    nw_cols.append('no.1_' + x)
df_1.columns = nw_cols

df = df.iloc[:,7:]
df_2 = df.iloc[:,:2]

df_2.columns = ['no.2', 'no.2 c기타']
df_2['no.2'] = df_2['no.2'].map({'산림 사업':1, '신재생 에너지 사업':2, '고효율 에너지 제품 보급 사업':3,
                                 '탄소 저장/제거':4})

df = df.iloc[:,2:]
df_3 = df.iloc[:,:6]

cols_tmp = df_3.columns
nw_cols = []
for x in cols_tmp:
    df_3[x] = df_3[x].map({'매우그렇다':5, '그렇다':4, '보통이다':3, '아니다':2, '전혀아니다':1})
    nw_cols.append('no.3_' + x)
df_3.columns = nw_cols

df = df.iloc[:,6:]
df_46 = df.iloc[:,:3]
df_46.columns = ['no.4','no.5','no.6']

cols_tmp = ['no.4','no.5']
for x in cols_tmp:
    df_46[x] = df_46[x].map({'예':1, '아니오':2})

    
df_lst = [df_1, df_2, df_3, df_46]
data3 = reduce(lambda left, right: pd.concat([left,right],axis=1), df_lst)

#%%
data1.to_excel("survey_case1_no_1.xlsx", index=False)
data2.to_excel("survey_case1_no_2.xlsx", index=False)
data3.to_excel("survey_case1_no_3.xlsx", index=False)
# %%
