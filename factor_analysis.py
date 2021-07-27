#%%
from factor_analyzer import FactorAnalyzer
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import openpyxl as xl
import os
import tensorflow_probability as tfp
# %%
os.chdir("E:\Data\\redd_plus\survey")
print("Current Directory : ", os.getcwd())
# %%
data2 = pd.read_excel("survey_case1_no_2.xlsx")
data3 = pd.read_excel("survey_case1_no_3.xlsx")
# %%
df_obsta = data2.iloc[:,:6]
df_value = data2.iloc[:,11:16]
df_knowl = data2.iloc[:,16:20]
df_recog = data3.iloc[:, 0:7]
# %%
X = pd.concat([df_knowl,df_recog], axis=1)
X = X.dropna(axis=0)
X.corr()
tfp.stats.correlation(X)

#%%
df = X

fa = FactorAnalyzer(n_factors=2, rotation="varimax")
fa.fit(df)

loadings = fa.loadings_

ev, v = fa.get_eigenvalues()

xvals = range(1,df.shape[1]+1)
plt.scatter(xvals, ev)
plt.plot(xvals, ev)
plt.title('Scree Plot')
plt.xlabel('Factor')
plt.ylabel('Eigenvalue')
plt.grid()
plt.show()

pd.DataFrame.from_records(loadings)
# %%
