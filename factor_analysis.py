#%%

from factor_analyzer import FactorAnalyzer
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import openpyxl as xl
import os
from sklearn import preprocessing
import statsmodels.api as sm
# %%

os.chdir("E:\Data\\redd_plus\survey")
print("Current Directory : ", os.getcwd())
# %%
data2 = pd.read_excel("survey_case1_no_2.xlsx")
data3 = pd.read_excel("survey_case1_no_3.xlsx")
# %%
# preprocessing

df_obsta = data2.iloc[:,:6]
df_value = data2.iloc[:,11:16]
df_knowl = data2.iloc[:,16:20]
df_recog = data3.iloc[:, 9:15]

df = pd.concat([df_obsta, df_value, df_knowl, df_recog],axis=1)

df = df.dropna(axis=0)
X1 = df.iloc[:,:6]
X2 = df.iloc[:,6:11]
X3 = df.iloc[:,11:15]
Y = df.iloc[:,15:]


scaler = preprocessing.StandardScaler().fit(Y)
Y_scaled = scaler.transform(Y)
Y_cols = ["Y1","Y2","Y3","Y4","Y5","Y6"]
Y_scaled = pd.DataFrame(Y_scaled, index=None, columns=Y_cols)
print(Y_scaled)
#%%

# factor analysis for all
X = pd.concat([X1,X2,X3],axis=1)
scaler = preprocessing.StandardScaler().fit(X)
X_scaled = scaler.transform(X)

#%%
import sklearn.decomposition.FactorAnalysis
fac = sklearn.decomposition._factor_analysis(n_components=2)
#%%

fa = FactorAnalyzer(n_factors=2, rotation="varimax")
fa.fit(X_scaled)

loadings = fa.loadings_

ev, v = fa.get_eigenvalues()
# print(ev / sum(ev))

xvals = range(1,X.shape[1]+1)
plt.scatter(xvals, ev)
plt.plot(xvals, ev)
plt.title('Scree Plot')
plt.xlabel('Factor') 
plt.ylabel('Eigenvalue')
plt.grid()
plt.show()

print('\n[Factor Scores]')
print(pd.DataFrame.from_records(loadings))
print("\n표본 갯수 :", len(X_scaled))

factor1 = pd.DataFrame.from_records(loadings)[0]
factor2 = pd.DataFrame.from_records(loadings)[1]

factor_X1 = np.matmul(X_scaled, factor1)
factor_X2 = np.matmul(X_scaled, factor2)
factor_X = pd.DataFrame({
    'factor_X1' : factor_X1,
    'factor_X2' : factor_X2
})
# print(factor_X)

print("\n[Factor Correlation]")
corr = pd.DataFrame.from_records(loadings).corr(method='pearson')
corr.style.background_gradient(cmap='coolwarm')

#%%
factor_X['intercept'] = 1
lm = sm.OLS(Y_scaled['Y2'], factor_X[['intercept','factor_X1','factor_X2']])
results = lm.fit()
results.summary()
#%%
# factor analysis for X1
fa = FactorAnalyzer(n_factors=1, rotation="varimax")
fa.fit(X1)

loadings = fa.loadings_

ev, v = fa.get_eigenvalues()
print(ev / sum(ev))

xvals = range(1,X1.shape[1]+1)
plt.scatter(xvals, ev)
plt.plot(xvals, ev)
plt.title('Scree Plot')
plt.xlabel('Factor')
plt.ylabel('Eigenvalue')
plt.grid()
plt.show()
print(pd.DataFrame.from_records(loadings))
print("표본 갯수 :", len(X))

#%%
# factor analysis for X2
fa = FactorAnalyzer(n_factors=1, rotation="varimax")
fa.fit(X2)

loadings = fa.loadings_

ev, v = fa.get_eigenvalues()
print(ev / sum(ev))

xvals = range(1,X2.shape[1]+1)
plt.scatter(xvals, ev)
plt.plot(xvals, ev)
plt.title('Scree Plot')
plt.xlabel('Factor')
plt.ylabel('Eigenvalue')
plt.grid()
plt.show()
print(pd.DataFrame.from_records(loadings))
print("표본 갯수 :", len(X))

#%%
# factor analysis for X3
fa = FactorAnalyzer(n_factors=1, rotation="varimax")
fa.fit(X3)
fa.get_eigenvalues()
loadings = fa.loadings_

ev, v = fa.get_eigenvalues()
print(ev / sum(ev))
xvals = range(1,X3.shape[1]+1)
plt.scatter(xvals, ev)
plt.plot(xvals, ev)
plt.title('Scree Plot')
plt.xlabel('Factor')
plt.ylabel('Eigenvalue')
plt.grid()
plt.show()
print(pd.DataFrame.from_records(loadings))
print("표본 갯수 :", len(X))

# %%

# %%

# %%

# %%
