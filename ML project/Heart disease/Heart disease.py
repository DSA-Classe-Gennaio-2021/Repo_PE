#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv('heart.csv')
df.head()


# In[3]:


df.info()


# In[4]:


df.describe().T


# In[6]:


sns.pairplot(df[['age','trestbps','chol','thalach','oldpeak']], kind='reg');


# In[7]:


mask =np.triu(np.ones_like(df[['age','trestbps','chol','thalach','oldpeak']].corr().abs(), dtype=bool))
sns.heatmap(df[['age','trestbps','chol','thalach','oldpeak']].corr(), cmap = 'PuOr',mask=mask, annot=True);
corr_matrix = df[['age','trestbps','chol','thalach','oldpeak']].corr()
corr_matrix.mask(mask)


# In[8]:


for i in df.drop('target',axis=1).columns:
    sns.kdeplot(data=df,x=i,hue='target',shade=True)
    plt.show()


# In[9]:


df.target.value_counts()


# Importing useful libraries. Note I used some custom functionality imported from func.py

# In[15]:


from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectFromModel, SelectKBest, chi2, RFE
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score

from func import PearsonSelector, MinMaxCust, CustomEstimator                #from func.py


# In[11]:


X = df.iloc[:,:-1]
y = df.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 15)


# Let's visualize whether the dataset could be described by few principal components

# In[12]:


pca = PCA()

pca.fit(X_train)

fig, ax = plt.subplots(1,2)
fig.set_size_inches(10,5)
ax[0].plot(pca.explained_variance_ratio_.cumsum(), 'pr--')
ax[0].set(title='Explained Variance Ratio cum', xticks = [i for i in range(13)], yticks = [i for i in np.arange(0.75,1,0.02)],xlabel = 'PCA components')
ax[1].plot(pca.explained_variance_ratio_, 'pb--')
ax[1].set(title='Explained Variance Ratio', xticks = [i for i in range(13)], xlabel = 'PCA components');


# Looks like 3 components should fit

# Defining a param grid with both estimators and associated parameters to feed the pipelines with

# In[13]:


params = [{'model': [LogisticRegression()],
                    'model__C': [1,5,10,20,100],
                    'model__max_iter': [300,500]},
          
          {'model': [KNeighborsClassifier()],
                    'model__n_neighbors': [i for i in range (1,11)]},
          
          {'model': [DecisionTreeClassifier()],
                    'model__criterion': ['gini', 'entropy'],
                    'model__min_samples_leaf': [i for i in range(1,6)],
                    'model__max_depth': [i for i in range(1,6)]},
          
          {'model': [RandomForestClassifier()],
                    'model__n_estimators': [200,300],
                    'model__max_depth': [i for i in range(1,6)]},
          
          {'model': [xgb.XGBClassifier()],
                    'model__n_estimators': [100,1000]},   
          ]


# Defining the pipelines I am willing to use. Note that CustomEstimator simply acts as a placeholder for the estimators in the grid above. The goal is performing hyperparameter tuning and selecting the best pipeline atoghether.

# In[ ]:


pipe_dict = {}
pipe_dict['PCA'] = Pipeline([('scaler', StandardScaler()), ('pca', PCA(n_components=3)), ('model', CustomEstimator())])
pipe_dict['RFE'] = Pipeline([('scaler', MinMaxScaler()),('rfe',RFE(LogisticRegression(max_iter=500))), ('model', CustomEstimator())])
pipe_dict['SelectFromModel'] = Pipeline([('selector', SelectFromModel(LogisticRegression(max_iter = 1000))), ('model', CustomEstimator())])
pipe_dict['SelectKBest'] = Pipeline([('selector', SelectKBest(chi2, k = 3)), ('model', CustomEstimator())])

results = {} 
for name, pipe in pipe_dict.items():
    gs = GridSearchCV(pipe, param_grid = params, scoring = 'roc_auc', cv=10, verbose=1)
    gs.fit(X_train, y_train)
    probs = gs.best_estimator_.predict_proba(X_test)[:,1]
    preds = gs.best_estimator_.predict(X_test)
    results[(name, gs.best_estimator_['model'])] = roc_auc_score(y_test, probs), f1_score(y_test, preds), recall_score(y_test, preds), precision_score(y_test, preds)


# In[18]:


results


# In[19]:


names = ['PCA - LogR', 'RFE - RF', 'SelectFromModel - RF', 'SelectKBest - RF']
scores = []
for k,v in results.items():
    scores.append(v)
pd.DataFrame(scores, index = names, columns = ['AUC', 'F1', 'Recall','Precision'])


# Looks like LogisticRegression associated to PCA performed the best according to almost all evaluated metrics
