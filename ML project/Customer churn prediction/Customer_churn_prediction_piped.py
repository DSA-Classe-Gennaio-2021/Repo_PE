#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv('train.csv')


# In[3]:


df.head()


# Looks like 4 columns have high variance and would possibly benefit some kind of normalization

# In[4]:


sns.heatmap(df.corr(),cmap='BuGn');


# Looks like there is multicollinearity among certain numerical features, hence some need be dropped

# In[5]:


df[df.select_dtypes(['int64','float64']).columns].boxplot()
plt.xticks(rotation=90);


# The boxplot overview suggests that, since some outliers are in, the options are either proceed with the removal or adopt a robust scaling

# In[6]:


for i in df.select_dtypes(['int64','float64']).columns:
    sns.histplot(df[i]);
    plt.show();


# The majority of numerical features shows a gaussian distribution

# In[7]:


df.select_dtypes('object').head()


# We have some categorical columns: we need to encode them

# In[8]:


df.churn.value_counts(normalize=True)


# The df is unbalanced and needs oversampling/undersampling

# I will proceed with building some pipelines that include preprocessing, feature selection and model training in order to evaluate which performs best. To do so, I will use, along with standard classes, some custom classes to be imported from the file func.py

# Importing libraries

# In[9]:


from sklearn.base import BaseEstimator, TransformerMixin
from func import PearsonSelector, MinMaxScalerCust, StandardScalerCust, BinaryEncoder, GetDummies, DropColumns, get_best, plot_roc_prc, get_results
from imblearn.pipeline import Pipeline, make_pipeline
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.feature_selection import SelectFromModel, SelectKBest, chi2, f_classif, RFE
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, roc_curve, precision_recall_curve


# Let's split the df into train and test

# In[10]:


X = df.drop('churn', axis=1)
y = df.churn.map({'yes':1,'no':0})                                                    #not encodable in the pipeline
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=8)


# Defining pipelines along with relative parameters to be used in GridSearchCV

# #### Pipe 1 ---> StandardScaler + PCA

# In[11]:


pipe_1 = make_pipeline(PearsonSelector(),                       #OutliersIQR(),\
                       StandardScalerCust(),\
                       BinaryEncoder(selected_columns = ['international_plan','voice_mail_plan']),\
                       DropColumns(['state','area_code']),\
                       #GetDummies(),
                       SMOTE(),\
                       PCA(),\
                       xgb.XGBClassifier(n_jobs=-1))

params_1 = [{'pearsonselector__limit': [0.2,0.4],
          'smote__k_neighbors': [3,5],
          'pca__n_components': [2,3],
          'xgbclassifier__n_estimators': [1000]}]


# #### Pipe 2 ---> MinMaxScaler + RFE

# In[12]:


pipe_2 = make_pipeline(PearsonSelector(),                       #OutliersIQR(),\
                       MinMaxScalerCust(),\
                       BinaryEncoder(selected_columns = ['international_plan','voice_mail_plan']),\
                       DropColumns(['state','area_code']),\
                       #GetDummies(),
                       SMOTE(),\
                       RFE(LogisticRegression(max_iter=500)),\
                       xgb.XGBClassifier(n_jobs=-1))

params_2 = [{'pearsonselector__limit': [0.2,0.4],
          'smote__k_neighbors': [5,7],
          'rfe__n_features_to_select': [0.3,0.5] ,
          'xgbclassifier__n_estimators': [1000]}]


# #### Pipe 3 ---> MinMaxScaler + SelectKBest

# In[13]:


pipe_3 = make_pipeline(PearsonSelector(),                       #OutliersIQR(),\
                       MinMaxScalerCust(),\
                       BinaryEncoder(selected_columns = ['international_plan','voice_mail_plan']),\
                       DropColumns(['state','area_code']),\
                       #GetDummies(),
                       SMOTE(),\
                       SelectKBest(),\
                       xgb.XGBClassifier(n_jobs=-1))

params_3 = [{'pearsonselector__limit': [0.2,0.4],
          'smote__k_neighbors': [5],
          'selectkbest__score_func': [chi2, f_classif],\
          'selectkbest__k': [2,5,10],\
          'xgbclassifier__n_estimators': [1000]}]


# Defining a dictionary that keeps pipelines and parameters defined above

# In[14]:


pipe_dict = {'pipe_1': [pipe_1, params_1],
             'pipe_2': [pipe_2, params_2],
             'pipe_3': [pipe_3, params_3]}


# Fitting the pipes and collecting best estimators for each pipe in a dictionary

# In[ ]:


best_estimators = get_best(pipe_dict, X_train, X_test, y_train)


# Let's print out the results

# In[18]:


get_results(best_estimators, X_test, y_test).T


# Looks like pipe_3 (i.e. MinMaxScaler + SelectKBest) performed the best according to all the considered metrics

# For a visual clue, let's plot roc and prc

# In[22]:


plot_roc_prc(best_estimators, X_test, y_test)


# ### Conclusion

# Pipe_3 assures a better precision/recall tradeoff, which is a good indicator for a model which has to predict customers that churn
