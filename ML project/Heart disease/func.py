#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np 
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline 


# In[12]:


from sklearn.base import BaseEstimator, TransformerMixin
class PearsonSelector(BaseEstimator, TransformerMixin): # TransformerMixin ensures we get fit_transform. BaseEstimator ensures we get                                                                 get_params and set_params
    
    def __init__(self, limit=0.4):                                                #sets a default limit for selecting features
        self.limit = limit
        
    def fit(self, X, y = None):
        return self
        
    def transform(self, X):
        corr_matrix = X.corr().abs()                                              #gets a positive correlation matrix
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))                     #gets half of the correlation matrix
        corr_df = corr_matrix.mask(mask)
        cols_to_drop = [col for col in corr_df.columns if any(corr_df[col]> self.limit)]  #select cols with an abs pc greater than limit
        
        self.cols_to_drop = cols_to_drop
        
        return (pd.DataFrame(X)).drop(self.cols_to_drop, axis=1)                   ##Note: Could not make it work!!!

class MinMaxCust(BaseEstimator, TransformerMixin):   #MinMaxScaler returns an array so I need to customize it to have a dataframe instead
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        scaled_df = (X - X.min())/(X.max()-X.min())
        return scaled_df
    
# In[ ]:


class CustomEstimator(BaseEstimator):                #act as placeholder for real estimators
        def fit(self): pass
        def score(self): pass
