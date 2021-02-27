from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np


class PearsonSelector(BaseEstimator, TransformerMixin):       # TransformerMixin ensures we get fit_transform. BaseEstimator ensures we get                                                                 get_params and set_params
    
    def __init__(self, limit=0.4,):                                #sets a default limit for selecting features
        self.limit = limit
        
    def fit(self, X, y = None):
        X = X.copy()
        num_cols = X.select_dtypes(['float64', 'int64']).columns
        corr_matrix = X[num_cols].corr().abs()                                    #gets a positive correlation matrix
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))                     #gets half of the correlation matrix
        corr_df = corr_matrix.mask(mask)
        cols_to_drop = [col for col in corr_df.columns if any(corr_df[col]> self.limit)]  #select cols with an abs pc greater than limit
        
        self.cols_to_drop = cols_to_drop
        return self
        
    def transform(self, X):
        return X.drop(self.cols_to_drop, axis=1)                 

class MinMaxCust(BaseEstimator, TransformerMixin):   #MinMaxScaler returns an array so I need to customize it to have a dataframe instead
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        num_cols = X.select_dtypes(['float64', 'int64']).columns
        X[num_cols] = (X[num_cols] - X[num_cols].min(axis=0)) / (X[num_cols].max(axis=0) - X[num_cols].min(axis=0))
        return X
    
# In[ ]:
class MinMaxScalerCust(BaseEstimator, TransformerMixin):
    
    def fit(self, X, y=None):
        X = X.copy()
        num_cols = X.select_dtypes(['float64', 'int64']).columns
        self.num_cols = num_cols
        scaler = MinMaxScaler()
        scaler_fitted = scaler.fit(X[num_cols])
        self.scaler_fitted = scaler_fitted
        return self
    
    def transform(self, X):
        X[self.num_cols] = self.scaler_fitted.transform(X[self.num_cols])
        return X

class RobustScalerCust(BaseEstimator, TransformerMixin):
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        num_cols = X.select_dtypes(['float64', 'int64']).columns
        scaler = RobustScaler()
        X[num_cols] = scaler.fit(X[num_cols])
        return X

class StandardScalerCust(BaseEstimator, TransformerMixin):
    
    def fit(self, X, y=None):
        X = X.copy()
        num_cols = X.select_dtypes(['float64', 'int64']).columns
        self.num_cols = num_cols
        scaler = StandardScaler()
        scaler_fitted = scaler.fit(X[num_cols])
        self.scaler_fitted = scaler_fitted
        return self
    
    def transform(self, X):
        X[self.num_cols] = self.scaler_fitted.transform(X[self.num_cols])
        return X

    
class CustomEstimator(BaseEstimator):                   #act as placeholder for real estimators
        def fit(self): pass
        def score(self): pass


        from sklearn.base import BaseEstimator, TransformerMixin

class OutliersIQR(BaseEstimator, TransformerMixin):     #removes rows with outliers based on iqr and whis
    
    def __init__(self, whis = 1.5):
        self.whis = whis
        
    def fit(self, X, y=None):
        X = X.copy()
        num_cols = X.select_dtypes(['float64', 'int64']).columns
        upper_limit = X[num_cols].quantile(0.75) + self.whis*(X[num_cols].quantile(0.75)- X[num_cols].quantile(0.25))
        lower_limit = X[num_cols].quantile(0.25) - self.whis*(X[num_cols].quantile(0.75)-X[num_cols].quantile(0.25))
        self.upper_limit = upper_limit
        self.lower_limit = lower_limit
        self.num_cols = num_cols
        return self
    
    def transform(self, X, y=None):
        X = X[~((X[self.num_cols]>self.upper_limit)|(X[self.num_cols]<self.lower_limit)).any(axis=1)]

        return X

class OutliersZScores(BaseEstimator, TransformerMixin):      #removes rows with outliers based on zscores and a threshold
    
    def __init__(self, threshold = 3):
        self.threshold = threshold
        
    def fit(self, X, y=None):
        num_cols = X.select_dtypes(['float64', 'int64']).columns
        z_scores = np.abs((X[num_cols] - X[num_cols].mean())/X[num_cols].std())
        return self
    
    def transform(self, X, y=None):
        X = X.copy()
        X = z_scores[z_scores<self.threshold].dropna(axis=0)
        return X
    
class BinaryEncoder(BaseEstimator, TransformerMixin):         #binary encodes selected columns
    
    def __init__(self, selected_columns):
        self.selected_columns = selected_columns
    
    def fit(self, X, y=None):
        return self
    
    def convert_binary(self, value):
        if value == 'no':
            return 0
        else:
            return 1
    
    def transform(self, X, y=None):
        X = X.copy()
        X[self.selected_columns] = X[self.selected_columns].applymap(self.convert_binary)
        return X
    
    
    
class GetDummies(BaseEstimator, TransformerMixin):            #gets dummy variables
        
    def fit(self, X, y=None):
        cols = X.select_dtypes('object').columns
        self.cols = cols
        return self
    
    def transform(self, X):
        X = X.copy()
        X = pd.get_dummies(X, drop_first=False)
        try:
            X.drop(self.cols, inplace=True)
        finally:
            return X
        
        
class DropColumns(BaseEstimator, TransformerMixin):            #drops selected columns
    
    def __init__(self, selected_cols):
        self.selected_cols = selected_cols
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X = X.copy()
        X = X.drop(self.selected_cols, axis=1)
        return X

def get_best(pipe_dict, X_train, X_test, y_train):    #gets best_estimator calculated by GridSearch
    
    best_estimators = {}
    for name, pipe in pipe_dict.items():
        gs = GridSearchCV(pipe[0], param_grid = pipe[1], cv = 5, scoring = 'roc_auc')
        gs.fit(X_train, y_train)
        best_estimators[name] = gs.best_estimator_
    return best_estimators 

    
def plot_roc_prc(best_estimators, X_test, y_test):                    #plots roc and prc given a dict of estimators

    fprs = {}
    tprs = {}
    prcs = {}
    recalls = {}
    for name, estimator in best_estimators.items():
            preds = estimator.predict(X_test)
            probs = estimator.predict_proba(X_test)[:,1]
            fprs[name], tprs[name], _ = roc_curve(y_test, probs)
            prcs[name], recalls[name], _ = precision_recall_curve(y_test, probs)


    fig, [ax1,ax2] = plt.subplots(1,2)
    fig.set_size_inches(10,5)
    colors = ['g','c','m']

    for item in zip(fprs.values(), tprs.values(), fprs.keys(), colors):  
        ax1.plot([0,1],'k--')
        ax1.plot(item[0] , item[1], label = item[2], c = item[3])
        ax1.set_xlabel('fpr')
        ax1.set_ylabel('tpr')
        ax1.set_title('ROC AUC Comparison', {'fontsize':15})
        ax1.legend()

    for item in zip(prcs.values(), recalls.values(), prcs.keys(), colors):  
        ax2.plot([1,0],'k--')
        ax2.plot(item[0] , item[1], label = item[2], c = item[3])
        ax2.set_xlabel('recall')
        ax2.set_ylabel('precision')
        ax2.set_title('ROC PRC Comparison', {'fontsize':15})
        ax2.legend();   

def get_results(best_estimators, X_test, y_test):
    
    results = pd.DataFrame(index = ['AUC', 'Precision', 'Recall', 'F1'])
    for name, estimator in best_estimators.items():
        preds = estimator.predict(X_test)
        probs = estimator.predict_proba(X_test)[:,1]
        auc, precision, recall, f1 = roc_auc_score(y_test, probs), precision_score(y_test, preds), recall_score(y_test, preds), f1_score(y_test, preds)
        results[name] = [auc, precision, recall, f1]

    return results