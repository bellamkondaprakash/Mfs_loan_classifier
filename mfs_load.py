# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
#%%
from IPython import get_ipython


#%%
import pandas as pd 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import gc
from datetime import datetime 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier


#%%
df = pd.read_csv('sample_data_intw.csv')


#%%
df.head()


#%%
df.set_index(df['pdate'],inplace = True)

df = df.sort_index()

temp = df.drop(df[['pcircle','pdate']],axis = 1)


#%%
temp.info()


#%%
#checking wheather is any null values 
temp.isna().sum()


#%%
plt.rcParams['figure.figsize'] = (20, 12)
plt.rcParams['font.size'] = 10

sns.heatmap(temp.corr())


#%%
temp.iloc[:,3:].head(10)


#%%
temp.describe()


#%%
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from scipy.stats import shapiro


#%%
# days30_user = temp[['daily_decr30','rental30','sumamnt_ma_rech30','cnt_loans30','payback30','maxamnt_loans30']]


# #%%
# data_30_medianamtrech = (df['medianamnt_ma_rech30'].loc[(df['pdate'] <= '2016-06-30')])**(1/2)


#%%
min_max_scaler = MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(temp.iloc[:,3:])


#%%
temp_normalized = pd.DataFrame(np_scaled, columns = temp.iloc[:,3:].columns)


#%%
stat, p = shapiro(temp_normalized['sumamnt_ma_rech30'][:150])
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
    print('Sample looks Gaussian (fail to reject H0)')
else:
    print('Sample does not look Gaussian (reject H0)')

#%% [markdown]
# > ## Feature extraction using PCA

#%%
from sklearn.decomposition import PCA


#%%
x,y = temp_normalized.iloc[:,3:],temp.iloc[:,0]


#%%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1365)


#%%
x_train.shape


#%%
pca = PCA(n_components=10)
X_train = pca.fit_transform(x_train)
X_test = pca.transform(x_test)

#%% [markdown]
# > ## Applying the ML Model on x_train 

#%%
from sklearn.ensemble import RandomForestClassifier

rf_classifier = RandomForestClassifier(max_depth=85, oob_score = True,random_state=2356)
rf_model = rf_classifier.fit(X_train, y_train)

#%%
y_pred = rf_model.predict(X_test)


#%%
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
from multiprocessing import Pool,cpu_count
from time import time

def model_accuracy(y_test,y_pred):
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print('Accuracy', accuracy_score(y_test, y_pred))


#%%
param_test1 = {'n_estimators': range(50, 150, 20)}

#best_params is a dict you can pass directly to train a model with optimal settings 
def best_para_2model():
    currtime = time()
    
    grid_search = GridSearchCV(rf_classifier, param_grid=param_test1, cv=10, scoring='f1_macro', n_jobs=4)

    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_ 

    #best_params is a dict you can pass directly to train a model with optimal settings 
    best_model = RandomForestClassifier(param_test1,oob_score = True,random_state=2356)
    return best_model

#%%
result_list = []
def log_result(result):
    # This is called whenever foo_pool(i) returns a result.
    # result_list is modified only by the main process, not the pool workers.
    result_list.append(result)


def save_model(trained_model, model_file_path ):
    """
    :type trained_model: object
    :type model_file_path: str
    """
    from sklearn.externals import joblib
    model = joblib.dump(trained_model, model_file_path)
    return (model_file_path)


#%%

if __name__ == "__main__":
    pool = Pool(processes=4)
    model_accuracy(y_test,y_pred)
    pool.apply_async(best_para_2model, args = (), callback = log_result)
    pool.close()
    pool.join()
    save_model(result_list,'randf_model.pkl')