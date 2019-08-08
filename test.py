train_file = 'F:\\lizhi\\pfm_train.csv'
test_file = 'F:\\lizhi\\pfm_test.csv'

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

one_hot_enc = OneHotEncoder()
m_scaler = MinMaxScaler()
estimator = PCA(n_components=10)

def data_process(data, model='test', booster='tree'):
    data.drop(['EmployeeNumber', 'Over18', 'StandardHours'], axis=1, inplace=True)
    
    
    data['fit'] = ((data.EducationField=='Human Resources')&(data.Department!='Human Resources')).astype('int')
    data['fit'] = ((data.EducationField!='Human Resources')&(data.Department=='Human Resources')&(data.fit!=1)).astype('int')
    data['fit'] = ((data.EducationField=='Marketing')&(data.Department!='Sales')).astype('int')
    data['fit'] = ((data.EducationField!='Marketing')&(data.Department=='Sales')&(data.fit!=1)).astype('int')
    
    data = data.replace('Non-Travel', 1)
    data = data.replace('Travel_Rarely', 2)
    data = data.replace('Travel_Frequently', 3)
    data = data.replace('Sales', 1)
    data = data.replace('Research & Development', 2)
    data = data.replace('Human Resources', 3)
    data = data.replace('Life Sciences', 1)
    data = data.replace('Medical', 2)
    data = data.replace('Marketing', 4)
    data = data.replace('Technical Degree', 5)
    data = data.replace('Other', 6)
    data = data.replace('Male', 1)
    data = data.replace('Female', 2)
    data = data.replace('Sales Executive', 1)
    data = data.replace('Research Scientist', 2)
    data = data.replace('Laboratory Technician', 4)
    data = data.replace('Manufacturing Director', 5)
    data = data.replace('Healthcare Representative', 6)
    data = data.replace('Manager', 7)
    data = data.replace('Sales Representative', 8)
    data = data.replace('Research Director', 9)
    data = data.replace('Single', 1)
    data = data.replace('Married', 2)
    data = data.replace('Divorced', 3)
    data = data.replace('Yes', 1)
    data = data.replace('No', 2)

    data['HikePerYear'] = data.PercentSalaryHike / (data.YearsAtCompany + 1)
    data['WorkChange'] = data.TotalWorkingYears / (data.NumCompaniesWorked + 1)
    data['Radio'] = data.YearsSinceLastPromotion / data.Age
    
    #data.drop(['JobLevel','TotalWorkingYears','PerformanceRating'],axis=1,inplace=True)
    
    one_hot_feature = [
                       'fit', 
                       'BusinessTravel', 'Department', 'EducationField', 'Gender',
                       'JobRole', 'MaritalStatus', 'OverTime']
    m_feature = [
                 'HikePerYear', 'WorkChange', 'Radio',
                 #'BusinessTravel', 'Department', 'EducationField', 'Gender',
                 #'JobRole', 'MaritalStatus', 'OverTime',
                 
                 'Age', 'Education', 'EnvironmentSatisfaction', 'JobInvolvement',
                 'JobLevel', 'TotalWorkingYears', 'PerformanceRating',
                 'PercentSalaryHike', 'RelationshipSatisfaction',
                 'StockOptionLevel', 'TrainingTimesLastYear','JobSatisfaction', 'MonthlyIncome', 'NumCompaniesWorked',
                 'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion',
                 'YearsWithCurrManager']
    if booster == 'tree':
        if model == 'train':
            label = data.Attrition
            train_data = data.drop('Attrition', axis=1)
            return train_data, label
        return data
    else:
        one_hot_data = data.loc[:, one_hot_feature]
        mm_data = data.loc[:, m_feature]
        if model == 'train':
            label = data.Attrition
            one_hot_enc.fit(one_hot_data)
            m_scaler.fit(mm_data)
        o_data = one_hot_enc.transform(one_hot_data).toarray()
        m_data = m_scaler.transform(mm_data)
        #train_data = m_data
        train_data = np.c_[o_data, m_data]
        train_data = estimator.fit_transform(train_data)
        if model == 'train':
            return train_data, label
        return train_data
        
train = pd.read_csv(train_file)
test = pd.read_csv(test_file)
train_data, label = data_process(train, 'train', 'lr')
test_data = data_process(test, booster='lr')

#x_train, x_test, y_train, y_test = train_test_split(trainData_scale, target, test_size=0.2, shuffle=True)
x_train, x_test, y_train, y_test = train_test_split(train_data, label, test_size=0.3, shuffle=True)

import xgboost as xgb
dtrain = xgb.DMatrix(x_train, y_train)
dval = xgb.DMatrix(x_test, y_test)
dtmp = xgb.DMatrix(x_test)
dtest = xgb.DMatrix(test_data)
params = {'objective':'reg:logistic',
          'gamma':0.5,
          'lambda':2,
          'eta':0.1,
          #'n_estimator':1,
          #'sample':0.8,
          #'colsampe_bytree':0.8,
          'booster':'gbtree',
          'max_depth':1,
          #'min_child_weigth':1,
          'scale_pos_weight':1,     
          'eval_metric':'auc'}

#plt = params.items()
watch_list = [(dval, 'val')]
xgb.cv(params, dtrain, 1000, verbose_eval=100, early_stopping_rounds=50, nfold=5)
xgb_model1 = xgb.train(params, dtrain, 1000, evals=watch_list, verbose_eval=100, early_stopping_rounds=50)
y_xgb1 = xgb_model1.predict(dtmp)
1 - np.mean((y_xgb1 - y_test) ** 2)

import lightgbm as lgb

dtrain = lgb.Dataset(x_train, y_train)
dtest = lgb.Dataset(x_test, y_test, reference=dtrain)
params = {'task':'train',
          'reg_alpha':0.5,
          'boosting_type':'gbdt',
          'objective':'regression',
          'metric':{'auc'},
          'learning_rate':0.1,
          'feature_fraction':0.8,
          'bagging_fraction':0.8,
          'max_depth':1,
          'num_leaves': 2,
          'is_unbalance': 'true',
          'min_child_samples': 110,
          'max_bin':10,
          'verbose':1}

lgb.cv(params, dtrain, 1000, verbose_eval=100, early_stopping_rounds=50, nfold=5)
lgb_model1 = lgb.train(params, dtrain, 1000, verbose_eval=100, valid_sets=dtest, early_stopping_rounds=50)

y_lgb1 = lgb_model1.predict(x_test)
1 - np.mean((y_lgb1 - y_test) ** 2)

from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
rf = RandomForestRegressor(n_estimators=100,  max_depth=3, max_features=0.1)
rf.fit(x_train, y_train)
y_rf = rf.predict(x_test)
test_auc = metrics.roc_auc_score(y_test, y_rf)#验证集上的auc值
print('auc', test_auc)
print('score', 1 - np.mean((y_rf - y_test) ** 2))

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
lr = LogisticRegression(random_state=0)
param_grid = {'penalty':['l1'], 'C':[2]}
grid_search = GridSearchCV(lr, param_grid, cv=5)#, scoring='roc_auc'
grid_search.fit(x_train, y_train)
best_model = grid_search.best_estimator_
y_lr = best_model.predict_proba(x_test)
test_auc = metrics.roc_auc_score(y_test, y_lr[:,1])#验证集上的auc值
print('score', best_model.score(x_test, y_test))
print('auc', test_auc)
print(grid_search.best_params_)

from sklearn.svm import SVC
from sklearn import metrics
svc = SVC(probability=True)
svc.fit(x_train, y_train)
y_svm = svc.predict_proba(x_test)
test_auc = metrics.roc_auc_score(y_test, y_svm[:,1])#验证集上的auc值
print('score', svc.score(x_test, y_test))
print('auc', test_auc)

#y_ = y_lgb1+y_xgb1+y_lgb2+y_xgb2+y_lgb3+y_xgb3+y_lr[:,1]
y_ = 0.1*y_lgb1+0.1*y_xgb1+0.1*y_svm[:,1]+0.1*y_rf+0.6*y_lr[:,1]
1- np.mean((y_/5 - y_test) ** 2)

