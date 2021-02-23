# House Prices: Advanced Regression Techniques 

# Import libraries

import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#import seaborn as sns

import argparse
import os
import joblib

from sklearn import datasets
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import xgboost as xgb

from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory
from azureml.core import Workspace, Dataset

train=pd.read_csv("https://raw.githubusercontent.com/ddgope/Udacity-Capstone-House-Price-Predication-Using-Azure-ML/master/house-price-train-data.csv")
test=pd.read_csv("https://raw.githubusercontent.com/ddgope/Udacity-Capstone-House-Price-Predication-Using-Azure-ML/master/house-price-test-data.csv")
train_len=len(train)

## Concat train and test data set
df=pd.concat((train,test),sort=False)

def missing_data(df):
    #Checking Null Values and get the Percentage of null values
    null_percent=df.isnull().sum()/df.shape[0]*100

    col_for_drop=null_percent[null_percent > 50].keys()

    df=df.drop(col_for_drop,"columns")

    # Handling Missing Value

    ## Fill Missing Values
    df['LotFrontage']=df['LotFrontage'].fillna(df['LotFrontage'].mean())
    df['BsmtCond']=df['BsmtCond'].fillna(df['BsmtCond'].mode()[0])
    df['BsmtQual']=df['BsmtQual'].fillna(df['BsmtQual'].mode()[0])
    df['BsmtFinSF1']=df['BsmtFinSF1'].fillna(df['BsmtFinSF1'].mode()[0])
    df['BsmtFinSF2']=df['BsmtFinSF2'].fillna(df['BsmtFinSF2'].mode()[0])
    df['BsmtUnfSF']=df['BsmtUnfSF'].fillna(df['BsmtUnfSF'].mode()[0])
    df['TotalBsmtSF']=df['TotalBsmtSF'].fillna(df['TotalBsmtSF'].mode()[0])
    df['BsmtFullBath']=df['BsmtFullBath'].fillna(df['BsmtFullBath'].mode()[0])
    df['BsmtHalfBath']=df['BsmtHalfBath'].fillna(df['BsmtHalfBath'].mode()[0])
    df['FireplaceQu']=df['FireplaceQu'].fillna(df['FireplaceQu'].mode()[0])
    df['GarageType']=df['GarageType'].fillna(df['GarageType'].mode()[0])
    df['GarageCars']=df['GarageCars'].fillna(df['GarageCars'].mode()[0])
    df['GarageArea']=df['GarageArea'].fillna(df['GarageArea'].mode()[0])
    df.drop(['GarageYrBlt'],axis=1,inplace=True)
    df['GarageFinish']=df['GarageFinish'].fillna(df['GarageFinish'].mode()[0])
    df['GarageQual']=df['GarageQual'].fillna(df['GarageQual'].mode()[0])
    df['GarageCond']=df['GarageCond'].fillna(df['GarageCond'].mode()[0])
    df['MasVnrType']=df['MasVnrType'].fillna(df['MasVnrType'].mode()[0])
    df['MasVnrArea']=df['MasVnrArea'].fillna(df['MasVnrArea'].mode()[0])
    df['BsmtExposure']=df['BsmtExposure'].fillna(df['BsmtExposure'].mode()[0])

    return df

columns=['MSZoning','Street','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood',
             'Condition2','BldgType','Condition1','HouseStyle','SaleType',
            'SaleCondition','ExterCond',
             'ExterQual','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',
            'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','Heating','HeatingQC',
             'CentralAir',
             'Electrical','KitchenQual','Functional',
             'FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive']

def category_onehot_multcols(multcolumns,final_df,df):
    df_final=final_df
    i=0
    for fields in multcolumns:
        
        #print(fields)
        df1=pd.get_dummies(final_df[fields],drop_first=True)
        
        final_df.drop([fields],axis=1,inplace=True)
        if i==0:
            df_final=df1.copy()
        else:
            
            df_final=pd.concat([df_final,df1],axis=1)
        i=i+1
       
        
    df_final=pd.concat([final_df,df_final],axis=1)
        
    return df_final

def clean_data(df):
    final_df=missing_data(df)
    final_df=category_onehot_multcols(columns,final_df,df)
    final_df =final_df.loc[:,~final_df.columns.duplicated()]

    # Feature Engineering / Selection to improve Accuracy
    # These columns are not importatnt.
    fe_col_for_drop=['YrSold','LowQualFinSF','MiscVal','BsmtHalfBath','BsmtFinSF2','3SsnPorch','MoSold']
    final_df=final_df.drop(fe_col_for_drop,"columns")

    df_Train=final_df[:train_len]
    df_Test=final_df[train_len:]

   # print(df_Train.shape)
    #print(df_Test.shape)

    # X -> features, y -> label/target variable 
    X_train=df_Train.drop(['SalePrice'],axis=1)
    y_train=df_Train['SalePrice'] #Separate the target variable 
    
    return X_train, y_train


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--max_depth', type=int, default=3, help="determines how deeply each tree is allowed to grow during any boosting round.")
    parser.add_argument('--learning_rate', type=float, default=0.1, help="step size shrinkage used to prevent overfitting. Range is [0,1]")
    parser.add_argument('--colsample_bytree', type=float, default=0.3, help="percentage of features used per tree. High value can lead to overfitting")
    parser.add_argument('--alpha', type=int, default=10, help="L1 regularization on leaf weights. A large value leads to more regularization.")
    parser.add_argument('--n_estimators', type=int, default=10, help="number of trees you want to build.")
    
    args = parser.parse_args()   
   
    # X -> features, y -> label
    X, y = clean_data(df)

    # training a xgboost Regression    
    data_dmatrix =xgb.DMatrix(data=X,label=y)
    
    #Split data into train and test sets.
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)    
    #print("Shapes of data: ", X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    
    #classifier=xgboost.XGBRegressor()
    #classifier.fit(X_train,y_train)   
    #preds=classifier.predict(X_test)
    #print(preds)

    #objective: determines the loss function to be used like reg:linear for regression problems, 
    #reg:logistic for classification problems with only decision, binary:logistic for classification problems with probability.

    xgboost_model_regression = xgb.XGBRegressor(objective ='reg:linear',max_depth=args.max_depth, learning_rate=args.learning_rate, colsample_bytree=args.colsample_bytree, alpha=args.alpha, n_estimators=args.n_estimators)
    
    xgboost_model_regression.fit(X_train,y_train)

    preds=xgboost_model_regression.predict(X_test)    

    rmse = np.sqrt(mean_squared_error(y_test, preds))    

    Rsquare = xgboost_model_regression.score(X_test, y_test)
    
    run = Run.get_context()
    
    run.log('mean_squared_error', np.float(rmse))                                            
    run.log("R-square", np.float(Rsquare))
    run.log("max_depth", np.int(args.max_depth))
    run.log("learning_rate",np.float(args.learning_rate))
    run.log("colsample_bytree",np.float(args.colsample_bytree))
    run.log("alpha", np.int(args.alpha))
    run.log("n_estimators", np.int(args.n_estimators))

    os.makedirs('outputs', exist_ok=True)
    # files saved in the "outputs" folder are automatically uploaded into run history
    joblib.dump(xgboost_model_regression, 'outputs/model.joblib')
 
    #https://towardsdatascience.com/how-to-deploy-a-local-ml-model-as-a-web-service-on-azure-machine-learning-studio-5eb788a2884c

if __name__ == '__main__':
    main()
