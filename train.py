# House Prices: Advanced Regression Techniques 

# Import libraries

import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#import seaborn as sns

import argparse
import os

from sklearn import datasets
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import joblib



train=pd.read_csv("https://raw.githubusercontent.com/ddgope/Udacity-Capstone-House-Price-Predication-Using-Azure-ML/master/house-price-train-data.csv")
test=pd.read_csv("https://raw.githubusercontent.com/ddgope/Udacity-Capstone-House-Price-Predication-Using-Azure-ML/master/house-price-test-data.csv")


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

from azureml.core.run import Run
run = Run.get_context()



def main():
    parser = argparse.ArgumentParser()

    # parser.add_argument('--kernel', type=str, default='linear',help='Kernel type to be used in the algorithm')
    # parser.add_argument('--penalty', type=float, default=1.0, help='Penalty parameter of the error term')

    args = parser.parse_args()
    # run.log('Kernel type', np.str(args.kernel))
    # run.log('Penalty', np.float(args.penalty))
   
    # X -> features, y -> label
    X, y = clean_data(df)

    # # training a xgboost Regression    
    # data_dmatrix =xgb.DMatrix(data=X,label=y)

    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
    #print("Shapes of data: ", X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    import xgboost
    classifier=xgboost.XGBRegressor()
    classifier.fit(X_train,y_train)    

    # xgboost_model_regression = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1,
    #             max_depth = 5, alpha = 10, n_estimators = 10)
    # xgboost_model_regression.fit(X_train,y_train)

    preds=classifier.predict(X_test)
    print(preds)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    print("RMSE: %f" % (rmse))

    run.log('mean squared error', np.float(rmse))     

    os.makedirs('outputs', exist_ok=True)
    # files saved in the "outputs" folder are automatically uploaded into run history
    joblib.dump(classifier, 'outputs/model.joblib')
 


if __name__ == '__main__':
    main()
