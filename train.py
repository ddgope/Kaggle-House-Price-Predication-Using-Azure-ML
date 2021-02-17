#!/usr/bin/env python
# coding: utf-8

# ## House Prices: Advanced Regression Techniques 

# In[ ]:


pip install seaborn


# In[3]:


# Import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import argparse
import os

# importing necessary libraries
import numpy as np

from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

import joblib

# In[4]:


train=pd.read_csv("https://raw.githubusercontent.com/ddgope/Udacity-Capstone-House-Price-Predication-Using-Azure-ML/master/house-price-train-data.csv")
test=pd.read_csv("https://raw.githubusercontent.com/ddgope/Udacity-Capstone-House-Price-Predication-Using-Azure-ML/master/house-price-test-data.csv")


# In[5]:


print("Shape of Train:", train.shape)
print("Shape of Test:", test.shape)


# In[6]:


train.head()


# In[7]:


test.head(10)


# In[8]:


## Concat train and test data set
df=pd.concat((train,test))
temp_df=df
print("Shape of df:", df.shape)


# # Exploratory Data Analysis

# In[9]:


pd.set_option("display.max_columns",2000)
pd.set_option("display.max_rows",85)


# In[10]:


df.head(6)


# In[11]:


df.tail(10)


# In[12]:


df.info()


# In[13]:


df.describe()


# In[14]:


df.select_dtypes(include=['int64','float64']).columns


# In[15]:


df.select_dtypes(include=['object']).columns


# In[16]:


df=df.set_index("Id")


# In[17]:


df.head(6)


# ### Show the Null Values using heatmap

# In[19]:


#Checking Null Values
plt.figure(figsize=(16,9))
sns.heatmap(df.isnull(),yticklabels=False,cbar=False)


# In[20]:


#gte the Percentage of null values
null_percent=df.isnull().sum()/df.shape[0]*100
null_percent


# In[21]:


col_for_drop=null_percent[null_percent > 50].keys()
col_for_drop


# In[22]:


df=df.drop(col_for_drop,"columns")


# In[23]:


df.shape


# In[24]:


#find the unique value count
for i in df.columns:
    print(i + "\t" + str(len(df[i].unique())))


# In[25]:


# Describe the target
train["SalePrice"].describe()


# In[26]:


# Plot the distplot of target
plt.figure(figsize=(10,8))
bar=sns.distplot(train["SalePrice"])
bar.legend(["Skewness:{:.2f}".format(train['SalePrice'].skew())])


# # Correlation Heatmap

# In[27]:


plt.figure(figsize=(25,25))
ax=sns.heatmap(train.corr(),cmap="coolwarm",annot=True,linewidth=2)

#to fix the bug first and last row cut in half of heatmap plot
bottom,top=ax.get_ylim()
ax.set_ylim(bottom +0.5, top-0.5)


# In[29]:


## correlation heatmap of highly correlated features with SalePrice
hig_corr=train.corr()
hig_corr_features=hig_corr.index[abs(hig_corr["SalePrice"]) >=0.5 ]
hig_corr_features


# In[30]:


plt.figure(figsize=(15,10))
ax=sns.heatmap(train[hig_corr_features].corr(),cmap="coolwarm",annot=True,linewidth=3)

#to fix the bug first and last row cut in half of heatmap plot
bottom,top=ax.get_ylim()
ax.set_ylim(bottom +0.5, top-0.5)


# In[31]:


#Plot regplot to get the nature of highly corelated data
plt.figure(figsize=(16,9))
for i in range(len(hig_corr_features)):
    if i<=9:
        plt.subplot(3,4,i+1)
        plt.subplots_adjust(hspace=0.5,wspace=0.5)
        sns.regplot(data=train,x=hig_corr_features[i],y='SalePrice')


# ## Handling Missing Value

# In[32]:


missing_col=df.columns[df.isnull().any()]
missing_col


# In[33]:


## Handling missing value of Bsmt feature
bsmt_col=[ 'BsmtQual', 'BsmtCond', 'BsmtExposure',
 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF',
 'BsmtFullBath', 'BsmtHalfBath']


# In[34]:


bsmt_feat=df[bsmt_col]
bsmt_feat


# In[35]:


## Fill Missing Values
df['LotFrontage']=df['LotFrontage'].fillna(df['LotFrontage'].mean())


# In[36]:


df['BsmtCond']=df['BsmtCond'].fillna(df['BsmtCond'].mode()[0])
df['BsmtQual']=df['BsmtQual'].fillna(df['BsmtQual'].mode()[0])
df['BsmtFinSF1']=df['BsmtFinSF1'].fillna(df['BsmtFinSF1'].mode()[0])
df['BsmtFinSF2']=df['BsmtFinSF2'].fillna(df['BsmtFinSF2'].mode()[0])
df['BsmtUnfSF']=df['BsmtUnfSF'].fillna(df['BsmtUnfSF'].mode()[0])
df['TotalBsmtSF']=df['TotalBsmtSF'].fillna(df['TotalBsmtSF'].mode()[0])
df['BsmtFullBath']=df['BsmtFullBath'].fillna(df['BsmtFullBath'].mode()[0])
df['BsmtHalfBath']=df['BsmtHalfBath'].fillna(df['BsmtHalfBath'].mode()[0])


# In[37]:


df['FireplaceQu']=df['FireplaceQu'].fillna(df['FireplaceQu'].mode()[0])
df['GarageType']=df['GarageType'].fillna(df['GarageType'].mode()[0])


# In[38]:


df['GarageCars']=df['GarageCars'].fillna(df['GarageCars'].mode()[0])
df['GarageArea']=df['GarageArea'].fillna(df['GarageArea'].mode()[0])


# In[39]:


df.drop(['GarageYrBlt'],axis=1,inplace=True)


# In[40]:


df['GarageFinish']=df['GarageFinish'].fillna(df['GarageFinish'].mode()[0])
df['GarageQual']=df['GarageQual'].fillna(df['GarageQual'].mode()[0])
df['GarageCond']=df['GarageCond'].fillna(df['GarageCond'].mode()[0])


# In[41]:


df.isnull().sum()


# In[42]:


df['MasVnrType']=df['MasVnrType'].fillna(df['MasVnrType'].mode()[0])
df['MasVnrArea']=df['MasVnrArea'].fillna(df['MasVnrArea'].mode()[0])


# In[43]:


sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='coolwarm')


# In[44]:


df['BsmtExposure']=df['BsmtExposure'].fillna(df['BsmtExposure'].mode()[0])


# In[45]:


sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='YlGnBu')


# In[46]:


df.shape


# In[47]:


df.head()


# ## Handling Categorical Features

# In[48]:


columns=['MSZoning','Street','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood',
         'Condition2','BldgType','Condition1','HouseStyle','SaleType',
        'SaleCondition','ExterCond',
         'ExterQual','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',
        'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','Heating','HeatingQC',
         'CentralAir',
         'Electrical','KitchenQual','Functional',
         'FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive']


# In[49]:


len(columns)


# In[50]:


def category_onehot_multcols(multcolumns):
    df_final=final_df
    i=0
    for fields in multcolumns:
        
        print(fields)
        df1=pd.get_dummies(final_df[fields],drop_first=True)
        
        final_df.drop([fields],axis=1,inplace=True)
        if i==0:
            df_final=df1.copy()
        else:
            
            df_final=pd.concat([df_final,df1],axis=1)
        i=i+1
       
        
    df_final=pd.concat([final_df,df_final],axis=1)
        
    return df_final


# In[51]:


final_df=df


# In[52]:


final_df['SalePrice']


# In[53]:


final_df.shape


# In[54]:


final_df=category_onehot_multcols(columns)


# In[55]:


final_df.shape


# In[56]:


final_df =final_df.loc[:,~final_df.columns.duplicated()]


# In[57]:


final_df.shape


# In[58]:


final_df


# In[59]:


final_df.columns[final_df.isnull().any()]


# In[60]:


final_df.head(10)


# In[61]:


final_df.shape


# ## Feature Engineering / Selection to improve Accuracy

# In[62]:


# correlation Barplot
plt.figure(figsize=(9,16))
corr_feat_series=pd.Series.sort_values(train.corrwith(train.SalePrice))
sns.barplot(x=corr_feat_series,y=corr_feat_series.index,orient='h')


# In[63]:


# These columns are not importatnt.
fe_col_for_drop=['YrSold','LowQualFinSF','MiscVal','BsmtHalfBath','BsmtFinSF2','3SsnPorch','MoSold']


# In[64]:


final_df=final_df.drop(fe_col_for_drop,"columns")


# In[65]:


final_df.shape


# In[66]:


final_df.to_csv('houseprice.csv',index=False)


# In[67]:


df_Train=final_df.iloc[:1422,:]
df_Test=final_df.iloc[1422:,:]


# In[68]:


df_Train.head()


# In[69]:


df_Train.tail()


# In[70]:


df_Train.shape


# In[71]:


df_Test.head()


# In[72]:


df_Train.shape


# In[73]:


df_Test.drop(['SalePrice'],axis=1,inplace=True)


# In[74]:


df_Test.shape


# In[75]:


X_train=df_Train.drop(['SalePrice'],axis=1)
y_train=df_Train['SalePrice']


# In[78]:

from azureml.core.run import Run
run = Run.get_context()


# In[79]:


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--kernel', type=str, default='linear',help='Kernel type to be used in the algorithm')
    parser.add_argument('--penalty', type=float, default=1.0, help='Penalty parameter of the error term')

    args = parser.parse_args()
    run.log('Kernel type', np.str(args.kernel))
    run.log('Penalty', np.float(args.penalty))
   
    # X -> features, y -> label
    X = X_train
    y = y_train

    # dividing X, y into train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    # training a linear SVM Regression
    from sklearn.svm import SVR
    svm_model_linear = SVR(kernel=args.kernel, C=args.penalty).fit(X_train, y_train)
    svm_predictions = svm_model_linear.predict(X_test)

    # model accuracy for X_test
    accuracy = svm_model_linear.score(X_test, y_test)
    print('Accuracy of SVM Regression on test set: {:.2f}'.format(accuracy))
    run.log('Accuracy', np.float(accuracy))
    # creating a confusion matrix
    cm = confusion_matrix(y_test, svm_predictions)
    print(cm)

    os.makedirs('outputs', exist_ok=True)
    # files saved in the "outputs" folder are automatically uploaded into run history
    joblib.dump(svm_model_linear, 'outputs/model.joblib')


# In[80]:


if __name__ == '__main__':
    main()
