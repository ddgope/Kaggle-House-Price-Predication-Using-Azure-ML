# House Prices: Advanced Regression Techniques

*Goal Of the Project:* Predict the price of a house by its features. If you are a buyer or seller of the house but you don't know the exact price of the house, so supervised machine learning regression algorithms can help you to predict the price of the house just providing features of the target house.
It is my job to predict the sales price for each house. For each Id in the test set, I must predict the value of the SalePrice variable. 

*Metric
Submissions are evaluated on Root-Mean-Squared-Error (RMSE) between the logarithm of the predicted value and the logarithm of the observed sales price. (Taking logs means that errors in predicting expensive houses and cheap houses will affect the result equally.)

I create two models in the environment of Azure Machine Learning Studio: one using Automated Machine Learning (i.e. AutoML) and one customized model uisng Python SDK whose hyperparameters are tuned using HyperDrive. I then compare the performance of both models and deploy the best performing model as a service using Azure Container Instances (ACI).

we will first compare the accuracy of AutoML vs HyperConfig's hyperparameter tuning of a XGBOOST Regressor in predicting house price given certain details about the house. We will then deploy the most accurate model to get an active endpoint that can be queried using those details to return a predicted house price.

The schematic below illustrates the path that is detailed in the rest of this write-up

![Project Workflow](img/Project_workflow.JPG?raw=true "Project Workflow") 

## Project Set Up and Installation
To set up this project in AzureML, please:

download the Auto MPG dataset and register as a dataset in Azure ML studio under the name 'mpg'

In order to run the project in Azure Machine Learning Studio, we will need the two Jupyter Notebooks:

automl.ipynb: for the AutoML experiment;
hyperparameter_tuning.ipynb: for the HyperDrive experiment.
The following files are also necessary:

heart_failure_clinical_records_dataset.csv: the dataset file. It can also be taken directly from Kaggle;
train.py: a basic script for manipulating the data used in the HyperDrive experiment;
scoring_file_v_1_0_0.py: the script used to deploy the model which is downloaded from within Azure Machine Learning Studio; &
env.yml: the environment file which is also downloaded from within Azure Machine Learning Studio.

## Dataset

The dataset used is taken from [Kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data) 

### File Description: This is Kaggle Competation. Downloaded from Kaggle.
* train.csv - the training set
* test.csv - the test set
* data_description.txt - full description of each column, originally prepared by Dean De Cock but lightly edited to match the column names used here
* sample_submission.csv - a benchmark submission from a linear regression on year and month of sale, lot square footage, and number of bedrooms

### Data Fileds
Here's a brief vabout data description file.
* SalePrice - the property's sale price in dollars. This is the target variable that you're trying to predict.
* MSSubClass: The building class
* MSZoning: The general zoning classification
* LotFrontage: Linear feet of street connected to property
* LotArea: Lot size in square feet
* Street: Type of road access
* Alley: Type of alley access
* LotShape: General shape of property
* LandContour: Flatness of the property
* Utilities: Type of utilities available
* LotConfig: Lot configuration
* LandSlope: Slope of property
* Neighborhood: Physical locations within Ames city limits
* Condition1: Proximity to main road or railroad
* Condition2: Proximity to main road or railroad (if a second is present)
* BldgType: Type of dwelling
* HouseStyle: Style of dwelling
* OverallQual: Overall material and finish quality
* OverallCond: Overall condition rating
* YearBuilt: Original construction date
* YearRemodAdd: Remodel date
* RoofStyle: Type of roof
RoofMatl: Roof material
Exterior1st: Exterior covering on house
Exterior2nd: Exterior covering on house (if more than one material)
MasVnrType: Masonry veneer type
MasVnrArea: Masonry veneer area in square feet
ExterQual: Exterior material quality
ExterCond: Present condition of the material on the exterior
Foundation: Type of foundation
BsmtQual: Height of the basement
BsmtCond: General condition of the basement
BsmtExposure: Walkout or garden level basement walls
BsmtFinType1: Quality of basement finished area
BsmtFinSF1: Type 1 finished square feet
BsmtFinType2: Quality of second finished area (if present)
BsmtFinSF2: Type 2 finished square feet
BsmtUnfSF: Unfinished square feet of basement area
TotalBsmtSF: Total square feet of basement area
Heating: Type of heating
HeatingQC: Heating quality and condition
CentralAir: Central air conditioning
Electrical: Electrical system
1stFlrSF: First Floor square feet
2ndFlrSF: Second floor square feet
LowQualFinSF: Low quality finished square feet (all floors)
GrLivArea: Above grade (ground) living area square feet
BsmtFullBath: Basement full bathrooms
BsmtHalfBath: Basement half bathrooms
FullBath: Full bathrooms above grade
HalfBath: Half baths above grade
Bedroom: Number of bedrooms above basement level
Kitchen: Number of kitchens
KitchenQual: Kitchen quality
TotRmsAbvGrd: Total rooms above grade (does not include bathrooms)
Functional: Home functionality rating
Fireplaces: Number of fireplaces
FireplaceQu: Fireplace quality
GarageType: Garage location
GarageYrBlt: Year garage was built
GarageFinish: Interior finish of the garage
GarageCars: Size of garage in car capacity
GarageArea: Size of garage in square feet
GarageQual: Garage quality
GarageCond: Garage condition
PavedDrive: Paved driveway
WoodDeckSF: Wood deck area in square feet
OpenPorchSF: Open porch area in square feet
EnclosedPorch: Enclosed porch area in square feet
3SsnPorch: Three season porch area in square feet
ScreenPorch: Screen porch area in square feet
PoolArea: Pool area in square feet
PoolQC: Pool quality
Fence: Fence quality
MiscFeature: Miscellaneous feature not covered in other categories
MiscVal: $Value of miscellaneous feature
MoSold: Month Sold
YrSold: Year Sold
SaleType: Type of sale
SaleCondition: Condition of sale

### Data Collection
*TODO*: Explain how you are accessing the data in your workspace.

### Data Preprocessing
Handling Missing Values

### Feature Engineering

###  Exploratory Data Analysis
	Autocorrelation Analysis
### Cross Validation

## Model Development

## Why Hyperparameter 

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search

### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Pipeline comparison

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions / Future work
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.

## Cluster clean up

## References
