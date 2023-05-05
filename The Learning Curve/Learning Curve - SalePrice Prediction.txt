import numpy as np
import pandas as pd
import sklearn.impute as sk
import sklearn_pandas as sp
#from sklearn.linear_model import LinearRegression
import pickle
import pprint
#from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVR
from termcolor import colored as cl # text customization
from sklearn.linear_model import Lasso, LassoCV, RidgeCV # Lasso algorithm
#from sklearn.linear_model import BayesianRidge # Bayesian algorithm
#from sklearn.linear_model import ElasticNet # ElasticNet algorithm
import datetime
from sklearn.pipeline import make_pipeline
import os
os.chdir("c:/users/moumi/appdata/local/packages/pythonsoftwarefoundation.python.3.11_qbz5n2kfra8p0/localcache/local-packages/python311/site-packages")
import numpy
import scipy
import scipy.stats
from scipy.stats import skew, norm,probplot



#Reading the train dataset
data=pd.read_csv(r"C:\Users\moumi\OneDrive\Desktop\Python\Wbid\2023-03-09\train.csv") #regex syntax
#data.to_csv(r"C:\Users\moumi\OneDrive\Desktop\Python\Wbid\2023-03-09\output files\train1.csv")

#Reading the test dataset
data_test=pd.read_csv(r"C:\Users\moumi\OneDrive\Desktop\Python\Wbid\2023-03-09\test.csv")
#data_test.to_csv(r"C:\Users\moumi\OneDrive\Desktop\Python\Wbid\2023-03-09\output files\test1.csv")


#Probability and histplot for SalePrice data distribution
def QQ_plot2(data, measure):
    fig = plt.figure(figsize=(5,5))

    #Get the fitted parameters used by the function
    (mu, sigma) = norm.fit(data)

    #Kernel Density plot
    fig1 = fig.add_subplot(121)
    sns.histplot(data, kde=True, color="#bada55",edgecolor='#046e70')
    fig1.set_title(measure + ' Distribution ( mu = {:.2f} and sigma = {:.2f} )'.format(mu, sigma))
    plt.xlabel(measure)
    plt.ylabel('Frequency')

    #QQ plot
    fig2 = fig.add_subplot(122)
    res = probplot(data, plot=fig2)
    fig2.set_title(measure + 'Probability Plot(skewness:{:.6f} and kurtosis:{:.6f} )'.format(data.skew(),data.kurt()))

    plt.tight_layout()
    plt.show()


QQ_plot2(data["SalePrice"], 'Sales Price') 

#Normalizing the SalePrice column
data["SalePrice"] = np.log1p(data["SalePrice"])

#Probability and histplot for SalePrice data distribution after converting the distribution to normal distribution
def QQ_plot(data, measure):
    fig = plt.figure(figsize=(5,5))

    #Get the fitted parameters used by the function
    (mu, sigma) = norm.fit(data)

    #Kernel Density plot
    fig1 = fig.add_subplot(121)
    sns.histplot(data, kde=True, color="#bada55",edgecolor='#046e70')
    fig1.set_title(measure + ' Distribution ( mu = {:.2f} and sigma = {:.2f} )'.format(mu, sigma))
    plt.xlabel(measure)
    plt.ylabel('Frequency')

    #QQ plot
    fig2 = fig.add_subplot(122)
    res = probplot(data, plot=fig2)
    fig2.set_title(measure + 'Probability Plot(skewness: {:.6f} and kurtosis: {:.6f} )'.format(data.skew(),data.kurt()))

    plt.tight_layout()
    plt.show()


QQ_plot(data["SalePrice"], 'Sales Price')

###########################################
#Checking for Missing Data
all_data_na = (data.isnull().sum() / len(data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
print(missing_data.head(20))

#Visualization of the most missing columns
f, ax = plt.subplots(figsize=(14, 10))
plt.xticks(rotation='vertical')
sns.barplot(x=all_data_na.index, y=all_data_na)
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of missing values', fontsize=15)
plt.title('Percent missing data by feature', fontsize=15)
plt.show()

#Removing the columns Alley and PoolQC to improve the Rsquare value of the model from the train data
data=data.drop(["Alley","PoolQC","MiscFeature","Fence"],axis=1)

#Removing the columns Alley and PoolQC from the test data
data_test=data_test.drop(["Alley","PoolQC","MiscFeature","Fence"],axis=1)

#Converting year features to Age for both train and test set
data["HouseAgeFromLastSale"]=datetime.date.today().year-data["YrSold"]
data["HouseAgeFromBuiltYear"]=datetime.date.today().year-data["YearBuilt"]
data["RemodAge"]=data["YearRemodAdd"]-data["YrSold"]
data["GarageAge"]=datetime.date.today().year-data["GarageYrBlt"]
data=data.drop(["YrSold","YearRemodAdd",'MoSold',"GarageYrBlt"],axis=1)

data_test["HouseAgeFromLastSale"]=datetime.date.today().year-data_test["YrSold"]
data_test["HouseAgeFromBuiltYear"]=datetime.date.today().year-data_test["YearBuilt"]
data_test["RemodAge"]=data_test["YearRemodAdd"]-data_test["YrSold"]
data_test["GarageAge"]=datetime.date.today().year-data_test["GarageYrBlt"]
data_test=data_test.drop(["YrSold","YearRemodAdd",'MoSold',"GarageYrBlt"],axis=1)

#Convert columns to string columns
data = data.astype({'MSSubClass': str})
data_test = data_test.astype({'MSSubClass': str})

#Calclating the number of columns present in both the train and the test dataset
train_no_of_columns=data.shape[1]
test_no_of_columns=data_test.shape[1]
#print(train_no_of_columns,test_no_of_columns)

#Extracting the list of columns in both the dataframes (i.e. train and test )
list_of_train_columns=data.columns
list_of_test_columns=data_test.columns
#print(list_of_train_columns,list_of_test_columns)

a=0
b=0
#Checking if number of columns in test = (train-1) dataframes are equal or not. If equal we are validating the sequence of the columns
#If number of columns does not satisfy that condition
if(test_no_of_columns==train_no_of_columns-1):
    for i in range(data_test.shape[1]): 
         a=a+1  
         for j in range(data.shape[1]):
              if(list_of_test_columns[i]==list_of_train_columns[j]):
                   #print(list_of_test_columns[i],list_of_train_columns[j])
                   b=b+1
                   continue


if(a==len(list_of_test_columns) & b==len(list_of_test_columns)):
     print(a,b)
     print("we are good")

#Checking if the test and the train dataframe has any null values
print(data.isna().sum())
print(data_test.isna().sum())
    

#Imputing the missing data in train dataset
for i in range(data.shape[1]):
    if(data.iloc[:,i].isna().sum()!=0):         
         if data.iloc[:,i].dtype == np.dtype('O'): 
              data.iloc[:,i]=data.iloc[:,i].fillna("unknown")
         else:
           impute_f=sk.SimpleImputer(strategy='mean')
           data.iloc[:,i]=impute_f.fit_transform(data.iloc[:,i].values.reshape(-1,1))
           #print(data.iloc[:,i])

#Imputing the missing data in test dataset
for i in range(data_test.shape[1]):
    if(data_test.iloc[:,i].isna().sum()!=0):         
         if data_test.iloc[:,i].dtype == np.dtype('O'):
              data_test.iloc[:,i]=data_test.iloc[:,i].fillna("unknown")
         else:
           impute_f=sk.SimpleImputer(strategy='mean')
           data_test.iloc[:,i]=impute_f.fit_transform(data_test.iloc[:,i].values.reshape(-1,1))
           #print(data.iloc[:,i])


#Checking if train data got imputed properly
for i in range(data.shape[1]):
   if(data.iloc[:,i].isna().sum()==0):
      print("data got imputed")

#Checking if test data got imputed properly
for i in range(data_test.shape[1]):
   if(data_test.iloc[:,i].isna().sum()==0):
      print("data_test got imputed")

#Combining Features on corr coeff wrt SalePrice to see if we can create better features
corr=data.corr()["SalePrice"]
corr=pd.DataFrame(corr)
corr.columns=["SalePrice"]
#corr.to_csv(r"C:\Users\moumi\OneDrive\Desktop\Python\04-21-2023\output\corr.csv")
#print(corr.sort_values("SalePrice", ascending=False).head(6))

#########################################################################################

#Scatterplots for BsmtFinSF1,BsmtFinSF2,1stFlrSF,2ndFlrSF separately with SalePrice
fig2 = plt.figure(figsize=(20,10))
fig3 = fig2.add_subplot(121)
sns.scatterplot(data,x=data["BsmtFinSF1"],y=data["SalePrice"])
plt.title("Correlation Coeff between BsmtFinSF1 and SalePrice: ({:1.5f})".format(data["BsmtFinSF1"].corr(data["SalePrice"])))
fig4 = fig2.add_subplot(122)
sns.scatterplot(data,x=data["BsmtFinSF2"],y=data["SalePrice"])
plt.title("Correlation Coeff between BsmtFinSF2 and SalePrice: ({:1.5f})".format(data["BsmtFinSF2"].corr(data["SalePrice"])))
plt.show()
fig3 = plt.figure(figsize=(20,10))
fig5 = fig3.add_subplot(121)
sns.scatterplot(data,x=data["1stFlrSF"],y=data["SalePrice"])
plt.title("Correlation Coeff between 1stFlrSF and SalePrice: ({:1.2f})".format(data["1stFlrSF"].corr(data["SalePrice"])))
fig6 = fig3.add_subplot(122)
sns.scatterplot(data,x=data["2ndFlrSF"],y=data["SalePrice"])
plt.title("Correlation Coeff between 2ndFlrSF and SalePrice: ({:1.2f})".format(data["2ndFlrSF"].corr(data["SalePrice"])))
plt.show()

df= data["BsmtFinSF1"] +data["BsmtFinSF2"] +data["1stFlrSF"] +data["2ndFlrSF"]
df=pd.DataFrame(df)
df.columns=["TotalSqrFootage"]
df=pd.concat([df["TotalSqrFootage"],data["SalePrice"]],axis=1)
#df.to_csv(r"C:\Users\moumi\OneDrive\Desktop\Python\04-21-2023\output\df1.csv")
fig4 = plt.figure(figsize=(20,10))
fig7 = fig4.add_subplot(121)
sns.scatterplot(df,x=df["TotalSqrFootage"],y=df["SalePrice"])
plt.title("Correlation Coeff between TotalSqrFootage and SalePrice: ({:1.2f})".format(df["TotalSqrFootage"].corr(df["SalePrice"])))
plt.show()

#Adding the new column "TotalSqrFootage" to train data and test data and droping the old columns
data["TotalSqrFootage"]=data["BsmtFinSF1"] +data["BsmtFinSF2"] +data["1stFlrSF"] +data["2ndFlrSF"]
data_test["TotalSqrFootage"]=data_test["BsmtFinSF1"] +data_test["BsmtFinSF2"] +data_test["1stFlrSF"] +data_test["2ndFlrSF"]
data=data.drop(["BsmtFinSF1","2ndFlrSF","1stFlrSF","BsmtFinSF2"],axis=1)
data_test=data_test.drop(["BsmtFinSF1","2ndFlrSF","1stFlrSF","BsmtFinSF2"],axis=1)
#data.to_csv(r"C:\Users\moumi\OneDrive\Desktop\Python\04-21-2023\output\data1.csv")

##########################################################################################

#Scatterplots for FullBath,HalfBath,BsmtFullBath,BsmtHalfBath separately with SalePrice
fig2 = plt.figure(figsize=(20,10))
fig3 = fig2.add_subplot(121)
sns.scatterplot(data,x=data["FullBath"],y=data["SalePrice"])
plt.title("Correlation Coeff between FullBath and SalePrice: ({:1.5f})".format(data["FullBath"].corr(data["SalePrice"])))
fig4 = fig2.add_subplot(122)
sns.scatterplot(data,x=data["HalfBath"],y=data["SalePrice"])
plt.title("Correlation Coeff between HalfBath and SalePrice: ({:1.5f})".format(data["HalfBath"].corr(data["SalePrice"])))
plt.show()
fig3 = plt.figure(figsize=(20,10))
fig5 = fig3.add_subplot(121)
sns.scatterplot(data,x=data["BsmtFullBath"],y=data["SalePrice"])
plt.title("Correlation Coeff between BsmtFullBath and SalePrice: ({:1.2f})".format(data["BsmtFullBath"].corr(data["SalePrice"])))
fig6 = fig3.add_subplot(122)
sns.scatterplot(data,x=data["BsmtHalfBath"],y=data["SalePrice"])
plt.title("Correlation Coeff between BsmtHalfBath and SalePrice: ({:1.2f})".format(data["BsmtHalfBath"].corr(data["SalePrice"])))
plt.show()

df= (data['FullBath'] + (0.5 * data['HalfBath']) + data['BsmtFullBath'] + (0.5 * data['BsmtHalfBath']))
df=pd.DataFrame(df)
df.columns=["TotalBathrooms"]
df=pd.concat([df["TotalBathrooms"],data["SalePrice"]],axis=1)
#df.to_csv(r"C:\Users\moumi\OneDrive\Desktop\Python\04-21-2023\output\df1.csv")
fig4 = plt.figure(figsize=(20,10))
fig7 = fig4.add_subplot(121)
sns.scatterplot(df,x=df["TotalBathrooms"],y=df["SalePrice"])
plt.title("Correlation Coeff between TotalBathrooms and SalePrice: ({:1.2f})".format(df["TotalBathrooms"].corr(df["SalePrice"])))
plt.show()

data['TotalBathrooms'] = (data['FullBath'] + (0.5 * data['HalfBath']) + data['BsmtFullBath'] + (0.5 * data['BsmtHalfBath']))
data_test["TotalBathrooms"]=(data_test['FullBath'] + (0.5 * data_test['HalfBath']) + data_test['BsmtFullBath'] + (0.5 * data_test['BsmtHalfBath']))
data=data.drop(["FullBath","HalfBath","BsmtFullBath","BsmtHalfBath"],axis=1)
data_test=data_test.drop(["FullBath","HalfBath","BsmtFullBath","BsmtHalfBath"],axis=1)

################################################
#Scatterplots for OpenPorchSF,3SsnPorch,EnclosedPorch,ScreenPorch and WoodDeckSF separately with SalePrice
fig5 = plt.figure(figsize=(20,10))
fig8 = fig5.add_subplot(121)
sns.scatterplot(data,x=data["OpenPorchSF"],y=data["SalePrice"])
plt.title("Correlation Coeff between OpenPorchSF and SalePrice: ({:1.5f})".format(data["OpenPorchSF"].corr(data["SalePrice"])))
fig9 = fig5.add_subplot(122)
sns.scatterplot(data,x=data["3SsnPorch"],y=data["SalePrice"])
plt.title("Correlation Coeff between 3SsnPorch and SalePrice: ({:1.5f})".format(data["3SsnPorch"].corr(data["SalePrice"])))
plt.show()
fig6 = plt.figure(figsize=(20,10))
fig10 = fig6.add_subplot(121)
sns.scatterplot(data,x=data["EnclosedPorch"],y=data["SalePrice"])
plt.title("Correlation Coeff between EnclosedPorch and SalePrice: ({:1.2f})".format(data["EnclosedPorch"].corr(data["SalePrice"])))
fig11 = fig6.add_subplot(122)
sns.scatterplot(data,x=data["ScreenPorch"],y=data["SalePrice"])
plt.title("Correlation Coeff between ScreenPorch and SalePrice: ({:1.2f})".format(data["ScreenPorch"].corr(data["SalePrice"])))
fig7 = plt.figure(figsize=(20,10))
fig11 = fig7.add_subplot(121)
sns.scatterplot(data,x=data["WoodDeckSF"],y=data["SalePrice"])
plt.title("Correlation Coeff between WoodDeckSF and SalePrice: ({:1.2f})".format(data["WoodDeckSF"].corr(data["SalePrice"])))
plt.show()

df=data["OpenPorchSF"] +data["3SsnPorch"] +data["EnclosedPorch"] +data["ScreenPorch"]+data["WoodDeckSF"]
df=pd.DataFrame(df)
df.columns=["TotalPorchSF"]
df=pd.concat([df["TotalPorchSF"],data["SalePrice"]],axis=1)
#df.to_csv(r"C:\Users\moumi\OneDrive\Desktop\Python\04-21-2023\output\df1.csv")
fig4 = plt.figure(figsize=(20,10))
fig7 = fig4.add_subplot(121)
sns.scatterplot(df,x=df["TotalPorchSF"],y=df["SalePrice"])
plt.title("Correlation Coeff between TotalPorchSF and SalePrice: ({:1.2f})".format(df["TotalPorchSF"].corr(df["SalePrice"])))
plt.show()

data['TotalPorchSF'] = data["OpenPorchSF"] +data["3SsnPorch"] +data["EnclosedPorch"] +data["ScreenPorch"]+data["WoodDeckSF"]
data_test["TotalPorchSF"]=data_test["OpenPorchSF"] +data_test["3SsnPorch"] +data_test["EnclosedPorch"] +data_test["ScreenPorch"]+data_test["WoodDeckSF"]
data=data.drop(["OpenPorchSF","3SsnPorch","EnclosedPorch","ScreenPorch","WoodDeckSF"],axis=1)
data_test=data_test.drop(["OpenPorchSF","3SsnPorch","EnclosedPorch","ScreenPorch","WoodDeckSF"],axis=1)
##########################
#plt.figure(figsize=(12, 6))
#sns.heatmap(data.corr(),
#            cmap = 'BrBG',
#            fmt = '.2f',
#            linewidths = 2,
#            annot = True)

################################################################################################
     
#extracting saleprice and Id into response_variable and ID respectively from the train data set
y_data_train=data["SalePrice"]
y_data_train=pd.DataFrame(y_data_train)
y_data_train.columns=["SalePrice"]
#y_data_train.to_csv(r"C:\Users\moumi\OneDrive\Desktop\Python\04-21-2023\output\y_data_train2.csv")
data=data.drop('SalePrice',axis=1)

ID_test_data=data_test["Id"]
#ID_test_data.to_csv(r"C:\Users\moumi\OneDrive\Desktop\Python\Wbid\2023-03-09\output files\x10000.csv")
x_data_train=data.drop('Id',axis=1)
data_test=data_test.drop('Id',axis=1)

#Extract the numerical and categorical columns names for future use
categorical_cols = x_data_train.select_dtypes(include='object').columns.tolist()
numerical_cols = x_data_train.select_dtypes(exclude='object').columns.tolist()

#creating dummies for train dataset      
x_data_train=pd.get_dummies(x_data_train)

# Saving the columns of the data dataframe after creating dummies in a list
cols = x_data_train.columns.tolist()

#creating dummies for test dataset      
data_test=pd.get_dummies(data_test)

#Using the columns list of the train data set after creating dummies to alter the number of columns in the test data set after creating dummies
data_test = data_test.reindex(columns=cols).fillna(0)

# Scaling the numercial x_data_train, y_data_train and data_test using StandardScaler()
scaler = StandardScaler()
scaler2 = StandardScaler()
x_data_train[numerical_cols] = scaler.fit_transform(x_data_train[numerical_cols])
y_data_train=scaler2.fit_transform(y_data_train)
#print(y_data_train)
data_test[numerical_cols] = scaler.fit_transform(data_test[numerical_cols])
#data_test.to_csv(r"C:\Users\moumi\OneDrive\Desktop\Python\Wbid\2023-03-09\output files\x3.csv")
x_data_train=pd.DataFrame(x_data_train)
y_data_train=pd.DataFrame(y_data_train)
y_data_train.columns=["SalePrice"]
#y_data_train.to_csv(r"C:\Users\moumi\OneDrive\Desktop\Python\04-21-2023\output\y_data_train1551.csv")


#checking the correlation between the predictors with the train data
cor_matrix = x_data_train.corr().abs()
#print(cor_matrix)

#extracting the upper traingle of the correlation matrix where correlation coeff is greater than 0.8
upper_tri = cor_matrix.where(abs(np.triu(cor_matrix,k=1))>0.8)
#print(upper_tri)

#Extracting the column name from upper_tri which has to be dropped due to corr coef >0.8
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.8)]
#print(to_drop)
x_data_train=x_data_train.drop(x_data_train[to_drop],axis=1)
x_data_train=pd.DataFrame(x_data_train)
#x_data_train.to_csv(r"C:\Users\moumi\OneDrive\Desktop\Python\Wbid\2023-03-09\output files\x_new.csv")

data_test=data_test.drop(data_test[to_drop],axis=1)
data_test=pd.DataFrame(data_test)


#data_test.to_csv(r"C:\Users\moumi\OneDrive\Desktop\Python\04-21-2023\output\data_test.csv")
#x_data_train.to_csv(r"C:\Users\moumi\OneDrive\Desktop\Python\04-21-2023\output\x_data_train.csv")

#Splitting the train data into 80% train and 20% validation
x_train, x_val, y_train, y_val = train_test_split(x_data_train, y_data_train, test_size=0.2, random_state=42)
#print(x_train,x_val, y_train, y_val)

#################
#Setup cross validation folds:
kf = KFold(n_splits=12, shuffle=True, random_state=11)

# Set up Parameters 
lasso_alphas = [0.001,0.0005, 0.00055, 0.0006, 0.00065, 0.0007]

svr_grid = {'C': [20, 22], 'epsilon':[0.008, 0.009], 'gamma': [0.001, 0.002, 0.0025]}


# Lasso Regression
lasso = make_pipeline(RobustScaler(),
                      LassoCV(max_iter=10000000, alphas=lasso_alphas,
                              random_state=11, cv=kf))
lasso.fit(x_train, np.ravel(y_train))

#Using the trained model to predict the known 'SalePrice' for the testing part of the train data set
y_pred_lasso = lasso.predict(x_val) 
#SalePrice_Pred_train=lasso.predict(x_val)
#SalePrice_Pred_train=pd.DataFrame(SalePrice_Pred_train)
#y_val=pd.DataFrame(y_val)

#Calculate the RMSE for train data
rmse = mean_squared_error(y_val, y_pred_lasso, squared=False)
print("RMSE for Lasso: ", rmse)

#RSquared value
r2score=metrics.r2_score(y_val,y_pred_lasso)
print("R2squared Lasso: ", r2score)

# Support Vector Regression
svr = make_pipeline(RobustScaler(),
                    GridSearchCV(SVR(),svr_grid, cv=kf))
svr.fit(x_train, np.ravel(y_train))

y_pred_SVR = svr.predict(x_val)

#SalePrice_Pred_train=svr.predict(x_val)
#SalePrice_Pred_train=pd.DataFrame(SalePrice_Pred_train)
#y_val=pd.DataFrame(y_val)

#Calculate the RMSE for train data
rmse = mean_squared_error(y_val, y_pred_SVR, squared=False)
print("RMSE for svr: ", rmse)

#RSquared value
r2score=metrics.r2_score(y_val,y_pred_SVR)
print("R2squared SVR: ", r2score)

#Gradient Boosting Regressor Model

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error, r2_score

# Train the model
RGB_model=make_pipeline(RobustScaler(),GradientBoostingRegressor(n_estimators=100, max_depth=4, learning_rate=0.2, loss='squared_error'))
RGB_model.fit(x_train, np.ravel(y_train))

# Evaluate the model on the train's test set which is x_val
y_pred_RGB = RGB_model.predict(x_val)

#Calculate RMSE
rmse = mean_squared_error(y_val, y_pred_RGB, squared=False)
print("RMSE for GradientBoostingRegressor: ", rmse)

#RSquared value
r2score=metrics.r2_score(y_val,y_pred_RGB)
print("R2squared GradientBoostingRegressor: ", r2score)

#Random Forest Model

from sklearn.ensemble import RandomForestRegressor

#Random forest regressor model part
#Creating the Random Forest Regressor Model
clf_rfr = make_pipeline(RobustScaler(),RandomForestRegressor(random_state=0))

#Training the Model with X_train_encoded & y_train
clf_rfr.fit(x_train, np.ravel(y_train))

#Predicting price in validation set
y_pred_rf = clf_rfr.predict(x_val)

#Calculate the RMSE for train data
rmse = mean_squared_error(y_val, y_pred_rf, squared=False)
mse=mean_squared_error(y_val, y_pred_rf)
print("RMSE for Random forest: ", rmse)
print("MSE for Random forest: ", mse)

#RSquared value
r2score=metrics.r2_score(y_val,y_pred_rf)
print("R2squared rf: ", r2score)

#Ridge Regression Model

from sklearn.linear_model import Ridge

# Train the model
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(x_train, y_train)

# Evaluate the model on the test set
y_pred_ridge = ridge_model.predict(x_val)

#Calculate RMSE
rmse = mean_squared_error(y_val, y_pred_ridge, squared=False)
print("RMSE for Ridge: ", rmse)

#RSquared value
r2score=metrics.r2_score(y_val,y_pred_ridge)
print("R2squared Ridge: ", r2score)

##############################################
# Predict test set
grb=0.2 * RGB_model.predict(data_test)#0.35
grb=pd.DataFrame(grb)

rf=0.05 * clf_rfr.predict(data_test)#0.2
rf=pd.DataFrame(rf)

svr=0.25 * svr.predict(data_test)#0.15
svr=pd.DataFrame(svr)

lasso=0.3 * lasso.predict(data_test)#0.3
lasso=pd.DataFrame(lasso)

ridge=0.2 * ridge_model.predict(data_test)
ridge=pd.DataFrame(ridge)

combined_dataframe=pd.concat([grb,rf,svr,lasso,ridge],axis=1)
sum=combined_dataframe.sum(axis=1)
print("Average of combined models:",sum)
sum=pd.DataFrame(sum)
Final_output=scaler2.inverse_transform(sum)

#Final_output=clf_rfr.inverse_transform(Final_output)
#Final_output=RGB_model.inverse_transform(Final_output)
#Final_output=svr.inverse_transform(Final_output)
#Final_output=lasso.inverse_transform(Final_output)
Final_output=np.exp(Final_output)

Final_output=pd.DataFrame(Final_output)
#Final_output.to_csv(r"C:\Users\moumi\OneDrive\Desktop\Python\Wbid\2023-03-09\output files\Final_output11.csv")

final_result=pd.concat([ID_test_data,Final_output],axis=1)
final_result.columns=['ID','SalePrice']
#final_result.to_csv(r"C:\Users\moumi\OneDrive\Desktop\Python\04-21-2023\output\jjjjj.csv")
























