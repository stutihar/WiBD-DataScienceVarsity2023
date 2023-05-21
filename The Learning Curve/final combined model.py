import numpy as np
import pandas as pd
import sklearn.impute as sk
import sklearn_pandas as sp
#from sklearn.linear_model import LinearRegression
import pickle
import pprint
#from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from termcolor import colored as cl # text customization
from sklearn.linear_model import Lasso # Lasso algorithm
#from sklearn.linear_model import BayesianRidge # Bayesian algorithm
#from sklearn.linear_model import ElasticNet # ElasticNet algorithm



#Reading the train dataset
data=pd.read_csv(r"C:\Users\moumi\OneDrive\Desktop\Python\Wbid\2023-03-09\train.csv") #regex syntax
#data.to_csv(r"C:\Users\moumi\OneDrive\Desktop\Python\Wbid\2023-03-09\output files\train1.csv")

#Reading the test dataset
data_test=pd.read_csv(r"C:\Users\moumi\OneDrive\Desktop\Python\Wbid\2023-03-09\test.csv")
#data_test.to_csv(r"C:\Users\moumi\OneDrive\Desktop\Python\Wbid\2023-03-09\output files\test1.csv")

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

#Removing the columns Alley and PoolQC to improve the Rsquare value of the model from the train data
data=data.drop(["Alley","PoolQC"],axis=1)

#Removing the columns Alley and PoolQC from the test data
data_test=data_test.drop(["Alley","PoolQC"],axis=1)

#Checking if train data got imputed properly
for i in range(data.shape[1]):
   if(data.iloc[:,i].isna().sum()==0):
      print("data got imputed")

#Checking if test data got imputed properly
for i in range(data_test.shape[1]):
   if(data_test.iloc[:,i].isna().sum()==0):
      print("data_test got imputed")



#extracting saleprice and Id into response_variable and ID respectively from the train data set
y_data_train=data["SalePrice"]
y_data_train=pd.DataFrame(y_data_train)
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


#x_data_train.to_csv(r"C:\Users\moumi\OneDrive\Desktop\Python\Wbid\2023-03-09\output files\avvvvv.csv")

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
print(y_data_train)
data_test[numerical_cols] = scaler.fit_transform(data_test[numerical_cols])
#data_test.to_csv(r"C:\Users\moumi\OneDrive\Desktop\Python\Wbid\2023-03-09\output files\x3.csv")


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



#Splitting the train data into 80% train and 20% validation
x_train, x_val, y_train, y_val = train_test_split(x_data_train, y_data_train, test_size=0.2, random_state=42)
#print(x_train,x_val, y_train, y_val)


##############################################################################

#Gradient Boosting Regressor Model

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error, r2_score

# Train the model
RGB_model=GradientBoostingRegressor(n_estimators=100, max_depth=4, learning_rate=0.2, loss='squared_error')
RGB_model.fit(x_train, np.ravel(y_train))

# Evaluate the model on the train's test set which is x_val
y_pred_RGB = RGB_model.predict(x_val)
rmse = mean_squared_error(y_val, y_pred_RGB, squared=False)
print("RMSE for GradientBoostingRegressor: ", rmse)

r2 = RGB_model.score(x_val, y_val)
print("R-squared GradientBoost: ", r2)

#########################################################
#Lasso Regression

#Training the model
lasso_model = Lasso(alpha = 0.001,max_iter=100000)
lasso_model.fit(x_train, np.ravel(y_train))


#Using the trained model to predict the known 'SalePrice' for the testing part of the train data set 
SalePrice_Pred_train=lasso_model.predict(x_val)
SalePrice_Pred_train=pd.DataFrame(SalePrice_Pred_train)
y_val=pd.DataFrame(y_val)

#Calculate the RMSE for train data
rmse = mean_squared_error(y_val, SalePrice_Pred_train, squared=False)
print("RMSE for Lasso: ", rmse)

#RSquared value
r2=lasso_model.score(x_train,np.ravel(y_train))
print("R-squared Lasso: ", r2)

##########################################################################

#Ridge Regression Model

from sklearn.linear_model import Ridge

# Train the model
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(x_train, y_train)

# Evaluate the model on the test set
y_pred_ridge = ridge_model.predict(x_val)
rmse = mean_squared_error(y_val, y_pred_ridge, squared=False)
r2 = ridge_model.score(x_val, y_val)
print("RMSE for Ridge: ", rmse)
print("R-squared Ridge:  ", r2)


################################################################################
#Random Forest Model

from sklearn.ensemble import RandomForestRegressor

#Random forest regressor model part
#Creating the Random Forest Regressor Model
clf_rfr = RandomForestRegressor(random_state=0)

#Training the Model with X_train_encoded & y_train
clf_rfr.fit(x_train, np.ravel(y_train))

#Predicting price in validation set
y_pred_rf = clf_rfr.predict(x_val)

#Calculate the RMSE for train data
rmse = mean_squared_error(y_val, y_pred_rf, squared=False)
print("RMSE for Random forest: ", rmse)

#RSquared value
r2 = ridge_model.score(x_val, y_val)
print("R-Squared random forest: ", r2)

# Predict test set
grb=0.35 * RGB_model.predict(data_test)#0.35
grb=pd.DataFrame(grb)
#grb.to_csv(r"C:\Users\moumi\OneDrive\Desktop\Python\Wbid\2023-03-09\output files\grb1.csv")

rf=0.2 * clf_rfr.predict(data_test)#0.2
rf=pd.DataFrame(rf)
#rf.to_csv(r"C:\Users\moumi\OneDrive\Desktop\Python\Wbid\2023-03-09\output files\rf2.csv")

ridge=0.15 * ridge_model.predict(data_test)#0.15
ridge=pd.DataFrame(ridge)
#ridge.to_csv(r"C:\Users\moumi\OneDrive\Desktop\Python\Wbid\2023-03-09\output files\ridge2.csv")

lasso=0.3 * lasso_model.predict(data_test)#0.3
lasso=pd.DataFrame(lasso)
#lasso.to_csv(r"C:\Users\moumi\OneDrive\Desktop\Python\Wbid\2023-03-09\output files\lasso2.csv")

combined_dataframe=pd.concat([grb,rf,ridge,lasso],axis=1)
#combined_dataframe.to_csv(r"C:\Users\moumi\OneDrive\Desktop\Python\Wbid\2023-03-09\output files\combined_dataframe2.csv")
sum=combined_dataframe.sum(axis=1)
#sum.to_csv(r"C:\Users\moumi\OneDrive\Desktop\Python\Wbid\2023-03-09\output files\sum.csv")
print("Average of combined models:",sum)
sum=pd.DataFrame(sum)
Final_output=scaler2.inverse_transform(sum)
Final_output=pd.DataFrame(Final_output)
#Final_output.to_csv(r"C:\Users\moumi\OneDrive\Desktop\Python\Wbid\2023-03-09\output files\Final_output11.csv")

final_result=pd.concat([ID_test_data,Final_output],axis=1)
final_result.columns=['ID','SalePrice']
final_result.to_csv(r"C:\Users\moumi\OneDrive\Desktop\Python\Wbid\2023-03-09\output files\combined_model.csv")


