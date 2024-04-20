import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statistics
from math import *
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import tensorflow as tf


unemployment_df = pd.read_csv(r"C:\Users\USER\Downloads\global_unemployment_data.csv")
print(unemployment_df)
print(unemployment_df.head())
print(unemployment_df.info())

#checking for null values
print(unemployment_df.isnull().sum())
#dropping null values
df = unemployment_df.dropna()
print(df.isnull().sum())
#shape
print(df.shape)
#check for duplicates
print(f"there is {df.duplicated().sum()} duplicates")
#descriptive statistics
print(df.describe())
#nunique values
print(df.nunique())
print(df['2024'])
print(df.head())


#print data for Ghana
Ghana_df = unemployment_df[unemployment_df.country_name == "Ghana"]
print(Ghana_df)
#Ghana distribution in 2024 based on age group
sns.barplot(y= "2024",x="age_group",hue="age_group",data=Ghana_df)
plt.title("Ghana unemployment for 2024 based on age group")
plt.show()
#group by age group for 2014-2016
AG_Ghana_df = Ghana_df.groupby("age_group")[["2014","2015","2016"]].sum()
print(AG_Ghana_df)
sns.barplot(x="age_group",y="2014",hue="age_group",data=AG_Ghana_df)
plt.title("Ghana unemployment for 2014 based on sum of age group")
plt.show()
sns.barplot(x="age_group",y="2015",hue="age_group",data=AG_Ghana_df)
plt.title("Ghana unemployment for 2015 based on sum of age group")
plt.show()
sns.barplot(x="age_group",y="2016",hue="age_group",data=AG_Ghana_df)
plt.title("Ghana unemployment for 2016 based on sum of age group")
plt.show()
#just try a pairplot on the above
sns.pairplot(AG_Ghana_df[["2014","2015","2016"]])
plt.show()
#make subplots
fig,axes = plt.subplots(2,2,figsize=(16, 8))
axes[0,0].set_title("Ghana age group distribution for 2014")
sns.barplot(x="age_group",y="2014",hue="age_group",data=AG_Ghana_df,ax=axes[0,0])
axes[0,1].set_title("Ghana age group distribution for 2015")
sns.barplot(x="age_group",y="2015",hue="age_group",data=AG_Ghana_df,ax=axes[0,1])
axes[1,0].set_title("Ghana age group distribution for 2016")
sns.barplot(x="age_group",y="2016",hue="age_group",data=AG_Ghana_df,ax=axes[1,0])
plt.show()
#group by age group and sex for the totals in years 2014-2017
AGS_Ghana_df = Ghana_df.groupby(["age_group","sex"])[["2014","2015","2016","2017"]].sum()
print(AGS_Ghana_df)
fig,axes = plt.subplots(2,2,figsize=(16, 8))
axes[0,0].set_title("Ghana age group distribution for 2014")
sns.barplot(x="age_group",y="2014",hue="sex",data=AGS_Ghana_df,ax=axes[0,0])
axes[0,1].set_title("Ghana age group distribution for 2015")
sns.barplot(x="age_group",y="2015",hue="sex",data=AGS_Ghana_df,ax=axes[0,1])
axes[1,0].set_title("Ghana age group distribution for 2016")
sns.barplot(x="age_group",y="2016",hue="sex",data=AGS_Ghana_df,ax=axes[1,0])
axes[1,1].set_title("Ghana age group distribution for 2017")
sns.barplot(x="age_group",y="2017",hue="sex",data=AGS_Ghana_df,ax=axes[1,1])
plt.show()
"""Sum_Ghana_df = Ghana_df[["2015","2016","2017"]].sum(axis=1)
print(Sum_Ghana_df)"""
#General 2024 distribution
sns.histplot(data= df,x='2024', bins=20, kde=True)
plt.title('Unemployment Rate Distribution in 2024')
plt.xlabel('Unemployment Rate (%)')
plt.show()
#General 2023 distribtion
sns.histplot(data=df,x="2023", bins=20, kde=True)
plt.title('Unemployment Rate Distribution in 2023')
plt.xlabel('Unemployment Rate (%)')
plt.show()
#Use pairplots
sns.pairplot(df[["2020","2021","2022","2023","2024"]])
plt.show()
#find the general distribution for each country in 2024 using a scatterplot
sns.scatterplot(x="country_name",y="2024",data=df,hue="age_group")
plt.title("Unemployment distribution for each country in 2024")
plt.show()
#find the total distribution for each country in 2024
country_df = df.groupby("country_name")[["2024"]].sum()
print(country_df)
sns.scatterplot(x="country_name",y="2024",data=country_df)
plt.title("Total Unemployment distribution for each country in 2024")
plt.show()
#find the  distribution for each country in 2020-2024 as subplots
fig,axes = plt.subplots(2,3,figsize=(16,8))
axes[0,0].set_title("Distribution for each country from 2020 to 2024")
sns.scatterplot(x="country_name",y="2020",hue="age_group",data=df,ax=axes[0,0])
sns.scatterplot(x="country_name",y="2021",hue="age_group",data=df,ax=axes[0,1])
sns.scatterplot(x="country_name",y="2022",hue="age_group",data=df,ax=axes[0,2])
sns.scatterplot(x="country_name",y="2023",hue="age_group",data=df,ax=axes[1,0])
sns.scatterplot(x="country_name",y="2024",hue="age_group",data=df,ax=axes[1,1])
plt.show()
#use a pairplot
sns.pairplot(df[["2020","2021","2022","2023","2024"]])
plt.show()

#find the total distribution for each country in 2020,2021....2024
country_years_df = df.groupby("country_name")[["2020","2021","2022","2023","2024"]].sum()
print(country_years_df)
fig,axes = plt.subplots(2,3,figsize=(16,8))
axes[0,0].set_title("Total Distribution for each country from 2020 to 2024")
sns.scatterplot(x="country_name",y="2020",data=country_years_df,ax=axes[0,0])
sns.scatterplot(x="country_name",y="2021",data=country_years_df,ax=axes[0,1])
sns.scatterplot(x="country_name",y="2022",data=country_years_df,ax=axes[0,2])
sns.scatterplot(x="country_name",y="2023",data=country_years_df,ax=axes[1,0])
sns.scatterplot(x="country_name",y="2024",data=country_years_df,ax=axes[1,1])
plt.show()
#use a pair plot
sns.pairplot(country_years_df[["2020","2021","2022","2023","2024"]])
plt.show()
#Using barplot for 2024 find the distribution across sex and age group
sns.barplot(x="sex",y="2024",data=df,hue="age_group")
plt.title("Unemployment distribution for each sex and age group in 2024")
plt.show()
#scatterplot
sns.scatterplot(x="age_group",y="2024",data=df,hue="sex")
plt.title("Unemployment distribution for each sex and age group in 2024")
plt.show()
#General 2024 distribution based on age categories
MAC_df = df.groupby("age_categories")[["2024"]].mean()
print(MAC_df)
sns.barplot(data=df,x="age_categories",y="2024")
plt.show()
#General 2024 distribution  based on total for age categories
AC_df = df.groupby("age_categories")[["2024"]].sum()
print(AC_df)
sns.barplot(data=AC_df,x="age_categories",y="2024")
plt.show()
#General 2019-2024 distribution  based on means for age categories
GMAC_df= df.groupby("age_categories")[["2019","2020","2021","2022","2023","2024"]].mean()
print(GMAC_df)
fig,axes = plt.subplots(3,2,figsize=(16,8))
axes[0,0].set_title("2019 Mean distribtion based on age categories")
sns.barplot(data=df,x="age_categories",y="2019",ax=axes[0,0])
axes[0,1].set_title("2020 Mean distribtion based on age categories")
sns.barplot(data=df,x="age_categories",y="2020",ax=axes[0,1])
axes[1,0].set_title("2021 Mean distribtion based on age categories")
sns.barplot(data=df,x="age_categories",y="2021",ax=axes[1,0])
axes[1,1].set_title("2022 Mean distribtion based on age categories")
sns.barplot(data=df,x="age_categories",y="2022",ax=axes[1,1])
axes[2,0].set_title("2023 Mean distribtion based on age categories")
sns.barplot(data=df,x="age_categories",y="2023",ax=axes[2,0])
axes[2,1].set_title("2024 Mean distribtion based on age categories")
sns.barplot(data=df,x="age_categories",y="2024",ax=axes[2,1])
plt.show()

"""
fig,axes = plt.subplots(3,2,figsize=(16,8))
axes[0,0].set_title("2019 Mean distribtion based on country names")
sns.scatterplot(data=df,x="country_name",y="2019",ax=axes[0,0])
axes[0,1].set_title("2020 Mean distribtion based on country names")
sns.scatterplot(data=df,x="country_name",y="2020",ax=axes[0,1])
axes[1,0].set_title("2021 Mean distribtion based on country names")
sns.scatterplot(data=df,x="country_name",y="2021",ax=axes[1,0])
axes[1,1].set_title("2022 Mean distribtion based on country names")
sns.scatterplot(data=df,x="country_name",y="2022",ax=axes[1,1])
axes[2,0].set_title("2023 Mean distribtion based on country names")
sns.scatterplot(data=df,x="country_name",y="2023",ax=axes[2,0])
axes[2,1].set_title("2024 Mean distribtion based on country names")
sns.scatterplot(data=df,x="country_name",y="2024",ax=axes[2,1])
plt.show()"""

#USING DECISION TREE
import sklearn.tree as tree
from sklearn.tree import DecisionTreeClassifier
#DROP columns not needed
unemployment_df = unemployment_df.drop(['indicator_name','age_group'],axis=1)
#defining x and y

#CREATING ENCODERS
label = LabelEncoder()
unemployment_df['sex']= label.fit_transform(unemployment_df['sex'])
unemployment_df['age_categories']= label.fit_transform(unemployment_df['age_categories'])
print(unemployment_df) 
x=unemployment_df.iloc[:,1:]
y=unemployment_df.iloc[:,0]#OR
#unemployment_df['sex'] = (unemployment_df['sex']=='male').astype(int)
#unemployment_df['age_group'] = (unemployment_df['age_group']=='Under 15').astype(int)
#print(unemployment_df)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
print(x_train)
#DECISION TREE MODEL
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
dt_model = DecisionTreeClassifier(max_depth=3,random_state=150)
dt_model.fit(x_train,y_train)
y_pred = dt_model.predict(x_test)
print(y_pred)
print(dt_model.score(x_test,y_test))
print(classification_report(y_test,y_pred))
conf_mat=confusion_matrix(y_test,y_pred)
print(conf_mat)
plt.figure(figsize=(5,4))
sns.heatmap(conf_mat,cmap="coolwarm",fmt='g',annot=True)
plt.xlabel("Predicted labels")
plt.ylabel("Actual labels")
plt.title("Confusion matrix")
plt.show()
#USING RANDOMFOREST CLASSIFIER
from sklearn.ensemble import RandomForestClassifier
x=unemployment_df.iloc[:,1:]
y=unemployment_df.iloc[:,0]
imputer=SimpleImputer(strategy='mean')
x_impute = imputer.fit_transform(x)
x_train,x_test,y_train,y_test = train_test_split(x_impute,y,test_size=0.2,random_state=0)
rf_model = RandomForestClassifier(n_estimators=2000,max_depth=500,random_state=200)
rf_model.fit(x_train,y_train)
y_pred = rf_model.predict(x_test)
print(y_pred)
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test,y_pred))
conf_mat=confusion_matrix(y_test,y_pred)
print(conf_mat)
plt.figure(figsize=(5,4))
sns.heatmap(conf_mat,cmap="coolwarm",fmt='g',annot=True)
plt.xlabel("Predicted labels")
plt.ylabel("Actual labels")
plt.title("Confusion matrix")
plt.show()
#USING LOGISTICREGRESSION
from sklearn.linear_model import LogisticRegression
LR_model = LogisticRegression()
LR_model.fit(x_train,y_train)
y_pred=LR_model.predict(x_test)
print(LR_model.score(x_test,y_test))
print(accuracy_score(y_test,y_pred))
conf_mat=confusion_matrix(y_test,y_pred)
print(conf_mat)
plt.figure(figsize=(5,4))
sns.heatmap(conf_mat,cmap="coolwarm",fmt='g',annot=True)
plt.xlabel("Predicted labels")
plt.ylabel("Actual labels")
plt.title("Confusion matrix")
plt.show()
#USING HISTOGRADIENTBOOSTING CLASSIFIER
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
HGBC_model = HistGradientBoostingClassifier(max_iter=100,learning_rate=0.1,max_depth=6,random_state=42)
HGBC_model.fit(x_train,y_train)
y_pred=HGBC_model.predict(x_test)
print(accuracy_score(y_test,y_pred))
conf_mat=confusion_matrix(y_test,y_pred)
print(conf_mat)
plt.figure(figsize=(5,4))
sns.heatmap(conf_mat,cmap="coolwarm",fmt='g',annot=True)
plt.xlabel("Predicted labels")
plt.ylabel("Actual labels")
plt.title("Confusion matrix")
plt.show()
#DROPPING COLUMNS NOT NEEDED FOR LINEAR REGRESSION.this takes into accout 2024 predictions
unemployment_df=unemployment_df.drop(['country_name','sex','age_categories'],axis=1)
print(unemployment_df)
correlation_matrix=unemployment_df.corr()
sns.heatmap(correlation_matrix,annot=True,cmap='coolwarm',fmt='.2f')
plt.title("correlation matrix")
plt.show()
#CREATING X AND Y
x=unemployment_df.iloc[:,:-1]
print(x.shape)
y=unemployment_df.iloc[:,-1]
y=y.values
y = y.reshape(-1,1)
print(y.shape)
 #this gives a df with labels #OR the below tow gives a numpy array
#x=unemployment_df.values[:,1:]
#x=unemployment_df[unemployment_df.columns[1:]].values
#IMPUTING to account for missing values of x in DATASET
imputer = SimpleImputer(strategy='mean')
x_imputed= imputer.fit_transform(x)
y_imputed = imputer.fit_transform(y)

#SPLITTING DATASET
x_train,x_test,y_train,y_test = train_test_split(x_imputed,y_imputed,test_size=0.2,random_state=42)
print(x_imputed.shape)
print(y_imputed.shape) #OR ERROR below
"""split_part = np.split(unemployment_df.sample(frac=1),[int(0.6*len(unemployment_df)),int(0.8*len(unemployment_df))])
num_split_part = len(split_part)
if num_split_part >=3:
    train,valid,test = split_part
train,x_train,y_train = train.iloc[:,1:],train.iloc[:,0]
valid,x_valid,y_valid = valid.iloc[:,1:],valid.iloc[:,0]
test,x_test,y_test =test.iloc[:,1:],test.iloc[:,0]
print(x_train)"""

#LINEAR REGRESSION MODEL
linreg_model = LinearRegression()
linreg_model.fit(x_train,y_train)
y_pred = linreg_model.predict(x_test)
print(y_pred)
r2 =r2_score(y_test,y_pred)
print(f"the root square value is {r2}")
mse = mean_squared_error(y_test,y_pred)
print(f"the mean squared error is {mse}")

#USING RANDOM FOREST REGRESSOR
from sklearn.ensemble import RandomForestRegressor
RFR_model = RandomForestRegressor(n_estimators=1000,criterion="squared_error",max_depth=50,random_state=42)
RFR_model.fit(x_imputed,y_imputed)
y_pred=RFR_model.predict(x_imputed)
r2=r2_score(y_imputed,y_pred)
print(f"the r squred value is {r2}")
plt.scatter(y_test,y_pred)
plt.plot([min(y_test),max(y_test)],[min(y_test),max(y_test)],color="red")
plt.xlabel("Actual labels")
plt.ylabel("Predicted labels")
plt.title("True vs Predicted Values using R2 square")
plt.show()

#USING DECISION TREE REGRESSOR
from sklearn.tree import DecisionTreeRegressor
DTR_model = DecisionTreeRegressor(criterion="squared_error",max_depth=50,random_state=42)
DTR_model.fit(x_imputed,y_imputed)
y_pred=DTR_model.predict(x_imputed)
r2=r2_score(y_imputed,y_pred)
print("the r squred value is {}".format(r2))
plt.scatter(y_test,y_pred)
plt.plot([min(y_test),max(y_test)],[min(y_test),max(y_test)],color="red")
plt.xlabel("Actual labels")
plt.ylabel("Predicted labels")
plt.title("True vs Predicted Values using R2 square")
plt.show()

#USING HISTOGRADIENTBOOSTER REGRESSOR
from sklearn.ensemble import HistGradientBoostingRegressor
HGBR_model = HistGradientBoostingRegressor(max_iter=100,learning_rate=0.1,max_depth=6,random_state=42)
HGBR_model.fit(x_train,y_train)
y_pred=HGBR_model.predict(x_test)
print(mean_squared_error(y_test,y_pred))
print(r2_score(y_test,y_pred))
plt.scatter(y_test,y_pred)
plt.plot([min(y_test),max(y_test)],[min(y_test),max(y_test)],color="red")
plt.xlabel("Actual labels")
plt.ylabel("Predicted labels")
plt.title("True vs Predicted Values using R2 square")
plt.show()

#TRY TENSORFLOW
import tensorflow as tf
nn_model = tf.keras.Sequential([
    tf.keras.layers.Dense(64,activation = 'relu'),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(64,activation='relu'),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(1,activation= 'sigmoid')
  ])
nn_model.compile(optimizer=tf.keras.optimizers.Adam(0.01),loss ='mean_squared_error')
history=nn_model.fit(x_train,y_train,epochs= 200,batch_size=32,validation_split=0.2,verbose=0)
y_pred=nn_model.predict(x_test)
r2=r2_score(y_test,y_pred)
print(r2)
plt.scatter(y_test,y_pred)
plt.plot([min(y_test),max(y_test)],[min(y_test),max(y_test)],color="red")
plt.xlabel("Actual labels")
plt.ylabel("Predicted labels")
plt.title("True vs Predicted Values using R2 square")
MSE=mean_squared_error(y_test,y_pred)
print(MSE)
plt.scatter(y_test,y_pred)
plt.plot([min(y_test),max(y_test)],[min(y_test),max(y_test)],color="red")
plt.xlabel("Actual labels")
plt.ylabel("Predicted labels")
plt.title("True vs Predicted Values using MSE")

plt.figure(figsize=(10,6))
sns.set_style("whitegrid")
plt.plot(history.history['loss'],label="Training loss")
plt.plot(history.history['val_loss'],label="validation loss")
plt.title("Training vs Validation loss")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.show()

#TENSOR FLOW WITH A FUNCTION
def train_model(x_train,y_train,dropout_prob,num_nodes,lr,epochs,batch_size):
  nn_model = tf.keras.Sequential([
      tf.keras.layers.Dense(num_nodes,activation = 'relu'),
      tf.keras.layers.Dropout(dropout_prob),
      tf.keras.layers.Dense(num_nodes,activation='relu'),
      tf.keras.layers.Dropout(dropout_prob),
      tf.keras.layers.Dense(1,activation= 'sigmoid')
  ])
  nn_model.compile(optimizer=tf.keras.optimizers.Adam(lr),loss ='mean_squared_error')

  history = nn_model.fit(x_train,y_train,epochs= epochs,batch_size=batch_size,validation_split=0.2,verbose=0)

  return nn_model,history
for num_nodes in [64,128,256]:
  for dropout_prob in [0,0.2,0.3]:
    for lr in [0.01,0.005,0.001]:
      for batch_size in [64,128,256]:
        for epochs in [100,500,1000]:
         print(f"{num_nodes} nodes,dropout{dropout_prob},lr{lr},batch_size{batch_size},epoch{epochs}")
         model,history = train_model(x_train,y_train,dropout_prob,num_nodes,lr,epochs,batch_size)
         r2=r2_score(y_test,y_pred)
         print(f"R squared ",r2)
         MSE=mean_squared_error(y_test,y_pred)
         print("Mean squared value is {}".format(MSE))
plt.figure(figsize=(10,6))
sns.set_style("whitegrid")
plt.plot(history.history['loss'],label="Training loss")
plt.plot(history.history['val_loss'],label="validation loss")
plt.title("Training vs Validation loss")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.show()