import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statistics
from math import *
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler

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



