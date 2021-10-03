# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 19:25:16 2021

@author: jspak


"""
def plot(x, city):
    fig, ax = plt.subplots(1,1,figsize=(20,20))
    ax.plot(x, label=city)
    ax.legend(loc='best', shadow=True, fancybox=True, fontsize=25, framealpha=0.5)
    plt.show


def display_dataset_distributions(dataset, city):
    fig = dataset.hist(xlabelsize=12, ylabelsize=12,figsize=(22,10))
    [x.title.set_size(14) for x in fig.ravel()]
    plt.tight_layout()
    plt.show()
    
def impute_column(df, col_to_predict, feature_columns):
    """ Imputar valores faltantes de una columna a partir de un 
        modelo LR sobre columnas restantes
    """
    nan_rows = np.where(np.isnan(df[col_to_predict]))
    print(nan_rows)
    all_rows = np.arange(0,len(df))
    train_rows_idx = np.argwhere(~np.isin(all_rows,nan_rows)).ravel()
    pred_rows_idx =  np.argwhere(np.isin(all_rows,nan_rows)).ravel()
    
    X_train,y_train = df[feature_columns].iloc[train_rows_idx],df[col_to_predict].iloc[train_rows_idx]
    X_pred = df[feature_columns].iloc[pred_rows_idx]
      
    model = LinearRegression()
    model.fit(X_train,y_train)
    df[col_to_predict].iloc[pred_rows_idx] = model.predict(X_pred.values.reshape(1,-1))
    return df

def col_to_impute(df_1, df_missing, col):
    tmp_feature_cols = [x for x in feature_cols if x != col] 
    df_1.loc[df_missing[col].isna(),col] = np.nan
    df_1 = impute_column(df_1, col, tmp_feature_cols)
    return df_1

def Wind_Dir(df, dir):
    Wind_dir=pd.get_dummies(df[dir])
    wind_directions = [column for column in Wind_dir]
    north=dir+"_N"
    south=dir+"_S"
    west=dir+"_W"
    east=dir+"_E"
    df.insert(len(df.columns), north, 0)
    df.insert(len(df.columns), south, 0)
    df.insert(len(df.columns), west, 0)
    df.insert(len(df.columns), east, 0)
    for direction in wind_directions:
        if "N" in direction:
            df[north]+=Wind_dir[direction]
        if "S" in direction:
            df[south]+=Wind_dir[direction]
        if "W" in direction:
            df[west]+=Wind_dir[direction]
        if "E" in direction:
            df[east]+=Wind_dir[direction]
    
    df=df.drop(dir, axis=1)
    return df

def promedio(df, v1, v2, nombre):
    df_prom=df[v1, v2]
    df[nombre] = df_prom
    df=df.drop([v1, v2], axis=1)
    return df


print("1. Importando librerias.")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns; sns.set()
import scipy.stats as stats
import datetime as dt

# Entrenamiento de modelos de prueba
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Evaluación de modelos de prueba
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error

# Partición de train-test set
from sklearn.model_selection import train_test_split

# Pipelines
from sklearn.pipeline import Pipeline

# Crear datasets
from sklearn.datasets import make_regression

# Esquemas de entrenamiento
from sklearn.model_selection import RepeatedStratifiedKFold, RepeatedKFold
from sklearn.model_selection import cross_val_score

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import geopy as gps

####### ver de usar geopy.geocoders

print("2. Carga datos de dataset")
path = r"C:/Especializacion IA/2do_bim/tp/weatherAUS.csv"
df = pd.read_csv(path, sep=',',engine='python')
print(df.head())
print("3. Análisis de datos")
total_samples = len(df)
cities = pd.unique(df.Location)
total_cities = len(cities)

# for city in cities:
#     mintemp=df[df.Location==city]["Rainfall"]
#     plot(mintemp, city)

df_nans_per_city = pd.DataFrame(index=cities, columns=df.columns)
for city in cities:
    # Filas con y nulo
    df_city = df[df.Location==city]
    for column in df_nans_per_city.columns:
        nans = len(df_city[df_city[column].isna()])
        df_nans_per_city.loc[city][column]=int(100*nans/len(df_city))

# df = df[df['Outcome'].notna()]
# df.isnull().sum()

print("3. One Hot Encoding")
df['Rainfall_tomorrow']=df['Rainfall'].shift(-1)
df['Rainfall_yesterday']=df['Rainfall'].shift(1)
max_date = df["Date"].max()
min_date = df["Date"].min()
delete = np.where(df.Date==max_date)
### dropeo el primer dia y el ultimo del dataset porque no tienen informacion sobre la lluvia de ayer y de mañana respectivamente.
df = df.drop(df[(df['Date']==max_date)].index)
df = df.drop(df[(df['Date']==min_date)].index)

#df["TempMedia"]=(df["MaxTemp"]+df["MinTemp"])/2
df["Month"]=pd.to_datetime(df["Date"]).dt.month

#df=df.drop(["MaxTemp", "MinTemp", "Temp9am", "Temp3pm"], axis=1)
df=df.replace(to_replace="Yes", value=1)
df=df.replace(to_replace="No", value=0)

# variables_categoricas=["Month","WindGustDir", "WindDir9am", "WindDir3pm"]
# for variable in variables_categoricas:
#     print(df.loc[:, variable].value_counts())
  
# fig,axes = plt.subplots(1,1,figsize=(20,4))
# sns.regplot(x="Month", y="Rainfall_tomorrow", data=df, order=1,ax=axes)

##### de los datos de lluvia por mes se puede ver para agrupar en: verano meses 12, 1, 2, otoño 3,4,5,6  invierno 7,8,9 , primavera 10,11

Month = pd.get_dummies(df["Month"])
df["Verano"]=Month[12]+Month[1]+Month[2]
df["Otoño"]=Month[3]+Month[4]+Month[5]+Month[6]
df["Invierno"]=Month[7]+Month[8]+Month[9]
df["Primavera"]=Month[10]+Month[11]
df=df.drop("Month", axis=1)

df = Wind_Dir(df, "WindGustDir")
df = Wind_Dir(df, "WindDir9am")
df = Wind_Dir(df, "WindDir3pm")

location = pd.get_dummies(df["Location"], prefix="city")
df= pd.concat([df, location], axis=1, join="inner")
df= df.drop("Location", axis=1)

df = df.sort_values("Date", ascending=True)
df_1 = df.drop(["RainTomorrow", "Rainfall_tomorrow", "Date"], axis=1)
y=df["RainTomorrow"]
X_train, X_test, y_train, y_test = train_test_split(df_1, y, test_size=0.1, random_state=1)
print(X_train.shape, X_test.shape)


feature_cols=["MaxTemp","MinTemp", "Temp9am", "Temp3pm", "Pressure9am", "Pressure3pm", "Cloud9am", "Cloud3pm", "Humidity9am", "Humidity3pm"]
df_2=X_train[feature_cols]
corr = df_2.corr()
print(corr.shape)
sns.heatmap(corr, cmap=sns.diverging_palette(220,10,as_cmap=True),annot=True,fmt=".2f")
sns.set(font_scale=0.9)

##### sólo usamos una temperatura, una presión, un nublado y una humedad. 

fig, ax = plt.subplots(2, 2, figsize=(25, 25))
sns.boxplot(x='RainTomorrow', y='MaxTemp', data=df, ax=ax[0,0])
sns.boxplot(x='RainTomorrow', y='MinTemp', data=df, ax=ax[0,1])
sns.boxplot(x='RainTomorrow', y='Temp9am', data=df, ax=ax[1,0])
sns.boxplot(x='RainTomorrow', y='Temp3pm', data=df, ax=ax[1,1])

## No hay una variable que parezca mejor que la otra para predecir.
fig, ax = plt.subplots(3, 2, figsize=(25, 25))
sns.boxplot(x='RainTomorrow', y='Pressure9am', data=df, ax=ax[0,0])
sns.boxplot(x='RainTomorrow', y='Pressure3pm', data=df, ax=ax[0,1])
sns.boxplot(x='RainTomorrow', y='Cloud9am', data=df, ax=ax[1,0])
sns.boxplot(x='RainTomorrow', y='Cloud3pm', data=df, ax=ax[1,1])
sns.boxplot(x='RainTomorrow', y='Humidity9am', data=df, ax=ax[2,0])
sns.boxplot(x='RainTomorrow', y='Humidity3pm', data=df, ax=ax[2,1])

### La humedad de las 3pm parece mejor. El resto es similar.

df = promedio(df, "Pressure9am", "Pressure3pm", "Pressure")
df = promedio(df, "Cloud9am", "Cloud3pm", "Cloud")
df = promedio(df, "Humidity9am", "Humidity3pm", "Humidity")


####### Se dropean las columnas Evaporation, Sunshine, Cloud3pm, Cloud9pm ya que hay ausencias 100% en muchas ciudades.
###### pese a que parecen datos muy importantes.
# treshold=70
# for column in df.columns:
#     if len(df_nans_per_city[df_nans_per_city[column]>treshold]):
#         df=df.drop(column, axis=1)

#### una vez dropeadas las columnas con muchos faltantes, procedo a realizar algo para despejar el resto de los Nans.
#### Los NaNs de Rainfall los dropeo, esto es porque es la variable de salida y no tiene sentido imputarla.

#df=df.dropna()



###### Imputacion MICE
# print("Imputación MICE")
# feature_cols=["Pressure9am", "Pressure3pm", "Cloud9am", "Cloud3pm", "Humidity9am", "Humidity3pm"]
# target_col="Rainfall"
# df_0 = df[feature_cols].copy()
# for column in feature_cols:
#     df_0.loc[df_0[column].isna(), column] = df_0[column].mean()

# df_1=df_0.copy()
# # for column in feature_cols:
# #     df_1=col_to_impute(df_1, df, column)


# col_to_impute = 'Pressure9am'
# tmp_feature_cols = [x for x in feature_cols if x != col_to_impute] 
# df_1.loc[df[col_to_impute].isna(),col_to_impute] = np.nan
# df_1 = impute_column(df_1, col_to_impute, tmp_feature_cols)







# for city in cities:
#     df_city=df[df.Location==city]
#     display_dataset_distributions(df_city, city)
# df["TempMedia"]=(df["MaxTemp"]+df["MinTemp"])/2
# df["CloudMedia"]=(df["Cloud9am"]+df["Cloud3pm"])/2

# city="SydneyAirport"
# df_city=df[df.Location==city]
# display_dataset_distributions(df_city, city)
# fig,axes = plt.subplots(1,6,figsize=(20,4))
# sns.regplot(x="TempMedia", y="Rainfall", data=df_city, order=1,ax=axes[0])
# sns.regplot(x="MaxTemp", y="Rainfall", data=df_city, order=1,ax=axes[1])
# sns.regplot(x="WindSpeed3pm", y="Rainfall", data=df_city, order=1,ax=axes[2])
# sns.regplot(x="Humidity3pm", y="Rainfall", data=df_city, order=1,ax=axes[3])
# sns.regplot(x="Cloud3pm", y="Rainfall", data=df_city, order=1,ax=axes[4])

# fig,axes=plt.subplots(1,4,figsize=(22,6))
# stats.probplot(df_city['TempMedia'], dist="norm", plot=axes[0])
# axes[0].set_title("TempMedia")

# stats.probplot(df_city['Cloud3pm'], dist="norm", plot=axes[1])
# axes[1].set_title("Cloud3pm")

# stats.probplot(df_city['Humidity3pm'], dist="norm", plot=axes[2])
# axes[2].set_title("Humidity3pm")

# stats.probplot(df_city['WindSpeed9am'], dist="norm", plot=axes[3]);
# axes[3].set_title("log(CRIM)")