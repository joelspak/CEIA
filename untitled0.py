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

print("1. Importando librerias.")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import scipy.stats as stats

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

####### Se dropean las columnas Evaporation, Sunshine, Cloud3pm, Cloud9pm ya que hay ausencias 100% en muchas ciudades.
###### pese a que parecen datos muy importantes.
treshold=70
for column in df.columns:
    if len(df_nans_per_city[df_nans_per_city[column]>treshold]):
        df=df.drop(column, axis=1)

#### una vez dropeadas las columnas con muchos faltantes, procedo a realizar algo para despejar el resto de los Nans.
#### Los NaNs de Rainfall los dropeo, esto es porque es la variable de salida y no tiene sentido imputarla.

#df=df.dropna()
