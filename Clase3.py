# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 19:24:21 2021

@author: jspak
"""
import numpy as np
########## ej_1 normalizacion

def ej_1(A):
    return (A - np.mean(A))/np.std(A)


A = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
# print(ej_1(A))


def ej_2(A):
    nans_row = np.where(np.isnan(A))[0]
    nans_col = np.where(np.isnan(A))[1]
    A = np.delete(A, nans_row,axis=0)
    A = np.delete(A, nans_col,axis=1)
    return A
                               

B=np.array([[1,2,3],[4,np.nan,6],[7,8,9]])
# print(B)
print(ej_2(B))


########## ej_3 Dado un dataset, hacer una función que utilizando numpy reemplace los NaNs por la media de la columna.
def ej_3(A):
    print(A)
    nans_positions = np.where(np.isnan(A))
    print(nans_positions)
    for nan in nans_positions:
        print(nan)
        b = np.delete(A, nans_positions[0],axis=0)
        b = b[:,nan[1]]
        print(b)
        A[nan] == np.mean(b, axis=1)
    return A

print(ej_3(B))


########## ej 4 - train/test
def ej_4(a, porcentaje_train, porcentaje_validation):
    train = int(porcentaje_train * a.shape[0] / 100)
    validation = int((porcentaje_validation + porcentaje_train) * a.shape[0] / 100)
    return a[0:train], a[train:validation], a[validation:]

# print(ej_4(A, 70, 20))

