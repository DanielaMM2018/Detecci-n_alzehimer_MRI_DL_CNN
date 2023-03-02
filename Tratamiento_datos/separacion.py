# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 09:09:44 2022

@author: CAP-01
"""
import numpy as np
import _pickle as pickle
import pandas as pd
import csv
def pickload(ruta):
    with open (ruta,"rb") as file:
        return pickle.load(file)

#     #importa las imagenes
array1 = pickload("E:/PickleData/data_CN_1.plk")
array2 = pickload("E:/PickleData/data_CN_2.plk")
array3 = pickload("E:/PickleData/data_AD_1.plk")

    #importar las direcciones para la futura distribuccion

array7 = pickload("E:/PickleData/diaaaaags_CN.plk")
array8 = pickload("E:/PickleData/diaaaags_AD.plk")

# #     #crear los arrays de los diagnoticos para la distribuccion a futuro

array = []
for i in range(len(array1)):
    array.append(array1[i])
    
for i in range(len(array2)):
    array.append(array2[i])
  
  
array1 = []
for i in range(59840):
    array1.append(0)
    
    
array2 = []
for i in range(56320):
    array2.append(1)
    
    
    
#     # separar los datos de manera odenada para acceder a ellos  (hasta 56320 porque el AD tiene ese numero de datos)

array4 = []
array5 = []
array6 = []  

for i in range(56320):
    a = array1[i]    # los unos del diag de CN
    b = array2[i]   # los ceros del diag de AD
    c = array[i]   # los datos de imagenes del diag de CN
    d = array3[i]   # los datos de imagenes del diag de AD
    e = array7[i]   # Las direcciones de los diag de CN
    f = array8[i]   # Las direcciones de los diag de AD
    array4.append(f)   # AD
    array4.append(e)   # CN
    array5.append(d)   #imagenes AD
    array5.append(c)   #imagenes CN
    array6.append(b)   #diag AD
    array6.append(a)   #diag CN

#      # separar los datos restantes que faltaron de CN

for i in range(56320,59840):
    b = array1[i]   # los ceros del diag de CN
    d = array[i]   # los datos de imagenes del diag de CN
    f = array7[i]   # Las direcciones de los diag de CN
    array4.append(f)
    array5.append(d)
    array6.append(b)



# #     # separar los datos a evaluar (para poder importar los picos) IMAGENES

array9 = []
array10 = []
array11 = []

for i in range(46464):
    array9.append(array5[i])
    
for i in range(46464,92928):
    array10.append(array5[i])

for i in range(92928,116160):  
    array11.append(array5[i])

    
    
#     # separar los datos a evaluar (para poder importar los picos) DIAG

array12 = []
array13 = []
array14 = []
for i in range(46464): 
    array12.append(array6[i])
    
for i in range(46464,92928):
    array13.append(array6[i])

for i in range(92928,116160):
    array14.append(array6[i])


#     # separar los datos a evaluar (para poder importar los picos) DIRECCION

array15 = []
array16 = []
array17 = []
for i in range(46464): 
    array15.append(array4[i])
    
for i in range(46464,92928):
    array16.append(array4[i])
    
for i in range(92928,116160):
    array17.append(array4[i])
    
    

# pickle.dump(array9, open('E:/PickleData/data_igual1.plk', 'wb'))
# pickle.dump(array10, open('E:/PickleData/data_igual2.plk', 'wb'))
# pickle.dump(array11, open('E:/PickleData/data_desigual.plk', 'wb'))
pickle.dump(array12, open('E:/PickleData/diag_igual1.plk', 'wb'))
pickle.dump(array13, open('E:/PickleData/diag_igual2.plk', 'wb'))
pickle.dump(array14, open('E:/PickleData/diag_desigual.plk', 'wb'))

pickle.dump(array15, open('G:/PickleData/dir_igual1.plk', 'wb'))
pickle.dump(array16, open('G:/PickleData/dir_igual2.plk', 'wb'))
pickle.dump(array17, open('G:/PickleData/dir_desigual.plk', 'wb'))



# pickle.dump(array13, open('G:/PickleData/diag2.plk', 'wb'))
# pickle.dump(array18, open('G:/PickleData/diag3.plk', 'wb'))
# pickle.dump(array17, open('G:/PickleData/dir3.plk', 'wb'))