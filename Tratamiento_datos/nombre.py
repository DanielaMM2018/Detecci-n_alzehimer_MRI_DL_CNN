# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 13:20:29 2022

@author: CAP-01
"""
import os 

contenido = os.listdir('F:\\ADNI1_Complete_3Yr_1.5T')



for i in range(len(contenido)):
    l = len(contenido[i])
    x = l - 11
    Frist_name = contenido[i][5:16]
    Last_name = contenido[i][x:l]
    
    if Last_name[0] == '_':
        Last_name = contenido[i][x+1:l]
    
    NewName = Frist_name+Last_name
    
    archivo = contenido[i]
    nombre_nuevo = NewName
    os.rename(archivo, nombre_nuevo)

print(NewName)
