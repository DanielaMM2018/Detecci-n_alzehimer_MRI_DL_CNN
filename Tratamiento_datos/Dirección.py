import numpy as np
import pandas as pd
import csv
import xlrd



df= pd.read_csv('ADNI1_Screening_1.csv', sep = ';')
df.describe()
l = len(df['Image Data ID'])

dir = []

for i in range (l):
        direccion = 'ADNI1_Screening\\' + str(df['Subject'][i]) + '_' + str(df['Image Data ID'][i]) + '.nii'
        dir.append(direccion)
        
df['dir'] = dir 
df.to_csv('ADNI1_Screening.csv', sep=';')