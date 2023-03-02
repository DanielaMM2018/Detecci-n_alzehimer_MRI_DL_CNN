# PARA LAS IMAGENES
import numpy as np
import _pickle as pickle
import pandas as pd
import csv
def pickload(ruta):
    with open (ruta,"rb") as file:
        return pickle.load(file)

# array=pickload("E:/data1.plk")

# array1=pickload("E:/ADNI1_Complete_3Yr_1.5T/PickleData/data1.plk")
# array1=array1[0]

# array2=pickload("E:/ADNI1_Complete_3Yr_1.5T/PickleData/data2.plk")
# array2=array2[0]

# array3=pickload("E:/ADNI1_Complete_3Yr_1.5T/PickleData/data3.plk")
# array3=array3[0]# array = []

# array4=pickload("E:/ADNI1_Complete_3Yr_1.5T/PickleData/data4.plk")
# array4=array4[0]

# array5=pickload("E:/ADNI1_Complete_3Yr_1.5T/PickleData/data5.plk")
# array5=array5[0]

# array=pickload("E:/ADNI1_Complete_1Yr_1.5T/PickleData/data5.plk")
# array5=array[0]


# l = 16000
# r = len(array2)

# array = [] 
# for i in range(l,r):
#     a = array2[i]
#     a = a.transpose()
#     array.append(a)


# for i in range(29000):
#     a = array3[i]
#     a = a.transpose()
#     array.append(a)
    
# pickle.dump(array, open('E:/PickleData/data2.plk', 'wb'))

# l = 26000
# r = len(array3)

# array = [] 
# for i in range(l,r):
#     a = array3[i]
#     a = a.transpose()
#     array.append(a)


# for i in range(33500):
#     a = array5[i]
#     a = a.transpose()
#     array.append(a)
    
# pickle.dump(array, open('E:/PickleData/data4.plk', 'wb'))
# pickle.dump(array, open('E:/PickleData/data2.plk', 'wb'))

# df = pd.DataFrame()
# for i in range(500):
#     a = array1[i]
#     a = a.transpose()
#     array.append(a)





# array1.append(array2)
# # DataPickle.append(array3)

# pickle.dump(array, open('E:/PickleData/data3.plk', 'wb'))

# array = []
# df = pd.DataFrame()
# for i in range(500):
#     a = array1[i]
#     a = a.transpose()
#     array.append(a)


    
# b = []
# for i in range(500):
#     a = np.asarray(array[i])
#     a_l = a.tolist()
#     a_l.insert(0,f'Numero_i')
#     a = np.asarray(a_l)
#     b.append(a)

#d = np.insert(a, 0, c)
   
# array=pickload("E:/ADNI1_Complete_1Yr_1.5T/PickleData/data5.plk")
# array1=array[0] 

# array=pickload("E:/ADNI1_Complete_2Yr_1.5T/PickleData/data5.plk")
# array2=array[0]

# array=pickload("E:/ADNI1_Complete_3Yr_1.5T/PickleData/data5.plk")
# array5=array[0]

# l = 0
# r = 1600
# for j in range(22):
#     with open(f'array1_{j}.csv', 'w', newline='', encoding='utf-8') as csvfile:
#         writer = csv.writer(csvfile)
#         for i in range(l,r):
#             array = array1[i]
#             array = array.transpose()
#             writer.writerows(array)
#     l+= 1600 
#     r+= 1600


# l = 0
# r = 1600
# for j in range(23):
#     with open(f'array2_{j}.csv', 'w', newline='', encoding='utf-8') as csvfile:
#         writer = csv.writer(csvfile)
#         for i in range(l,r):
#             array = array2[i]
#             array = array.transpose()
#             writer.writerows(array)
#     l+= 1600 
#     r+= 1600


# l = 0
# r = 1600
# for j in range(22):
#     with open(f'array3_{j}.csv', 'w', newline='', encoding='utf-8') as csvfile:
#         writer = csv.writer(csvfile)
#         for i in range(l,r):
#             array = array3[i]
#             array = array.transpose()
#             writer.writerows(array)
#     l+= 1600 
#     r+= 1600


# l = 0
# r = 1600
# for j in range(25):
#     with open(f'array5_{j}.csv', 'w', newline='', encoding='utf-8') as csvfile:
#         writer = csv.writer(csvfile)
#         for i in range(l,r):
#             array = array5[i]
#             array = array.transpose()
#             writer.writerows(array)
#     l+= 1600 
#     r+= 1600



# m = 0
# n = 1600
# for p in range(23):
#     with open(f'array5_{p}.csv', 'w', newline='', encoding='utf-8') as csvfile:
#         writer = csv.writer(csvfile)
#         for k in range(m,n):
#             writer.writerows(array5[k])
#     m+= 1600 
#     n+= 1600





# Los que quedan sobrando

# with open('array3_20.csv', 'w', newline='', encoding='utf-8') as csvfile:
#     writer = csv.writer(csvfile)
#     for i in range(32000,32640):
#         writer.writerows(array3[i])













# PARA LAS RUTAS





# ruta1=pickload("F:/ADNI1_Screening/PickleData/rutas1.plk")

# ruta2=pickload("F:/ADNI1_Screening/PickleData/rutas2.plk")

# ruta3=pickload("F:/ADNI1_Screening/PickleData/rutas3.plk")

# ruta4=pickload("F:/ADNI1_Complete_1Yr_1.5T/PickleData/rutas4.plk")

# ruta5=pickload("F:/ADNI1_Screening/PickleData/rutas5.plk")



# import openpyxl


# wb = openpyxl.Workbook()
# hoja = wb.active
# hoja.append(ruta1)
# wb.save('ruta1.xlsx')


# wb = openpyxl.Workbook()
# hoja = wb.active
# hoja.append(ruta2)
# wb.save('ruta2.xlsx')

# wb = openpyxl.Workbook()
# hoja = wb.active
# hoja.append(ruta3)
# wb.save('ruta3.xlsx')

# wb = openpyxl.Workbook()
# hoja = wb.active
# hoja.append(ruta4)
# wb.save('ruta4.xlsx')

# wb = openpyxl.Workbook()
# hoja = wb.active
# hoja.append(ruta5)
# wb.save('ruta5.xlsx')


