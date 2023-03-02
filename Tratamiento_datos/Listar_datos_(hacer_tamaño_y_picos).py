import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import skimage.transform as skTrans
from pathlib import Path
import _pickle as pickle

def ls(ruta = Path.cwd()):
    return [arch.name for arch in Path(ruta).iterdir() if arch.is_file()]

#ruta="F:/ADNI1_Screening/"         #La direccion de cada carpeta donde estan los datos
# ruta="F:/ADNI1_Complete_1Yr_1.5T/"
#ruta="F:/ADNI1_Complete_2Yr_1.5T/"
#ruta="F:/ADNI1_Complete_3Yr_1.5T/" 
rutass=ls(ruta)
rutas = rutass[:-1] #se le resta por si tiene codigos o cosas que no son de archivos extra
List = []
elemtos = []
for j in rutas:
    elemtos.append(j)   #para verificar por donde paso
    im = nib.load(ruta + j)  #con esto recorre la carpeta y carga cada imagen 
    img = im.get_fdata()
    data = skTrans.resize(img, (200,200,160), order=1, preserve_range=True) #todas las imagenes quedan de 200x200 y serian 160
    DataPickle=[]
    for i in range(40,120):
        z_slice = data[:,:,i]  #con esto recorre las imagenes que selecciones por mayor informacion 
        dataReshape = np.reshape(z_slice,(1,40000)) #convierto la matriz de la imganes en un vector para que esta informacion en un futuro se maneje con mayor facilidad
        List.append(dataReshape) #las guardo con los arreglos anteriores
DataPickle.append(List) #ESTE SE PUEDE OMITIR Y IR DIRECTAMENTE CON LIST
pickle.dump(DataPickle, open('./PickleData/data.plk', 'wb')) #guardo las imagenes adecuadas y seleccionadas
pickle.dump(rutas, open('./PickleData/rutas.plk', 'wb')) #gurardo la ruta para en un futuro juntarla en un excel con cada fila de imagen y mantener un orden


with open(f'array1_{j}.csv', 'w', newline='', encoding='utf-8') as csvfile:
      writer = csv.writer(csvfile)     
      writer.writerows(array3)