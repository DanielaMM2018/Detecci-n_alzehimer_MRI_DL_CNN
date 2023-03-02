
import numpy as np
import pandas as pd
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import decomposition
from sklearn.decomposition import PCA
# from sklearn import svm
import tensorflow as tf
# from sklearn.metrics import matthews_corrcoef
from sklearn.neighbors import KNeighborsClassifier
import _pickle as pickle
import csv
def pickload(ruta):
    with open (ruta,"rb") as file:
        return pickle.load(file)
    
    
#     #importar los datos
array1 = pickload("E:/PickleData/primero.plk")
# array2 = pickload("E:/PickleData/segundo.plk")
# array1 = pickload("E:/PickleData/trece.plk")

    #importar los diagnosticos
array3 = pickload("E:/PickleData/diag.plk")
# array4 = pickload("E:/PickleData/diag2.plk")


#     #importar las direcciones 
# array5 = pickload("E:/PickleData/dir.plk")
# array6 = pickload("E:/PickleData/dir2.plk")


array1 = array1.transpose()
array3 = pd.DataFrame(array3)
# array2 = array2.transpose()
# array4 = pd.DataFrame(array4)

X = array1[:]
y = array3[:]
# X2 = array2[:]
# y2 = array4[:]


X_train, X_test = train_test_split(X, test_size=0.2, random_state=None)
y_train, y_test = train_test_split(y, test_size=0.2, random_state=None)

# # y2_train, y2_test = train_test_split(y2, test_size=0.2, random_state=None)
# # X2_train, X2_test = train_test_split(X2, test_size=0.2, random_state=None)


#     # Normalización
# scaler = StandardScaler()
# scaler.fit(X_train)
# X_train = scaler.transform(X_train)
# X_test = scaler.transform(X_test)

# # # scaler = StandardScaler()
# # # scaler.fit(X2_train)
# # # X2_train = scaler.transform(X2_train)
# # # X2_test = scaler.transform(X2_test)



pca = decomposition.PCA(n_components=127)


pca.fit(X_train)

PCA_X_train = pca.transform(X_train)
PCA_X_test = pca.transform(X_test)
PCA_X = pca.transform(X)



def build_clf(unit):
# creating the layers of the NN
  ann = tf.keras.models.Sequential()
  ann.add(tf.keras.layers.Dense(units=unit, activation='relu'))
  ann.add(tf.keras.layers.Dense(units=unit, activation='relu'))
  ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
  ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
  return ann



model=KerasClassifier(build_fn=build_clf)



params={'batch_size':[100, 20, 50, 25, 32],
		'nb_epoch':[200, 100, 300, 400],
		'unit':[200, 300, 400, 500, 700, 1000],
		}
gs=GridSearchCV(estimator=model, param_grid=params, cv=10)
# now fit the dataset to the GridSearchCV object.



gs = gs.fit(PCA_X_train, y_train)



# #Accuracy  #PONER F1
# print("Clasificador: " + str(gs.best_estimator_) + " usando: " + str(gs.best_params_) + " - Accuracy: " + str(gs.best_score_))




# from keras import backend as K

# def recall_m(y_true, y_pred):
#     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#     possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
#     recall = true_positives / (possible_positives + K.epsilon())
#     return recall

# def precision_m(y_true, y_pred):
#     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#     predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
#     precision = true_positives / (predicted_positives + K.epsilon())
#     return precision

# def f1_m(y_true, y_pred):
#     precision = precision_m(y_true, y_pred)
#     recall = recall_m(y_true, y_pred)
#     return 2*((precision*recall)/(precision+recall+K.epsilon()))

# # compile the model
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc',f1_m,precision_m, recall_m])

# # fit the model
# history = model.fit(X_train, y_train, validation_split=0.3, epochs=10, verbose=0)

# # evaluate the model
# loss, accuracy, f1_score, precision, recall = model.evaluate(X_test, y_test, verbose=0)


































































# import tensorflow.keras as kr

# model = kr.Sequential()

# # Añadimos la capa 1
# l1 = model.add(kr.layers.Dense(1000, activation='relu'))

# # Añadimos la capa 2
# l2 = model.add(kr.layers.Dense(1000, activation='relu'))

# # Añadimos la capa 3
# l3 = model.add(kr.layers.Dense(1000, activation='sigmoid'))

# # Compilamos el modelo, definiendo la función de coste y el optimizador.
# model.compile(loss='mse', optimizer=kr.optimizers.SGD(lr=0.05), metrics=['acc'])

# # Y entrenamos al modelo. Los callbacks 
# model.fit(PCA_X_train, y_train, epochs=400,batch_size= 20)






