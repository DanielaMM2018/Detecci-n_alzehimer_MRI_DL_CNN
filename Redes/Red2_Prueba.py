import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pickle
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, roc_auc_score, cohen_kappa_score
import matplotlib.pyplot as plt
import itertools



# Función para cargar los datos
def pickload(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

# Carga los datos
X_train = np.array(pickload("D:/PickleData/data_igual1.plk"))
X_test = np.array(pickload("D:/PickleData/data_peq_igual.plk"))

y_train = pickload("D:/PickleData/diag_igual1.plk")
y_test = pickload("D:/PickleData/diag_peq_igual.plk")


print("ya")


# Transformar las formas de las entradas
X_train = X_train.reshape(X_train.shape[0], 200, 200, 1)
X_test = X_test.reshape(X_test.shape[0], 200, 200, 1)


# División de los datos de prueba en validación y test
validation_split = 0.2
validation_size = int(len(X_test) * validation_split)

X_val = X_test[:validation_size]
y_val = y_test[:validation_size]

X_test = X_test[validation_size:]
y_test = y_test[validation_size:]

# Data preprocessing
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)

validation_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow(
    X_train, y_train, batch_size=32
)

validation_generator = validation_datagen.flow(
    X_val, y_val, batch_size=32
)

# Model definition
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(200, 200, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation="relu"))
model.add(layers.Dense(1, activation="sigmoid"))

## Model compilation
#model.compile(
#    loss="binary_crossentropy",
#    optimizer=optimizers.RMSprop(lr=1e-4),
#    metrics=["acc"]
#)

# compilar modelo
model.compile(
    loss="binary_crossentropy",
    optimizer=optimizers.RMSprop(lr=1e-4),
    metrics=["acc", "loss"] # agregar la métrica de pérdida
)

# Model training
history = model.fit(
    train_generator,
    steps_per_epoch=len(X_train) // 32,
    epochs=5,
    validation_data=validation_generator,
    validation_steps=len(X_val) // 32
)

# Obtener las etiquetas verdaderas y las predicciones
val_steps = len(X_val) // 32 + 1
y_true = []
y_pred_prob = []
for i in range(val_steps):
    batch_X, batch_y = next(validation_generator)
    y_true.extend(batch_y)
    y_pred_prob.extend(model.predict(batch_X))

# Convertir las predicciones en etiquetas
y_pred = np.round(y_pred_prob)

# Matriz de confusión
cm = confusion_matrix(y_true, y_pred)
print("Matriz de confusión:\n", cm)

# Reporte de clasificación
cr = classification_report(y_true, y_pred)
print("Reporte de clasificación:\n", cr)


# Métricas adicionales
tn, fp, fn, tp = cm.ravel()
precision = tp / (tp + fp)
sensibilidad = tp / (tp + fn)
especificidad = tn / (tn + fp)
f1_score = 2 * (precision * sensibilidad) / (precision + sensibilidad)
auc = roc_auc_score(y_true, y_pred_prob)
loss, accuracy = model.evaluate(validation_generator, verbose=0)
kappa = cohen_kappa_score(y_true, y_pred)
print("Métricas adicionales:")
print(f"   Pérdida logarítmica: {loss:.4f}")
print(f"   Precisión: {precision:.4f}")
print(f"   Sensibilidad: {sensibilidad:.4f}")
print(f"   Especificidad: {especificidad:.4f}")
print(f"   F1-Score: {f1_score:.4f}")
print(f"   Área bajo la curva (AUC): {auc:.4f}")
print(f"   Cohen's Kappa: {kappa:.4f}")



acc = history.history["acc"]
val_acc = history.history["val_acc"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]

epochs = range(1, len(acc) + 1)

# Plot de precisión
plt.plot(epochs, acc, "bo", label="Precisión de entrenamiento")
plt.plot(epochs, val_acc, "b", label="Precisión de validación")
plt.title("Precisión de entrenamiento y validación")
plt.xlabel("Época")
plt.ylabel("Precisión")
plt.legend()
plt.show()

# Plot de pérdida
plt.plot(epochs, loss, "bo", label="Pérdida de entrenamiento")
plt.plot(epochs, val_loss, "b", label="Pérdida de validación")
plt.title("Pérdida de entrenamiento y validación")
plt.xlabel("Época")
plt.ylabel("Pérdida")
plt.legend()
plt.show()


# Función para graficar la matriz de confusión
def plot_confusion_matrix(cm, classes, normalize=False, title='Matriz de confusión', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Matriz de confusión normalizada")
    else:
        print('Matriz de confusión sin normalizar')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Etiqueta verdadera')
    plt.xlabel('Etiqueta predicha')



# Graficar la matriz de confusión
classes = ['Negativo', 'Positivo']
plot_confusion_matrix(cm, classes, normalize=False, title='Matriz de confusión')
plt.show()


# Definir las etiquetas y los valores de las métricas
labels = ['Loss', 'Precision', 'Sensitivity', 'Specificity', 'F1-Score', 'AUC', 'Cohen\'s Kappa']
values = [loss, precision, sensibilidad, especificidad, f1_score, auc, kappa]

# Crear el gráfico de barras
plt.figure(figsize=(8,6))
plt.bar(labels, values, color=['gray', 'blue', 'green', 'red', 'purple', 'orange', 'brown'])
plt.ylim([0,1]) # Establecer los límites del eje y
plt.title('Métricas adicionales')
plt.xlabel('Métricas')
plt.ylabel('Valores')
plt.show()

