import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, models

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Configurar TensorFlow para usar la GPU
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Función para leer vectores de pose desde un archivo .txt con separación por comas
def leer_archivo(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    vectors = []
    labels = []  # Lista para almacenar las etiquetas
    for i, line in enumerate(lines, start=1):
        try:
            # Separar los valores de la línea
            values = list(map(float, line.strip().split(',')))
            # El último valor es la etiqueta
            label = int(values.pop())
            # Añadir el vector y la etiqueta a las listas correspondientes
            vectors.append(values)
            labels.append(label)
        except ValueError:
            print(f"Warning: Line {i} in {file_path} is not a valid vector of pose:", line.strip())
    return vectors, labels


# Rutas a los archivos .txt
ruta_derecha = 'outputDerecha.txt'
ruta_izquierda = 'outputIzquierda.txt'

# Leer vectores de pose
vectores_derecha = leer_archivo(ruta_derecha)
print("Longitud del primer vector de pose:", len(vectores_derecha[0]))

vectores_izquierda = leer_archivo(ruta_izquierda)
print("Longitud del primer vector de pose:", len(vectores_izquierda[0]))


# Convertir a tensores de TensorFlow
# Convertir a tensores de TensorFlow con tamaño de lote 1
dataset_derecha = tf.data.Dataset.from_tensor_slices(vectores_derecha).batch(1)
dataset_izquierda = tf.data.Dataset.from_tensor_slices(vectores_izquierda).batch(1)
test_dataset = tf.data.Dataset.from_tensor_slices(vectores_derecha[:10]).batch(1)


# Ahora puedes continuar con el preprocesamiento y entrenamiento de tus datos utilizando TensorFlow
# Define the architecture of your model
# Define the architecture of your model
# Define the architecture of your model
model = models.Sequential([
    layers.Input(shape=(None,)),  # Define the shape of your input data
    # Add layers as needed, for example:
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(2, activation='softmax')  # Assuming 2 classes (left and right)
])



# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])

# Train the model
model.fit(dataset_derecha, epochs=10, validation_data=dataset_izquierda)

# Evaluate the model on a test dataset
test_loss, test_acc = model.evaluate(test_dataset)
print('Test accuracy:', test_acc)
