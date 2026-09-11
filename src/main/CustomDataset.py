import tensorflow as tf
from tensorflow.keras.utils import Sequence
import numpy as np
from custom_transforms import *
import random

class CustomDataGenerator(Sequence):
    def __init__(self, data, labels, batch_size=32, shuffle=True, transformations=None, transformation_prob=0.8):
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.transformations = transformations if transformations is not None else []
        self.transformation_prob = transformation_prob
        self.transformed_data, self.transformed_labels = self.precalculate_transformations()
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.transformed_data) / self.batch_size))

    def __getitem__(self, index):
        indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        data_batch = [self.transformed_data[i] for i in indices]
        labels_batch = [self.transformed_labels[i] for i in indices]

        return np.array(data_batch), np.array(labels_batch)

    def on_epoch_end(self):
        self.indices = np.arange(len(self.transformed_data))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def precalculate_transformations(self):
        transformed_data = []
        transformed_labels = []

        # Agregar datos originales sin transformar
        transformed_data.extend(self.data)
        transformed_labels.extend(self.labels)

        # Agregar datos transformados
        for data_item, label in zip(self.data, self.labels):
            for transform in self.transformations:
                if random.random() < self.transformation_prob:
                    transformed_data_item, transformed_label = transform(tf.identity(data_item), tf.identity(label))
                    transformed_data.append(transformed_data_item)
                    transformed_labels.append(transformed_label)

        return transformed_data, transformed_labels

def crear_dataloader(datos_entrenamiento, etiquetas_entrenamiento, datos_validacion, etiquetas_validacion, batch_size, transformation_prob=0.7):
    transformaciones = [desplazar_posesY, flip_poses, transformacion_zoom ]
    
    # Imprimir el tamaño del dataset original
    print("Tamaño del dataset original:", len(datos_entrenamiento))
    
    train_generator = CustomDataGenerator(datos_entrenamiento, etiquetas_entrenamiento, batch_size, shuffle=True, transformations=transformaciones, transformation_prob=transformation_prob)
    
    # Imprimir el tamaño total después de agregar los datos transformados
    print("Tamaño total después de agregar los datos transformados:", len(train_generator.transformed_data))
    
    val_generator = CustomDataGenerator(datos_validacion, etiquetas_validacion, batch_size, shuffle=False)

    return train_generator, val_generator 