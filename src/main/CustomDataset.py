import tensorflow as tf
from tensorflow.keras.utils import Sequence
import numpy as np
from custom_transforms import *
import random
import datetime
import os
from tensorflow.keras import mixed_precision
from generar_gif import *
# Enable mixed precision training
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

class CustomDataGenerator(Sequence):
    def __init__(self, data, labels, batch_size=32, shuffle=True, transformations=None, transformation_prob=0.8):
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.transformations = transformations if transformations is not None else []
        self.transformation_prob = transformation_prob
        self.on_epoch_end()
        
    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))
    
    def __getitem__(self, index):
        indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        data_batch = [self.data[i] for i in indices]
        labels_batch = [self.labels[i] for i in indices]

        # Apply transformations on the fly
        transformed_data_batch = []
        transformed_labels_batch = []
        for data_item, label in zip(data_batch, labels_batch):
            transformed_data_item = tf.identity(data_item)
            transformed_label = tf.identity(label)
            for transform in self.transformations:
                if random.random() < self.transformation_prob:
                    transformed_data_item, transformed_label = transform(transformed_data_item, transformed_label)
            transformed_data_batch.append(transformed_data_item)
            transformed_labels_batch.append(transformed_label)
            
            #imprimir cada elemento de transformed_data_batch
            for transformed, lbl in zip(transformed_data_batch, transformed_labels_batch):
                gif(transformed)
                # Imprimir su etiqueta
                print("Etiqueta correspondiente:", lbl)
                
                
        return np.array(transformed_data_batch), np.array(transformed_labels_batch)
    
    def on_epoch_end(self):
        self.indices = np.arange(len(self.data))
        if self.shuffle:
            np.random.shuffle(self.indices)

def crear_dataloader(datos_entrenamiento, etiquetas_entrenamiento, datos_validacion, etiquetas_validacion, batch_size, transformation_prob=0.7):
    transformaciones = [desplazar_posesY, flip_poses, transformacion_zoom]
    print("Tamaño del dataset original:", len(datos_entrenamiento))
    train_generator = CustomDataGenerator(datos_entrenamiento, etiquetas_entrenamiento, batch_size, shuffle=True, transformations=transformaciones, transformation_prob=transformation_prob)
    print("Tamaño total después de agregar los datos transformados:", len(datos_entrenamiento))
    val_generator = CustomDataGenerator(datos_validacion, etiquetas_validacion, batch_size, shuffle=False)
    return train_generator, val_generator
