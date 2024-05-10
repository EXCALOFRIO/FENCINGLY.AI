from tensorflow.keras.utils import Sequence
import tensorflow as tf
import numpy as np
from custom_transforms import *

class CustomDataset(tf.keras.utils.Sequence):
    def __init__(self, datos, etiquetas, input_shape, entrenamiento=True):
        self.datos = datos
        self.etiquetas = etiquetas
        self.input_shape = input_shape
        self.entrenamiento = entrenamiento

    def __len__(self):
        return len(self.datos)

    def __getitem__(self, idx):
        datos = self.datos[idx]
        etiquetas = self.etiquetas[idx]
        if self.entrenamiento:
            datos, etiquetas = self.transformar_datos(datos, etiquetas)
        
        etiquetas = np.expand_dims(etiquetas, axis=-1)
        
        return datos, etiquetas

    def transformar_datos(self, datos, etiquetas):
        if self.entrenamiento:  # Solo aplicar transformaciones durante el entrenamiento
            # Lista de transformaciones disponibles
            transformaciones = [transformacion_zoom, desplazar_posesY, desplazar_posesX, flip_poses]

            # Aplicar transformaciones aleatorias
            for transformacion in transformaciones:
                if tf.random.uniform(shape=()) > 0.5:  # Probabilidad de aplicar la transformación 
                    datos, etiquetas = transformacion(datos, etiquetas)
        return datos, etiquetas
    
    
    
def crear_dataloader(datos_entrenamiento, etiquetas_entrenamiento, datos_validacion, etiquetas_validacion, datos_validacion_early,etiquetas_validacion_early, batch_size):
    input_shape = datos_entrenamiento.shape[-1]
    
    # Obtener índices aleatorios únicos para el subconjunto del tamaño del lote
    indices_aleatorios = np.random.choice(len(etiquetas_entrenamiento), size=batch_size, replace=False)

    # Seleccionar subconjunto de datos y etiquetas
    datos_entrenamiento_lote = datos_entrenamiento[indices_aleatorios]
    etiquetas_entrenamiento_lote = etiquetas_entrenamiento[indices_aleatorios]

    dataset_entrenamiento = CustomDataset(datos_entrenamiento_lote, etiquetas_entrenamiento_lote, input_shape, entrenamiento=True)
    dataset_validacion = CustomDataset(datos_validacion, etiquetas_validacion, input_shape, entrenamiento=False)
    dataset_validacion_early = CustomDataset(datos_validacion_early, etiquetas_validacion_early, input_shape, entrenamiento=False)

    train_loader = tf.data.Dataset.from_generator(lambda: dataset_entrenamiento,
                                                output_types=(tf.float32, tf.float32),
                                                output_shapes=(None, (None,))) \
                                                .shuffle(len(dataset_entrenamiento)).batch(batch_size)                                 

    val_loader = tf.data.Dataset.from_generator(lambda: dataset_validacion,
                                                output_types=(tf.float32, tf.float32),
                                                output_shapes=(None, (None,))) \
                                                    .batch(batch_size)
                                                    
    val_earlystopping_loader = tf.data.Dataset.from_generator(lambda: dataset_validacion_early,
                                                output_types=(tf.float32, tf.float32),
                                                output_shapes=(None, (None,))) \
                                                    .batch(batch_size)

    return train_loader, val_loader, val_earlystopping_loader


