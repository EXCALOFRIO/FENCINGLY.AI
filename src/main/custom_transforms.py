import random
import numpy as np
import tensorflow as tf

def transformacion_zoom(datos_entrenamiento, etiquetas_entrenamiento):
    # Convertir los datos de entrenamiento a un tensor de TensorFlow
    datos_entrenamiento_tf = tf.convert_to_tensor(datos_entrenamiento)

    # Obtener las dimensiones de los datos de entrenamiento
    num_poses, num_puntos, _ = datos_entrenamiento_tf.shape

    # Calcular los rangos de coordenadas x e y
    datos_no_cero = datos_entrenamiento_tf[tf.reduce_all(datos_entrenamiento_tf != 0, axis=-1)]
    x_min = tf.reduce_min(datos_no_cero[:, 0::3]) if datos_no_cero.numpy().size > 0 else 0
    y_min = tf.reduce_min(datos_no_cero[:, 1::3]) if datos_no_cero.numpy().size > 0 else 0
    x_max = tf.reduce_max(datos_entrenamiento_tf[:, :, 0::3])
    y_max = tf.reduce_max(datos_entrenamiento_tf[:, :, 1::3])

    # Determinar aleatoriamente si se aplicará zoom in o zoom out
    zoom_in = tf.random.uniform([], minval=0, maxval=2, dtype=tf.int32)

    # Calcular el rango de zoom permitido
    rango_zoom_x = tf.maximum(x_max - x_min, 1e-6)
    rango_zoom_y = tf.maximum(y_max - y_min, 1e-6)

    rangomin = 1 / (1 + 0.5 * tf.maximum(rango_zoom_x, rango_zoom_y))
    rangomax = (1 + 0.5) * tf.minimum(rango_zoom_x, rango_zoom_y)

    # Generar un factor de zoom aleatorio dentro del rango permitido
    factor_zoom = tf.random.uniform([], minval=rangomin, maxval=rangomax)

    # Aplicar el zoom a las coordenadas x e y de cada punto de la pose
    datos_entrenamiento_trans = tf.Variable(datos_entrenamiento_tf)
    for i in range(num_puntos):
        coordenadas_x = datos_entrenamiento_trans[:, i, 0::3]
        coordenadas_y = datos_entrenamiento_trans[:, i, 1::3]

        # Calcular el punto central de la pose
        centro_x = (x_max + x_min) / 2
        centro_y = (y_max + y_min) / 2

        # Aplicar el zoom
        nueva_x = (coordenadas_x - centro_x) * factor_zoom + centro_x
        nueva_y = (coordenadas_y - centro_y) * factor_zoom + centro_y

        # Actualizar las coordenadas en los datos de entrenamiento
        datos_entrenamiento_trans[:, i, 0::3].assign(nueva_x)
        datos_entrenamiento_trans[:, i, 1::3].assign(nueva_y)

    # Devolver los datos de entrenamiento con zoom aplicado y las etiquetas de entrenamiento originales
    return datos_entrenamiento_trans.numpy(), etiquetas_entrenamiento


def desplazar_posesX(datos_entrenamiento, etiquetas_entrenamiento):
    # Convertir los datos de entrenamiento a tensores de TensorFlow
    datos_entrenamiento_tf = tf.convert_to_tensor(datos_entrenamiento)
    
    # Obtener las dimensiones de los datos de entrenamiento
    num_poses, num_puntos, _ = datos_entrenamiento_tf.shape
    
    # Calcular el rango permitido para el desplazamiento hacia la derecha (positivo)
    max_x = tf.reduce_max(datos_entrenamiento_tf[:, :, 0::3])
    rango_desplazamiento_derecha = 1 - max_x

    datos_no_cero = datos_entrenamiento_tf[tf.reduce_all(datos_entrenamiento_tf != 0, axis=-1)]
    min_x = tf.reduce_min(datos_no_cero[:, 0::3]) if datos_no_cero.numpy().size > 0 else 0
    rango_desplazamiento_izquierda = min_x

    # Generar un desplazamiento aleatorio en el eje x dentro del rango permitido
    desplazamiento_x = tf.random.uniform([], minval=-rango_desplazamiento_izquierda, maxval=rango_desplazamiento_derecha)

    # Aplicar el desplazamiento a las coordenadas x de cada punto de la pose
    datos_entrenamiento_trans = tf.Variable(datos_entrenamiento_tf)
    for i in range(num_puntos):
        coordenadas_x = datos_entrenamiento_trans[:, i, 0::3]
        no_cero_mask = tf.cast(coordenadas_x != 0, tf.float32)
        nueva_x = coordenadas_x + desplazamiento_x * no_cero_mask
        nueva_x = tf.clip_by_value(nueva_x, 0, 1)  # Asegurarse de que la coordenada x después del desplazamiento esté dentro del rango [0, 1]
        datos_entrenamiento_trans[:, i, 0::3].assign(nueva_x)
    
    # Devolver los datos de entrenamiento desplazados y las etiquetas de entrenamiento originales
    return datos_entrenamiento_trans.numpy(), etiquetas_entrenamiento

def desplazar_posesY(datos_entrenamiento, etiquetas_entrenamiento):
    # Convertir los datos de entrenamiento a tensores de TensorFlow
    datos_entrenamiento_tf = tf.convert_to_tensor(datos_entrenamiento)
    
    # Obtener las dimensiones de los datos de entrenamiento
    num_poses, num_puntos, _ = datos_entrenamiento_tf.shape
    
    # Calcular el rango permitido para el desplazamiento hacia arriba (positivo)
    max_y = tf.reduce_max(datos_entrenamiento_tf[:, :, 1::3])
    rango_desplazamiento_arriba = 1 - max_y

    # Filtrar puntos con todas las coordenadas iguales a 0 y confianza igual a 0
    datos_no_cero = datos_entrenamiento_tf[tf.reduce_all(datos_entrenamiento_tf != 0, axis=-1)]
    min_y = tf.reduce_min(datos_no_cero[:, 1::3]) if datos_no_cero.numpy().size > 0 else 0
    rango_desplazamiento_abajo = min_y
    
    # Generar un desplazamiento aleatorio en el eje y dentro del rango permitido
    desplazamiento_y = tf.random.uniform([], minval=-rango_desplazamiento_abajo, maxval=rango_desplazamiento_arriba)
    
    # Aplicar el desplazamiento a las coordenadas y de cada punto de la pose
    datos_entrenamiento_trans = tf.Variable(datos_entrenamiento_tf)
    for i in range(num_puntos):
        coordenadas_y = datos_entrenamiento_trans[:, i, 1::3]
        no_cero_mask = tf.cast(coordenadas_y != 0, tf.float32)
        nueva_y = coordenadas_y + desplazamiento_y * no_cero_mask
        nueva_y = tf.clip_by_value(nueva_y, 0, 1)  # Asegurarse de que la coordenada y después del desplazamiento esté dentro del rango [0, 1]
        datos_entrenamiento_trans[:, i, 1::3].assign(nueva_y)
    
    # Devolver los datos de entrenamiento desplazados y las etiquetas de entrenamiento originales
    return datos_entrenamiento_trans.numpy(), etiquetas_entrenamiento

def flip_poses(datos_entrenamiento, etiquetas_entrenamiento):
    # Convertir los datos de entrenamiento a un tensor de TensorFlow
    datos_entrenamiento_tf = tf.convert_to_tensor(datos_entrenamiento)
    
    # Obtener las dimensiones de los datos de entrenamiento
    num_poses, num_puntos, _ = datos_entrenamiento_tf.shape
    
    # Aplicar el flip horizontal a las coordenadas x de cada punto de la pose
    datos_entrenamiento_trans = tf.Variable(datos_entrenamiento_tf)
    datos_entrenamiento_trans[:, :, 0::3].assign(1 - datos_entrenamiento_trans[:, :, 0::3])

    # Cambiar el orden de las poses
    datos_entrenamiento_trans = tf.transpose(datos_entrenamiento_trans, perm=[0, 2, 1])
    datos_entrenamiento_trans = tf.transpose(datos_entrenamiento_trans, perm=[0, 2, 1])
    
    # Crear un tensor con el orden invertido [1, 0]
    reverse_tensor = tf.constant([1, 0], dtype=tf.int32)
    
    # Indexar el tensor con el orden invertido
    datos_entrenamiento_trans = tf.gather(datos_entrenamiento_trans, reverse_tensor, axis=1)
    
    # Devolver los datos de entrenamiento con flip aplicado y las etiquetas invertidas
    return datos_entrenamiento_trans.numpy(), invertir_etiquetas(etiquetas_entrenamiento)



def invertir_etiquetas(etiquetas_entrenamiento):
    etiquetas_entrenamiento_tf = tf.convert_to_tensor(etiquetas_entrenamiento)
    return 1 - etiquetas_entrenamiento_tf