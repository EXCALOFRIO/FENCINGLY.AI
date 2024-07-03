import random
import tensorflow as tf
from tensorflow.keras import mixed_precision
import numpy as np

# Enable mixed precision training
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# Configurar TensorFlow para usar GPU si está disponible
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    except RuntimeError as e:
        print(e)

@tf.function
def transformacion_zoom(datos_entrenamiento, etiquetas_entrenamiento):
    # Convertir los datos de entrenamiento a un tensor de TensorFlow
    datos_entrenamiento_tf = tf.convert_to_tensor(datos_entrenamiento)

    # Obtener las dimensiones de los datos de entrenamiento
    shape = tf.shape(datos_entrenamiento_tf)
    num_poses = shape[0]
    num_puntos = shape[1]

    # Calcular los rangos de coordenadas x e y
    datos_no_cero = tf.boolean_mask(datos_entrenamiento_tf, tf.reduce_any(datos_entrenamiento_tf != 0, axis=-1))
    x_min = tf.reduce_min(datos_no_cero[:, 0::2]) if tf.size(datos_no_cero) > 0 else tf.constant(0, dtype=tf.float32)
    y_min = tf.reduce_min(datos_no_cero[:, 1::2]) if tf.size(datos_no_cero) > 0 else tf.constant(0, dtype=tf.float32)
    x_max = tf.reduce_max(datos_entrenamiento_tf[:, :, 0::2])
    y_max = tf.reduce_max(datos_entrenamiento_tf[:, :, 1::2])

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
    centro_x = (x_max + x_min) / 2
    centro_y = (y_max + y_min) / 2

    nueva_x = tf.where(
        datos_entrenamiento_tf[:, :, 0::2] == 0,
        datos_entrenamiento_tf[:, :, 0::2],
        (datos_entrenamiento_tf[:, :, 0::2] - centro_x) * factor_zoom + centro_x
    )

    nueva_y = tf.where(
        datos_entrenamiento_tf[:, :, 1::2] == 0,
        datos_entrenamiento_tf[:, :, 1::2],
        (datos_entrenamiento_tf[:, :, 1::2] - centro_y) * factor_zoom + centro_y
    )

    datos_entrenamiento_trans = tf.reshape(
        tf.stack([nueva_x, nueva_y], axis=-1),
        (num_poses, num_puntos, -1)
    )

    # Devolver los datos de entrenamiento con zoom aplicado y las etiquetas de entrenamiento originales
    return datos_entrenamiento_trans, etiquetas_entrenamiento

@tf.function
def desplazar_posesX(datos_entrenamiento, etiquetas_entrenamiento):
    # Convertir los datos de entrenamiento a tensores de TensorFlow
    datos_entrenamiento_tf = tf.convert_to_tensor(datos_entrenamiento)

    # Obtener las dimensiones de los datos de entrenamiento
    shape = tf.shape(datos_entrenamiento_tf)
    num_poses = shape[0]
    num_puntos = shape[1]
    
    # Calcular el rango permitido para el desplazamiento hacia la derecha (positivo)
    max_x = tf.reduce_max(datos_entrenamiento_tf[:, :, 0::2])
    rango_desplazamiento_derecha = 1 - max_x

    datos_no_cero = tf.boolean_mask(datos_entrenamiento_tf, tf.reduce_all(datos_entrenamiento_tf != 0, axis=-1))
    min_x = tf.reduce_min(datos_no_cero[:, 0::2]) if tf.size(datos_no_cero) > 0 else tf.constant(0, dtype=tf.float32)
    rango_desplazamiento_izquierda = min_x

    # Generar un desplazamiento aleatorio en el eje x dentro del rango permitido
    desplazamiento_x = tf.random.uniform([], minval=-rango_desplazamiento_izquierda, maxval=rango_desplazamiento_derecha)

    # Aplicar el desplazamiento a las coordenadas x de cada punto de la pose
    coordenadas_x = datos_entrenamiento_tf[:, :, 0::2]
    no_cero_mask = tf.cast(coordenadas_x != 0, tf.float32)
    nueva_x = coordenadas_x + desplazamiento_x * no_cero_mask
    nueva_x = tf.clip_by_value(nueva_x, 0, 1)  # Asegurarse de que la coordenada x después del desplazamiento esté dentro del rango [0, 1]

    datos_entrenamiento_trans = tf.stack([
        nueva_x,
        datos_entrenamiento_tf[:, :, 1::2]
    ], axis=-1)

    datos_entrenamiento_trans = tf.reshape(datos_entrenamiento_trans, (num_poses, num_puntos, -1))
    
    # Devolver los datos de entrenamiento desplazados y las etiquetas de entrenamiento originales
    return datos_entrenamiento_trans, etiquetas_entrenamiento

@tf.function
def desplazar_posesY(datos_entrenamiento, etiquetas_entrenamiento):
    # Convertir los datos de entrenamiento a tensores de TensorFlow
    datos_entrenamiento_tf = tf.convert_to_tensor(datos_entrenamiento)

    # Obtener las dimensiones de los datos de entrenamiento
    shape = tf.shape(datos_entrenamiento_tf)
    num_poses = shape[0]
    num_puntos = shape[1]
    
    # Calcular el rango permitido para el desplazamiento hacia arriba (positivo)
    max_y = tf.reduce_max(datos_entrenamiento_tf[:, :, 1::2])
    rango_desplazamiento_arriba = 1 - max_y

    # Filtrar puntos con todas las coordenadas iguales a 0
    datos_no_cero = tf.boolean_mask(datos_entrenamiento_tf, tf.reduce_all(datos_entrenamiento_tf != 0, axis=-1))
    min_y = tf.reduce_min(datos_no_cero[:, 1::2]) if tf.size(datos_no_cero) > 0 else tf.constant(0, dtype=tf.float32)
    rango_desplazamiento_abajo = min_y
    
    # Generar un desplazamiento aleatorio en el eje y dentro del rango permitido
    desplazamiento_y = tf.random.uniform([], minval=-rango_desplazamiento_abajo, maxval=rango_desplazamiento_arriba)
    
    # Aplicar el desplazamiento a las coordenadas y de cada punto de la pose
    coordenadas_y = datos_entrenamiento_tf[:, :, 1::2]
    no_cero_mask = tf.cast(coordenadas_y != 0, tf.float32)
    nueva_y = coordenadas_y + desplazamiento_y * no_cero_mask
    nueva_y = tf.clip_by_value(nueva_y, 0, 1)  # Asegurarse de que la coordenada y después del desplazamiento esté dentro del rango [0, 1]

    datos_entrenamiento_trans = tf.stack([
        datos_entrenamiento_tf[:, :, 0::2],
        nueva_y
    ], axis=-1)

    datos_entrenamiento_trans = tf.reshape(datos_entrenamiento_trans, (num_poses, num_puntos, -1))
    
    # Devolver los datos de entrenamiento desplazados y las etiquetas de entrenamiento originales
    
    return datos_entrenamiento_trans, etiquetas_entrenamiento

@tf.function
def flip_poses(datos_entrenamiento, etiquetas_entrenamiento):
    # Convertir los datos de entrenamiento y las etiquetas a tensores de TensorFlow
    datos_entrenamiento_tf = tf.convert_to_tensor(datos_entrenamiento)
    etiquetas_entrenamiento_tf = tf.convert_to_tensor(etiquetas_entrenamiento)
    
    # Obtener las dimensiones de los datos de entrenamiento
    num_poses = tf.shape(datos_entrenamiento_tf)[0]
    num_puntos = tf.shape(datos_entrenamiento_tf)[1]
    
    # Aplicar el flip horizontal a las coordenadas x de cada punto de la pose
    coordenadas_x = datos_entrenamiento_tf[:, :, 0::2]
    nueva_x = 1 - coordenadas_x
    
    datos_entrenamiento_trans = tf.reshape(
        tf.stack([nueva_x, datos_entrenamiento_tf[:, :, 1::2]], axis=-1),
        (num_poses, num_puntos, -1)
    )
    
    # Cambiar el orden de las poses
    indices_orden = tf.constant([1, 0], dtype=tf.int32)
    datos_entrenamiento_trans = tf.gather(datos_entrenamiento_trans, indices_orden, axis=1)

    # Invertir las etiquetas de entrenamiento
    etiquetas_entrenamiento_trans = 1 - etiquetas_entrenamiento_tf
    
    # Devolver los datos de entrenamiento con flip aplicado y las etiquetas invertidas
    return datos_entrenamiento_trans, etiquetas_entrenamiento_trans

@tf.function
def rotar_poses(datos_entrenamiento, etiquetas_entrenamiento):
    datos_entrenamiento_tf = tf.convert_to_tensor(datos_entrenamiento, dtype=tf.float32)
    shape = tf.shape(datos_entrenamiento_tf)
    num_poses = shape[0]
    num_puntos = shape[1]

    centro_x = tf.reduce_mean(datos_entrenamiento_tf[:, :, 0::2])
    centro_y = tf.reduce_mean(datos_entrenamiento_tf[:, :, 1::2])

    # Ángulo aleatorio entre -90 y 90 grados en radianes
    angulo_rotacion = tf.random.uniform([], minval=-np.pi/2, maxval=np.pi/2)

    cos_val = tf.cos(angulo_rotacion)
    sin_val = tf.sin(angulo_rotacion)
    matriz_rotacion = tf.reshape(tf.stack([cos_val, -sin_val, sin_val, cos_val]), [2, 2])

    datos_entrenamiento_reshaped = tf.reshape(datos_entrenamiento_tf, [-1, 2])
    centros = tf.tile([[centro_x, centro_y]], [tf.shape(datos_entrenamiento_reshaped)[0], 1])

    # Crear una máscara para los puntos con coordenadas no cero
    no_cero_mask = tf.reduce_any(datos_entrenamiento_reshaped != 0, axis=-1, keepdims=True)

    # Aplicar la rotación solo a los puntos no cero
    datos_rotados = tf.where(
        no_cero_mask,
        tf.matmul(datos_entrenamiento_reshaped - centros, matriz_rotacion) + centros,
        datos_entrenamiento_reshaped
    )

    min_coords = tf.reduce_min(datos_rotados, axis=0)
    max_coords = tf.reduce_max(datos_rotados, axis=0)

    factor_escala = tf.minimum(1.0 / (max_coords - min_coords), 1.0)
    datos_escalados = (datos_rotados - min_coords) * factor_escala

    datos_entrenamiento_trans = tf.reshape(datos_escalados, tf.shape(datos_entrenamiento_tf))
    datos_entrenamiento_trans = tf.reshape(datos_entrenamiento_trans, (num_poses, num_puntos, -1))

    return datos_entrenamiento_trans, etiquetas_entrenamiento

@tf.function
def eliminar_puntos_aleatorios(datos_entrenamiento, etiquetas_entrenamiento):
    # Convertir los datos de entrenamiento a un tensor de TensorFlow
    datos_entrenamiento_tf = tf.convert_to_tensor(datos_entrenamiento)

    # Obtener las dimensiones de los datos de entrenamiento
    shape = tf.shape(datos_entrenamiento_tf)
    num_poses, num_puntos, dim_punto = tf.unstack(shape)

    # Generar una máscara aleatoria para cada pose y cada punto
    mascaras = tf.random.uniform((num_poses, num_puntos, dim_punto), minval=0, maxval=1, dtype=tf.float32)
    mascaras = tf.cast(mascaras > 0.98, tf.float32)  # Establecer entre 0 y 6 puntos a 1 (los demás serán 0)
    num_puntos_eliminados = tf.cast(tf.reduce_sum(mascaras, axis=[1, 2]), tf.int32)  # Número de puntos eliminados por pose

    # Aplicar la máscara a los datos de entrenamiento
    datos_entrenamiento_trans = datos_entrenamiento_tf * (1 - mascaras)
    
    datos_entrenamiento_trans = tf.reshape(datos_entrenamiento_trans, (num_poses, num_puntos, -1))

    return datos_entrenamiento_trans, etiquetas_entrenamiento