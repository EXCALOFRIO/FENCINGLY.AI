import torch
import random
import numpy as np


#Genera la transformacion e imprime el tensor original y el transformado y la diferencia
def transformacion_dato(datos_entrenamiento, etiquetas_entrenamiento, datos_transformados, etiquetas_transformadas):
    # Seleccionar un índice aleatorio
    indice_ejemplo = np.random.randint(len(datos_transformados))
    
    # Seleccionar el dato original y su etiqueta correspondiente
    dato_original = datos_entrenamiento[indice_ejemplo]
    etiqueta_original = etiquetas_entrenamiento[indice_ejemplo]
    
    # Aplicar transformaciones individualmente
    dato_transformado = datos_transformados[indice_ejemplo]
    etiqueta_transformada = etiquetas_transformadas[indice_ejemplo]
    
    # Imprimir el dato original y su versión transformada, junto con sus etiquetas
    print("Dato original:")
    print(dato_original)
    print("Etiqueta original:", etiqueta_original)
    print("\nDato transformado:")
    print(dato_transformado)
    print("Etiqueta transformada:", etiqueta_transformada)
    print("Diferencia entre dato original y transformado:")
    print(dato_original - dato_transformado)


def transformacion_zoom(datos_entrenamiento, etiquetas_entrenamiento):
    # Copiar los datos de entrenamiento para no modificar los originales
    datos_entrenamiento_trans = datos_entrenamiento.clone()

    # Obtener las dimensiones de los datos de entrenamiento
    num_poses, num_puntos, _ = datos_entrenamiento.size()

    # Calcular los rangos de coordenadas x e y
    datos_no_cero = datos_entrenamiento[torch.all(datos_entrenamiento != 0, dim=-1)]
    x_min = datos_no_cero[:, 0::3].min().item() if datos_no_cero.numel() > 0 else 0
    y_min = datos_no_cero[:, 1::3].min().item() if datos_no_cero.numel() > 0 else 0
    x_max = datos_entrenamiento_trans[:, :, 0::3].max().item()
    y_max = datos_entrenamiento_trans[:, :, 0::3].max().item()

    # Determinar aleatoriamente si se aplicará zoom in o zoom out
    zoom_in = random.choice([True, False])

    # Calcular el rango de zoom permitido
    rango_zoom_x = max(x_max - x_min, 1e-6)
    rango_zoom_y = max(y_max - y_min, 1e-6)
    
    rangomin = ((1 / (1 + 0.5 * max(rango_zoom_x, rango_zoom_y))))
    rangomax = ((1 + 0.5) * min(rango_zoom_x, rango_zoom_y))
    # Generar un factor de zoom aleatorio dentro del rango permitido

    factor_zoom = random.uniform(rangomin, rangomax)


    # Aplicar el zoom a las coordenadas x e y de cada punto de la pose
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
        datos_entrenamiento_trans[:, i, 0::3] = nueva_x
        datos_entrenamiento_trans[:, i, 1::3] = nueva_y

    # Devolver los datos de entrenamiento con zoom aplicado y las etiquetas de entrenamiento originales
    return datos_entrenamiento_trans, etiquetas_entrenamiento


def desplazar_posesX(datos_entrenamiento, etiquetas_entrenamiento):
    # Copiar los datos de entrenamiento para no modificar los originales
    datos_entrenamiento_trans = datos_entrenamiento.clone()
    
    # Obtener las dimensiones de los datos de entrenamiento
    num_poses, num_puntos, _ = datos_entrenamiento.size()
    
    # Calcular el rango permitido para el desplazamiento hacia la derecha (positivo)
    max_x = datos_entrenamiento_trans[:, :, 0::3].max().item()
    rango_desplazamiento_derecha = 1 - max_x

    datos_no_cero = datos_entrenamiento[torch.all(datos_entrenamiento != 0, dim=-1)]
    min_x = datos_no_cero[:, 0::3].min().item() if datos_no_cero.numel() > 0 else 0
    rango_desplazamiento_izquierda = min_x

    
    # Generar un desplazamiento aleatorio en el eje x dentro del rango permitido
    desplazamiento_x = random.uniform(-rango_desplazamiento_izquierda, rango_desplazamiento_derecha)

    # Aplicar el desplazamiento a las coordenadas x de cada punto de la pose
    for i in range(num_puntos):
        coordenadas_x = datos_entrenamiento_trans[:, i, 0::3]
        no_cero_mask = coordenadas_x != 0
        nueva_x = coordenadas_x.clone()
        nueva_x[no_cero_mask] += desplazamiento_x
        nueva_x = nueva_x.clamp(0, 1)  # Asegurarse de que la coordenada x después del desplazamiento esté dentro del rango [0, 1]
        datos_entrenamiento_trans[:, i, 0::3] = nueva_x
    
    # Devolver los datos de entrenamiento desplazados y las etiquetas de entrenamiento originales
    return datos_entrenamiento_trans, etiquetas_entrenamiento

def desplazar_posesY(datos_entrenamiento, etiquetas_entrenamiento):
    # Copiar los datos de entrenamiento para no modificar los originales
    datos_entrenamiento_trans = datos_entrenamiento.clone()
    
    # Obtener las dimensiones de los datos de entrenamiento
    num_poses, num_puntos, _ = datos_entrenamiento.size()
    
    # Calcular el rango permitido para el desplazamiento hacia arriba (positivo)
    max_y = datos_entrenamiento_trans[:, :, 1::3].max().item()
    rango_desplazamiento_arriba = 1 - max_y

    # Filtrar puntos con todas las coordenadas iguales a 0 y confianza igual a 0
    datos_no_cero = datos_entrenamiento[torch.all(datos_entrenamiento != 0, dim=-1)]
    min_y = datos_no_cero[:, 1::3].min().item() if datos_no_cero.numel() > 0 else 0

    rango_desplazamiento_abajo = min_y
    
    # Generar un desplazamiento aleatorio en el eje y dentro del rango permitido
    desplazamiento_y = random.uniform(-rango_desplazamiento_abajo, rango_desplazamiento_arriba)
    
    # Aplicar el desplazamiento a las coordenadas y de cada punto de la pose
    for i in range(num_puntos):
        coordenadas_y = datos_entrenamiento_trans[:, i, 1::3]
        no_cero_mask = coordenadas_y != 0
        nueva_y = coordenadas_y.clone()
        nueva_y[no_cero_mask] += desplazamiento_y
        nueva_y = nueva_y.clamp(0, 1)  # Asegurarse de que la coordenada y después del desplazamiento esté dentro del rango [0, 1]
        datos_entrenamiento_trans[:, i, 1::3] = nueva_y
    
    # Devolver los datos de entrenamiento desplazados y las etiquetas de entrenamiento originales
    return datos_entrenamiento_trans, etiquetas_entrenamiento

def flip_poses(datos_entrenamiento, etiquetas_entrenamiento):
    # Copiar los datos de entrenamiento para no modificar los originales
    datos_entrenamiento_trans = datos_entrenamiento.clone()
    
    # Obtener las dimensiones de los datos de entrenamiento
    num_poses, num_puntos, _ = datos_entrenamiento.size()
    
    # Aplicar el flip horizontal a las coordenadas x de cada punto de la pose
    datos_entrenamiento_trans[:, 0::3] = 1 - datos_entrenamiento_trans[:, 0::3]

    # Devolver los datos de entrenamiento con flip aplicado y las etiquetas invertidas
    return datos_entrenamiento_trans, invertir_etiquetas(etiquetas_entrenamiento)


def invertir_etiquetas(etiquetas_entrenamiento):
    return 1 - etiquetas_entrenamiento
