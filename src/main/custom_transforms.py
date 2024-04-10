import torch
import random
import numpy as np


#Genera la transformacion e imprime el tensor original y el transformado y la diferencia
def transformacion_dato(datos_entrenamiento, etiquetas_entrenamiento, datos_transformados, etiquetas_transformadas):
    # Seleccionar un índice aleatorio
    indice_ejemplo = np.random.randint(len(datos_entrenamiento))
    
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
    
    
def transformacion_zoom(datos_originales, factor_zoom = 200 ):
    # Generar un tensor de desplazamientos con una dimensión menos que los datos originales
    desplazamientos = torch.randn(datos_originales.size()[:-3])
    # Expandir el tensor de desplazamientos para que tenga las mismas dimensiones que los datos originales
    desplazamientos_expandidos = desplazamientos.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    # **Máscara para seleccionar solo valores distintos de 0**
    mascara = datos_originales != 0
    # Aplicar la máscara a los desplazamientos expandidos
    desplazamientos_filtrados = desplazamientos_expandidos * mascara.float()
    # Multiplicar los desplazamientos filtrados por el factor de zoom
    desplazamientos_zoom = desplazamientos_filtrados * factor_zoom
    # Sumar los desplazamientos filtrados a los datos originales
    datos_transformados = datos_originales + desplazamientos_zoom
    return datos_transformados.clone().detach().requires_grad_(True)

#Las coordenadas del vextor son 32 puntos con 3 coordenadas cada uno (x, y, z) la z es la confianza de la detección
def desplazar_posesX(datos_entrenamiento, desplazamiento_max=1000):
    # Obtener las dimensiones del tensor de entrada
    num_videos, num_frames, num_personas, num_puntos = datos_entrenamiento.shape

    # Generar un único desplazamiento por video
    desplazamientos = torch.randint(-desplazamiento_max, desplazamiento_max + 1, (num_videos, 1, 1, 1))

    # Crear una máscara para las coordenadas x de los puntos de la pose
    mascara_x = torch.tensor([1, 0, 0] * (num_puntos // 3), dtype=torch.bool)

    # Obtener los puntos con valor 0 en las coordenadas x originales
    puntos_cero_x = datos_entrenamiento[:, :, :, mascara_x] == 0

    # Aplicar el desplazamiento a las coordenadas x de forma vectorizada
    desplazamiento_aplicado = datos_entrenamiento[:, :, :, mascara_x] + desplazamientos.expand(-1, num_frames, num_personas, num_puntos // 3)

    # Sustituir las coordenadas x con el desplazamiento aplicado solo si el valor original no es 0
    desplazamiento_aplicado[puntos_cero_x] = 0

    # Crear una máscara para las coordenadas y y z de los puntos de la pose
    mascara_yz = ~mascara_x

    # Copiar las coordenadas y z sin aplicar desplazamiento
    datos_entrenamiento_trans = datos_entrenamiento.clone()
    datos_entrenamiento_trans[:, :, :, mascara_yz] = datos_entrenamiento[:, :, :, mascara_yz]

    # Sustituir las coordenadas x con el desplazamiento aplicado
    datos_entrenamiento_trans[:, :, :, mascara_x] = desplazamiento_aplicado

    return datos_entrenamiento_trans.clone().detach().requires_grad_(True)

def desplazar_posesY(datos_entrenamiento, desplazamiento_max=1000):
    # Obtener las dimensiones del tensor de entrada
    num_videos, num_frames, num_personas, num_puntos = datos_entrenamiento.shape

    # Generar un único desplazamiento por video
    desplazamientos = torch.randint(-desplazamiento_max, desplazamiento_max + 1, (num_videos, 1, 1, 1))

    # Crear una máscara para las coordenadas y de los puntos de la pose
    mascara_y = torch.tensor([0, 1, 0] * (num_puntos // 3), dtype=torch.bool)

    # Obtener los puntos con valor 0 en las coordenadas y originales
    puntos_cero_y = datos_entrenamiento[:, :, :, mascara_y] == 0

    # Aplicar el desplazamiento a las coordenadas y de forma vectorizada
    desplazamiento_aplicado = datos_entrenamiento[:, :, :, mascara_y] + desplazamientos.expand(-1, num_frames, num_personas, num_puntos // 3)

    # Sustituir las coordenadas y con el desplazamiento aplicado solo si el valor original no es 0
    desplazamiento_aplicado[puntos_cero_y] = 0

    # Crear una máscara para las coordenadas x y z de los puntos de la pose
    mascara_xz = ~mascara_y

    # Copiar las coordenadas x y z sin aplicar desplazamiento
    datos_entrenamiento_trans = datos_entrenamiento.clone()
    datos_entrenamiento_trans[:, :, :, mascara_xz] = datos_entrenamiento[:, :, :, mascara_xz]

    # Sustituir las coordenadas y con el desplazamiento aplicado
    datos_entrenamiento_trans[:, :, :, mascara_y] = desplazamiento_aplicado

    return datos_entrenamiento_trans.clone().detach().requires_grad_(True)

def flip_poses(datos_entrenamiento):
    # Reflejar las coordenadas x
    datos_entrenamiento_flipped = datos_entrenamiento.clone()
    datos_entrenamiento_flipped[:, :, :, 0::3] *= -1  # Reflejar coordenadas x (cada tercer valor)
    
    return datos_entrenamiento_flipped.clone().detach().requires_grad_(True)

def invertir_etiquetas(etiquetas_entrenamiento):
    return 1 - etiquetas_entrenamiento