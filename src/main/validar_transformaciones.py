import torch
import torch.nn.functional as F
import random
from custom_transforms import *

def validacionX(datos_entrenamiento, etiquetas_entrenamiento):
    # Definir los rangos recomendados para las métricas de validación
    rango_mse = (0, 0.2)  # Rango recomendado para MSE
    rango_correlacion = (0.9, 1)  # Rango recomendado para la correlación

    # Inicializar contadores para contar las transformaciones dentro y fuera de los rangos recomendados
    dentro_rango = 0
    fuera_rango = 0

    # Iterar sobre todos los datos de entrenamiento y aplicar las transformaciones
    for i in range(len(datos_entrenamiento)):
        datos_original = datos_entrenamiento[i]
        etiquetas_original = etiquetas_entrenamiento[i]

        # Aplicar la función de desplazamiento
        datos_transformados, _ = desplazar_posesX(datos_original, etiquetas_original)

        # Calcular las métricas de validación
        mse = calcular_mse(datos_original, datos_transformados)
        correlacion = calcular_correlacion(datos_original, datos_transformados)
        
        # Verificar si las métricas están dentro de los rangos recomendados
        if rango_mse[0] <= mse <= rango_mse[1]  and rango_correlacion[0] <= correlacion <= rango_correlacion[1]:
            dentro_rango += 1
        else:
            fuera_rango += 1

    # Calcular el porcentaje de transformaciones dentro y fuera de los rangos recomendados
    total_transformaciones = dentro_rango + fuera_rango
    porcentaje_dentro = (dentro_rango / total_transformaciones) * 100
    porcentaje_fuera = (fuera_rango / total_transformaciones) * 100

    # Imprimir resultados
    print("Transformaciones dentro del rango recomendado:", porcentaje_dentro, "%")
    print("Transformaciones fuera del rango recomendado:", porcentaje_fuera, "%")

def validacionY(datos_entrenamiento, etiquetas_entrenamiento):
    # Definir los rangos recomendados para las métricas de validación
    rango_mse = (0, 0.2)  # Rango recomendado para MSE
    rango_correlacion = (0.9, 1)  # Rango recomendado para la correlación

    # Inicializar contadores para contar las transformaciones dentro y fuera de los rangos recomendados
    dentro_rango = 0
    fuera_rango = 0

    # Iterar sobre todos los datos de entrenamiento y aplicar las transformaciones
    for i in range(len(datos_entrenamiento)):
        datos_original = datos_entrenamiento[i]
        etiquetas_original = etiquetas_entrenamiento[i]

        # Aplicar la función de desplazamiento
        datos_transformados, _ = desplazar_posesY(datos_original, etiquetas_original)

        # Calcular las métricas de validación
        mse = calcular_mse(datos_original, datos_transformados)
        correlacion = calcular_correlacion(datos_original, datos_transformados)

        # Verificar si las métricas están dentro de los rangos recomendados
        if rango_mse[0] <= mse <= rango_mse[1]  and rango_correlacion[0] <= correlacion <= rango_correlacion[1]:
            dentro_rango += 1
        else:
            fuera_rango += 1

    # Calcular el porcentaje de transformaciones dentro y fuera de los rangos recomendados
    total_transformaciones = dentro_rango + fuera_rango
    porcentaje_dentro = (dentro_rango / total_transformaciones) * 100
    porcentaje_fuera = (fuera_rango / total_transformaciones) * 100

    # Imprimir resultados
    print("Transformaciones dentro del rango recomendado:", porcentaje_dentro, "%")
    print("Transformaciones fuera del rango recomendado:", porcentaje_fuera, "%")

def validacionZoom(datos_entrenamiento, etiquetas_entrenamiento):
    # Definir los rangos recomendados para las métricas de validación
    rango_mse = (0, 0.2)  # Rango recomendado para MSE
    rango_correlacion = (0.9, 1)  # Rango recomendado para la correlación

    # Inicializar contadores para contar las transformaciones dentro y fuera de los rangos recomendados
    dentro_rango = 0
    fuera_rango = 0
    fuera_rango_total = 0
    total_puntos_analizados = 0
    total_puntos_fuera_rango = 0

    # Iterar sobre todos los datos de entrenamiento y aplicar las transformaciones
    for i in range(len(datos_entrenamiento)):
        datos_original = datos_entrenamiento[i]
        etiquetas_original = etiquetas_entrenamiento[i]

        # Aplicar la función de desplazamiento
        datos_transformados, _ = transformacion_zoom(datos_original, etiquetas_original)
        puntos_totales, puntos_fuera_de_rango, _ = transformacion_zoom_validacion2(datos_original, etiquetas_original, 100)
        total_puntos_analizados += puntos_totales
        total_puntos_fuera_rango += puntos_fuera_de_rango

        # Calcular las métricas de validación
        mse = calcular_mse(datos_original, datos_transformados)
        correlacion = calcular_correlacion(datos_original, datos_transformados)
        
        # Verificar si las métricas están dentro de los rangos recomendados
        if rango_mse[0] <= mse <= rango_mse[1] and  rango_correlacion[0] <= correlacion <= rango_correlacion[1]:
            dentro_rango += 1
        else:
            fuera_rango += 1

    # Calcular el porcentaje de transformaciones dentro y fuera de los rangos recomendados
    total_transformaciones = dentro_rango + fuera_rango
    porcentaje_dentro = (dentro_rango / total_transformaciones) * 100
    porcentaje_fuera = (fuera_rango / total_transformaciones) * 100

    # Calcular el porcentaje total de puntos fuera del rango
    porcentaje_fuera_rango_total = (total_puntos_fuera_rango / total_puntos_analizados) * 100 if total_puntos_analizados > 0 else 0

    # Imprimir resultados
    print("Transformaciones dentro del rango recomendado:", porcentaje_dentro, "%")
    print("Transformaciones fuera del rango recomendado:", porcentaje_fuera, "%")
    print("Porcentaje total de puntos fuera de los límites del lienzo:", porcentaje_fuera_rango_total, "%")
    print("número total de puntos analizados:", total_puntos_analizados)



# Función para calcular el error cuadrático medio (MSE)
def calcular_mse(poses_original, poses_transformadas):
    return F.mse_loss(poses_original, poses_transformadas)


# Función para calcular la correlación entre las poses
def calcular_correlacion(poses_original, poses_transformadas):
    # Aplanar las poses para calcular la correlación
    poses_original_flat = poses_original.view(poses_original.shape[0], -1)
    poses_transformadas_flat = poses_transformadas.view(poses_transformadas.shape[0], -1)
    # Calcular la matriz de correlación
    correlacion_matrix = torch.mm(poses_original_flat.t(), poses_transformadas_flat)
    # Calcular el coeficiente de correlación
    correlacion = torch.trace(correlacion_matrix) / (torch.norm(poses_original_flat) * torch.norm(poses_transformadas_flat))
    return correlacion.item()

def transformacion_zoom_validacion2(datos_entrenamiento, etiquetas_entrenamiento, intentos=100):
    puntos_totales = 100*75*2*intentos
    puntos_fuera_de_rango = 0
    
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
    rango_zoom_x = max(x_max - x_min, 1e-6)
    rango_zoom_y = max(y_max - y_min, 1e-6)
    
        
    rangomin = ((1 / (1 + 0.5 * max(rango_zoom_x, rango_zoom_y))))
    rangomax = ((1 + 0.5) * min(rango_zoom_x, rango_zoom_y))-0.1
    
    for _ in range(intentos):
        # Determinar aleatoriamente si se aplicará zoom in o zoom out

        # Calcular el rango de zoom permitido
        
        
        # Generar un factor de zoom aleatorio dentro del rango permitido
        factor_zoom = random.uniform(rangomin, rangomax)


        # Aplicar el zoom a las coordenadas x e y de cada punto de la pose
        for i in range(num_puntos):
            
            coordenadas_x = datos_entrenamiento_trans[:, i, 0::3]
            coordenadas_y = datos_entrenamiento_trans[:, i, 1::3]

            # Calcular el punto central de la pose
            centro_x = coordenadas_x.mean()
            centro_y = coordenadas_y.mean()

            # Aplicar el zoom
            nueva_x = (coordenadas_x - centro_x) * factor_zoom + centro_x
            nueva_y = (coordenadas_y - centro_y) * factor_zoom + centro_y
            # Actualizar las coordenadas en los datos de entrenamiento
            datos_entrenamiento_trans[:, i, 0::3] = nueva_x
            datos_entrenamiento_trans[:, i, 1::3] = nueva_y
            # Verificar si las coordenadas están dentro del rango [0, 1]
    if torch.any((datos_entrenamiento_trans < 0) | (datos_entrenamiento_trans > 1)):
        puntos_fuera_de_rango += torch.sum((datos_entrenamiento_trans < 0) | (datos_entrenamiento_trans > 1)).item()

    
    # Calcular el porcentaje de puntos fuera de rango
    porcentaje_fuera_de_rango = (puntos_fuera_de_rango / puntos_totales ) * 100 if puntos_totales > 0 else 0
    
    return puntos_totales, puntos_fuera_de_rango, porcentaje_fuera_de_rango

def validacionFlip(datos_entrenamiento, etiquetas_entrenamiento):
    # Inicializar contador para contar las transformaciones válidas
    transformaciones_validas = 0

    # Iterar sobre todos los datos de entrenamiento y aplicar las transformaciones
    for i in range(len(datos_entrenamiento)):
        datos_original = datos_entrenamiento[i]
        etiquetas_original = etiquetas_entrenamiento[i]

        # Aplicar la función de flip vertical
        datos_transformados, etiquetas_transformadas = flip_poses(datos_original, etiquetas_original)

        # Verificar si las etiquetas se invierten correctamente
        if torch.all(etiquetas_original == invertir_etiquetas(etiquetas_transformadas)):
            transformaciones_validas += 1

    # Calcular el porcentaje de transformaciones válidas
    porcentaje_valido = (transformaciones_validas / len(datos_entrenamiento)) * 100

    # Imprimir resultados
    print("Porcentaje de transformaciones válidas:", porcentaje_valido, "%")