import numpy as np
from sklearn.model_selection import train_test_split
from dataset import PosesDataset
from torchvision import transforms
import torch

# Directorios donde se encuentran los datos
izquierda_dir = r"C:/Users/Alejandro/TFG_ENTRENAMIENTO/JSON_FORMADOS/IZQUIERDA_NORMALIZADO"
derecha_dir = r"C:/Users/Alejandro/TFG_ENTRENAMIENTO/JSON_FORMADOS/DERECHA_NORMALIZADO"

# Cargar los datos usando PyTorch Dataset
dataset_izquierda = PosesDataset(izquierda_dir)
dataset_derecha = PosesDataset(derecha_dir)

# Dividir datos en conjuntos de entrenamiento, validación y prueba
porcentaje_entrenamiento = 0.70
porcentaje_validacion = 0.15
porcentaje_prueba = 0.15

# Convertir a arrays numpy
datos_entrenamiento_izquierda, datos_validacion_izquierda, _, _ = train_test_split(dataset_izquierda.data, np.ones(len(dataset_izquierda)), test_size=1 - porcentaje_entrenamiento)
datos_validacion_izquierda, datos_prueba_izquierda, _, _ = train_test_split(datos_validacion_izquierda, np.ones(len(datos_validacion_izquierda)), test_size=porcentaje_prueba / (porcentaje_validacion + porcentaje_prueba))

datos_entrenamiento_derecha, datos_validacion_derecha, _, _ = train_test_split(dataset_derecha.data, np.zeros(len(dataset_derecha)), test_size=1 - porcentaje_entrenamiento)
datos_validacion_derecha, datos_prueba_derecha, _, _ = train_test_split(datos_validacion_derecha, np.zeros(len(datos_validacion_derecha)), test_size=porcentaje_prueba / (porcentaje_validacion + porcentaje_prueba))

datos_entrenamiento = np.concatenate((datos_entrenamiento_izquierda, datos_entrenamiento_derecha))
datos_validacion = np.concatenate((datos_validacion_izquierda, datos_validacion_derecha))
datos_prueba = np.concatenate((datos_prueba_izquierda, datos_prueba_derecha))

etiquetas_entrenamiento = np.concatenate((np.ones(len(datos_entrenamiento_izquierda)), np.zeros(len(datos_entrenamiento_derecha))))
etiquetas_validacion = np.concatenate((np.ones(len(datos_validacion_izquierda)), np.zeros(len(datos_validacion_derecha))))
etiquetas_prueba = np.concatenate((np.ones(len(datos_prueba_izquierda)), np.zeros(len(datos_prueba_derecha))))

# Convertir a tensores PyTorch
datos_entrenamiento = torch.tensor(datos_entrenamiento, dtype=torch.float32)
datos_validacion = torch.tensor(datos_validacion, dtype=torch.float32)
datos_prueba = torch.tensor(datos_prueba, dtype=torch.float32)
etiquetas_entrenamiento = torch.tensor(etiquetas_entrenamiento, dtype=torch.float32)
etiquetas_validacion = torch.tensor(etiquetas_validacion, dtype=torch.float32)
etiquetas_prueba = torch.tensor(etiquetas_prueba, dtype=torch.float32)

transform = transforms.Compose([
    RandomTranslation(),
    RandomZoom(),
    HorizontalFlip()
])

datos_entrenamiento_transformados = []
etiquetas_entrenamiento_transformadas = []
for dato, etiqueta in zip(datos_entrenamiento, etiquetas_entrenamiento):
    dato_transformado, etiqueta_transformada = transform(dato, etiqueta)
    datos_entrenamiento_transformados.append(dato_transformado)
    etiquetas_entrenamiento_transformadas.append(etiqueta_transformada)

# Aplicar transformaciones a los conjuntos de datos de validación y prueba
datos_validacion_transformados = []
etiquetas_validacion_transformadas = []
for dato, etiqueta in zip(datos_validacion, etiquetas_validacion):
    dato_transformado, etiqueta_transformada = transform(dato, etiqueta)
    datos_validacion_transformados.append(dato_transformado)
    etiquetas_validacion_transformadas.append(etiqueta_transformada)

datos_prueba_transformados = []
etiquetas_prueba_transformadas = []
for dato, etiqueta in zip(datos_prueba, etiquetas_prueba):
    dato_transformado, etiqueta_transformada = transform(dato, etiqueta)
    datos_prueba_transformados.append(dato_transformado)
    etiquetas_prueba_transformadas.append(etiqueta_transformada)
    
# Eliminar variables innecesarias
del datos_entrenamiento_izquierda
del datos_entrenamiento_derecha
del datos_validacion_izquierda
del datos_validacion_derecha
del datos_prueba_izquierda
del datos_prueba_derecha