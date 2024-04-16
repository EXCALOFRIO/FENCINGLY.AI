import torch
from torch.utils.data import Dataset, DataLoader
from custom_transforms import transformacion_zoom, desplazar_posesX, desplazar_posesY, flip_poses
import random

class CustomDataset(Dataset):
    def __init__(self, datos, etiquetas, input_shape):
        self.datos = datos
        self.etiquetas = etiquetas
        self.input_shape = input_shape

    def __len__(self):
        return len(self.datos)

    def __getitem__(self, idx):
        datos = self.datos[idx]
        etiquetas = self.etiquetas[idx]
        datos, etiquetas = self.transformar_datos(datos, etiquetas)  # Llamada a transformar_datos
        return datos, etiquetas

    def transformar_datos(self, datos, etiquetas):
        # Lista de transformaciones disponibles
        transformaciones = [transformacion_zoom, desplazar_posesX, desplazar_posesY, flip_poses]

        # Aplicar transformaciones aleatorias
        for transformacion in transformaciones:
            if random.random() > 0.5:  # Probabilidad de aplicar la transformación: 50%
                datos, etiquetas = transformacion(datos, etiquetas)

        return datos, etiquetas

# Función para crear los DataLoader
def crear_dataloader(datos_entrenamiento, etiquetas_entrenamiento, datos_validacion, etiquetas_validacion, batch_size):
    input_shape = datos_entrenamiento.shape[-1]
    dataset_entrenamiento = CustomDataset(datos_entrenamiento, etiquetas_entrenamiento, input_shape)
    dataset_validacion = CustomDataset(datos_validacion, etiquetas_validacion, input_shape)

    train_loader = DataLoader(dataset_entrenamiento, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset_validacion, batch_size=batch_size)

    return train_loader, val_loader