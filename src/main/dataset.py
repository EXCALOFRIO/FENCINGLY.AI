import os
import json
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import tensorflow as tf

class PosesDataset:
    def __init__(self, directory, num_frames=100):
        self.data = self.load_data(directory, num_frames)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def load_data(self, directory, num_frames=100):
        data = []
        with ThreadPoolExecutor() as executor:
            for archivo_json in tqdm(os.listdir(directory), desc="Loading data"):
                ruta_json = os.path.join(directory, archivo_json)
                with open(ruta_json) as f:
                    datos_json = json.load(f)
                    sequence = np.zeros((num_frames, 2, 75))  # Initialize matrix for the sequence
                    for i, frame in enumerate(datos_json['frames'][:num_frames]):
                        poses = frame['poses'][:2]  # Take only the first 2 poses
                        for j, pose in enumerate(poses):
                            keypoints = pose['pose_keypoints_2d'][:75]  # Take only the first 75 keypoints
                            sequence[i, j, :] = keypoints
                    data.append(sequence)
        return data
        
def cargar_datos(izquierda_dir, derecha_dir):
    dataset_izquierda = PosesDataset(izquierda_dir)
    dataset_derecha = PosesDataset(derecha_dir)
    return dataset_izquierda, dataset_derecha
    
def dividir_datos_entrenamiento_validacion_prueba(dataset, porcentaje_entrenamiento, porcentaje_validacion, porcentaje_prueba):
    datos = np.array(dataset.data)
    datos_entrenamiento, datos_prueba = train_test_split(datos, test_size=1 - porcentaje_entrenamiento, random_state=42)
    datos_validacion, datos_prueba = train_test_split(datos_prueba, test_size=porcentaje_prueba / (porcentaje_validacion + porcentaje_prueba), random_state=42)
    return datos_entrenamiento, datos_validacion, datos_prueba
    
def preparar_datos(datos_entrenamiento_izquierda, datos_entrenamiento_derecha, datos_validacion_izquierda, datos_validacion_derecha, datos_prueba_izquierda, datos_prueba_derecha):
    datos_entrenamiento = np.concatenate((datos_entrenamiento_izquierda, datos_entrenamiento_derecha))
    datos_validacion = np.concatenate((datos_validacion_izquierda, datos_validacion_derecha))
    datos_prueba = np.concatenate((datos_prueba_izquierda, datos_prueba_derecha))
    etiquetas_entrenamiento = np.concatenate((np.ones(len(datos_entrenamiento_izquierda)), np.zeros(len(datos_entrenamiento_derecha))))
    etiquetas_validacion = np.concatenate((np.ones(len(datos_validacion_izquierda)), np.zeros(len(datos_validacion_derecha))))
    etiquetas_prueba = np.concatenate((np.ones(len(datos_prueba_izquierda)), np.zeros(len(datos_prueba_derecha))))
    return datos_entrenamiento, datos_validacion, datos_prueba, etiquetas_entrenamiento, etiquetas_validacion, etiquetas_prueba
    
def convertir_a_tensores(datos_entrenamiento, datos_validacion, datos_prueba, etiquetas_entrenamiento, etiquetas_validacion, etiquetas_prueba):
    datos_entrenamiento = tf.convert_to_tensor(datos_entrenamiento, dtype=tf.float32)
    datos_validacion = tf.convert_to_tensor(datos_validacion, dtype=tf.float32)
    datos_prueba = tf.convert_to_tensor(datos_prueba, dtype=tf.float32)
    etiquetas_entrenamiento = tf.convert_to_tensor(etiquetas_entrenamiento, dtype=tf.float32)
    etiquetas_validacion = tf.convert_to_tensor(etiquetas_validacion, dtype=tf.float32)
    etiquetas_prueba = tf.convert_to_tensor(etiquetas_prueba, dtype=tf.float32)
    return datos_entrenamiento, datos_validacion, datos_prueba, etiquetas_entrenamiento, etiquetas_validacion, etiquetas_prueba
    
def cargar_y_preparar_datos(izquierda_dir, derecha_dir, porcentaje_entrenamiento, porcentaje_validacion, porcentaje_prueba):
    # Cargar datos
    dataset_izquierda, dataset_derecha = cargar_datos(izquierda_dir, derecha_dir)
    
    # Dividir datos
    datos_entrenamiento_izquierda, datos_validacion_izquierda, datos_prueba_izquierda = dividir_datos_entrenamiento_validacion_prueba(dataset_izquierda, porcentaje_entrenamiento, porcentaje_validacion, porcentaje_prueba)
    datos_entrenamiento_derecha, datos_validacion_derecha, datos_prueba_derecha = dividir_datos_entrenamiento_validacion_prueba(dataset_derecha, porcentaje_entrenamiento, porcentaje_validacion, porcentaje_prueba)
    
    # Preparar datos
    datos_entrenamiento, datos_validacion, datos_prueba, etiquetas_entrenamiento, etiquetas_validacion, etiquetas_prueba = preparar_datos(datos_entrenamiento_izquierda, datos_entrenamiento_derecha, datos_validacion_izquierda, datos_validacion_derecha, datos_prueba_izquierda, datos_prueba_derecha)
    
    # Normalizar datos
    max_valor = np.max(datos_entrenamiento)  # Calcular el máximo valor en los datos de entrenamiento
    datos_entrenamiento = datos_entrenamiento / max_valor
    datos_validacion = datos_validacion / max_valor
    datos_prueba = datos_prueba / max_valor
    
    # Convertir a tensores
    datos_entrenamiento, datos_validacion, datos_prueba, etiquetas_entrenamiento, etiquetas_validacion, etiquetas_prueba = convertir_a_tensores(datos_entrenamiento, datos_validacion, datos_prueba, etiquetas_entrenamiento, etiquetas_validacion, etiquetas_prueba)
    
    return datos_entrenamiento, datos_validacion, datos_prueba, etiquetas_entrenamiento, etiquetas_validacion, etiquetas_prueba