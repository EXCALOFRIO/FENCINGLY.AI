import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import optuna
from datetime import datetime
import os
import numpy as np
from tqdm import tqdm
from custom_transforms import transformacion_zoom, desplazar_posesX, desplazar_posesY, flip_poses

class Modelo(nn.Module):
    def __init__(self, input_size, lstm_units, dense_units, dropout_rate, learning_rate):
        super(Modelo, self).__init__()
        self.lstm = nn.LSTM(input_size, lstm_units, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(lstm_units, dense_units)
        self.fc2 = nn.Linear(dense_units, 1)
        self.learning_rate = learning_rate

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        x = self.dropout(hidden[-1])
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x


def transformar_datos(datos_entrenamiento, etiquetas_entrenamiento):
    # Aquí aplicamos todas las transformaciones necesarias
    datos_entrenamiento, etiquetas_entrenamiento = transformacion_zoom(datos_entrenamiento, etiquetas_entrenamiento)
    datos_entrenamiento, etiquetas_entrenamiento = desplazar_posesX(datos_entrenamiento, etiquetas_entrenamiento)
    datos_entrenamiento, etiquetas_entrenamiento = desplazar_posesY(datos_entrenamiento, etiquetas_entrenamiento)
    datos_entrenamiento, etiquetas_entrenamiento = flip_poses(datos_entrenamiento, 1-etiquetas_entrenamiento)
    
    return datos_entrenamiento, etiquetas_entrenamiento


def train_with_optuna(train_loader, val_loader, num_trials=100, num_epochs=20, batch_size=32):
    def objective(trial):
        learning_rate = trial.suggest_float('learning_rate', 1e-7, 1e-5, log=True)
        lstm_units = trial.suggest_int('lstm_units', 1100, 1300)
        dense_units = trial.suggest_int('dense_units', 1000, 1300)
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        model_params = {
            'input_size': train_loader.dataset.tensors[0].shape[-1]*2,
            'lstm_units': lstm_units,
            'dense_units': dense_units,
            'dropout_rate': dropout_rate,
            'learning_rate': learning_rate
        }
        
        model = Modelo(**model_params).to(device)
        
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        model.train()
        for epoch in tqdm(range(num_epochs), desc=f"Trial {trial.number}", unit="epoch"):
            running_loss = 0.0
            correct = 0
            total = 0
            for datos, etiquetas in train_loader:
                datos_transformados, etiquetas_transformadas = transformar_datos(datos, etiquetas)
                datos_transformados, etiquetas_transformadas = datos_transformados.to(device), etiquetas_transformadas.to(device)
                datos_transformados_flat = datos_transformados.view(datos_transformados.shape[0], datos_transformados.shape[1], -1)

                optimizer.zero_grad()
                outputs = model(datos_transformados_flat)
                loss = criterion(outputs.squeeze(), etiquetas_transformadas)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * datos_transformados.size(0)  # Fix here
                predicted = torch.round(outputs.squeeze().cpu())  # Fix here
                correct += (predicted == etiquetas).sum().item()
                total += etiquetas.size(0)
            
        accuracy = correct / total
        
        return accuracy

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=num_trials)
    
    return study


def train_final_model(train_loader, val_loader, study, transformaciones, num_epochs=20, batch_size=32):
    best_params = study.best_params
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = Modelo(input_size=train_loader.dataset.tensors[0].shape[-1], **best_params).to(device)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=best_params['learning_rate'])
    
    for epoch in tqdm(range(num_epochs), desc="Final Training", unit="epoch"):
        for datos, etiquetas in train_loader:
            # Aplicar transformaciones
            datos_transformados, etiquetas_transformadas = transformar_datos(datos, etiquetas)
            
            # Mover los datos transformados al dispositivo
            datos_transformados, etiquetas_transformadas = datos_transformados.to(device), etiquetas_transformadas.to(device)
            datos_transformados_flat = datos_transformados.view(datos_transformados.shape[0], datos_transformados.shape[1], -1)
            optimizer.zero_grad()
            outputs = model(datos_transformados_flat)
            loss = criterion(outputs.squeeze(), etiquetas_transformadas)
            loss.backward()
            optimizer.step()

    # Evaluar el modelo final en el conjunto de validación
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for datos, etiquetas in val_loader:
            datos, etiquetas = datos.to(device), etiquetas.to(device)
            outputs = model(datos)
            predicted = torch.round(outputs.squeeze())
            correct += (predicted == etiquetas).sum().item()
            total += etiquetas.size(0)

    accuracy = correct / total
    
    return model, accuracy


def preparar_entrenamiento(datos_entrenamiento, etiquetas_entrenamiento, datos_validacion, etiquetas_validacion, batch_size):
    # Convertir los datos a tensores de PyTorch
    datos_entrenamiento_tensor = torch.tensor(datos_entrenamiento, dtype=torch.float32)
    etiquetas_entrenamiento_tensor = torch.tensor(etiquetas_entrenamiento, dtype=torch.float32)
    datos_validacion_tensor = torch.tensor(datos_validacion, dtype=torch.float32)
    etiquetas_validacion_tensor = torch.tensor(etiquetas_validacion, dtype=torch.float32)

    # Crear datasets
    dataset_entrenamiento = TensorDataset(datos_entrenamiento_tensor, etiquetas_entrenamiento_tensor)
    dataset_validacion = TensorDataset(datos_validacion_tensor, etiquetas_validacion_tensor)

    # Crear DataLoader para el conjunto de entrenamiento
    train_loader = DataLoader(dataset_entrenamiento, batch_size=batch_size, shuffle=True)

    # Crear DataLoader para el conjunto de validación
    val_loader = DataLoader(dataset_validacion, batch_size=batch_size)
    
    return train_loader, val_loader