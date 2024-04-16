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
from CustomDataset import *

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


def train_with_optuna(datos_entrenamiento, etiquetas_entrenamiento, datos_validacion, etiquetas_validacion, num_trials=100, num_epochs=20, batch_size=32):
    input_shape = datos_entrenamiento.shape[-1]

    def objective(trial):
        learning_rate = trial.suggest_float('learning_rate', 1e-7, 1e-5, log=True)
        lstm_units = trial.suggest_int('lstm_units', 1100, 1300)
        dense_units = trial.suggest_int('dense_units', 1000, 1300)
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_params = {
            'input_size': input_shape * 2,
            'lstm_units': lstm_units,
            'dense_units': dense_units,
            'dropout_rate': dropout_rate,
            'learning_rate': learning_rate
        }

        train_loader, val_loader = crear_dataloader(datos_entrenamiento, etiquetas_entrenamiento, datos_validacion, etiquetas_validacion, batch_size)

        model = Modelo(**model_params).to(device)

        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        for epoch in tqdm(range(num_epochs), desc=f"Trial {trial.number}/ {num_trials}", unit="epoch"):
            # Training
            model.train()
            train_running_loss = 0.0
            train_correct = 0
            train_total = 0
            for datos, etiquetas in train_loader:
                datos, etiquetas = datos.to(device), etiquetas.to(device)
                datos_flat = datos.view(datos.shape[0], datos.shape[1], -1)

                optimizer.zero_grad()
                outputs = model(datos_flat)
                train_loss = criterion(outputs.squeeze(), etiquetas)
                train_loss.backward()
                optimizer.step()

                train_running_loss += train_loss.item() * datos.size(0)
                predicted = torch.round(outputs.squeeze().cpu())
                train_correct += (predicted == etiquetas.cpu()).sum().item()
                train_total += etiquetas.size(0)

            train_accuracy = train_correct / train_total

            # Validation
            model.eval()
            val_running_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for datos_val, etiquetas_val in val_loader:
                    datos_val, etiquetas_val = datos_val.to(device), etiquetas_val.to(device)
                    datos_val_flat = datos_val.view(datos_val.shape[0], datos_val.shape[1], -1)

                    val_outputs = model(datos_val_flat)
                    val_loss = criterion(val_outputs.squeeze(), etiquetas_val)

                    val_running_loss += val_loss.item() * datos_val.size(0)
                    val_predicted = torch.round(val_outputs.squeeze().cpu())
                    val_correct += (val_predicted == etiquetas_val.cpu()).sum().item()
                    val_total += etiquetas_val.size(0)

            val_accuracy = val_correct / val_total

            #tqdm.write(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_running_loss/train_total:.4f}, Train Acc: {train_accuracy:.4f}, Val Loss: {val_running_loss/val_total:.4f}, Val Acc: {val_accuracy:.4f}")

        return val_accuracy

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=num_trials)

    return study


def train_final_model(datos_entrenamiento, etiquetas_entrenamiento, datos_validacion, etiquetas_validacion, study, num_epochs=20, batch_size=32):
    best_params = study.best_params
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_shape = datos_entrenamiento.shape[-1]
    
    model = Modelo(input_size=input_shape*2, **best_params).to(device)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=best_params['learning_rate'])

    train_loader, val_loader = crear_dataloader(datos_entrenamiento, etiquetas_entrenamiento, datos_validacion, etiquetas_validacion, batch_size)
    
    for epoch in tqdm(range(num_epochs), desc="Final Training", unit="epoch"):
        # Entrenamiento
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for datos, etiquetas in train_loader:
            datos, etiquetas = datos.to(device), etiquetas.to(device)
            datos_flat = datos.view(datos.shape[0], datos.shape[1], -1)
            optimizer.zero_grad()
            outputs = model(datos_flat)
            loss = criterion(outputs.squeeze(), etiquetas)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * datos.size(0)
            predicted = torch.round(outputs.squeeze())
            train_correct += (predicted == etiquetas).sum().item()
            train_total += etiquetas.size(0)
            
        train_loss /= train_total
        train_accuracy = train_correct / train_total
        
        # Validación
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for datos_val, etiquetas_val in val_loader:
                datos_val, etiquetas_val = datos_val.to(device), etiquetas_val.to(device)
                datos_val_flat = datos_val.view(datos_val.shape[0], datos_val.shape[1], -1)
                outputs_val = model(datos_val_flat)
                val_loss += criterion(outputs_val.squeeze(), etiquetas_val).item() * datos_val.size(0)
                val_predicted = torch.round(outputs_val.squeeze())
                val_correct += (val_predicted == etiquetas_val).sum().item()
                val_total += etiquetas_val.size(0)
                
        val_loss /= val_total
        val_accuracy = val_correct / val_total
        
        tqdm.write(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

    return model

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