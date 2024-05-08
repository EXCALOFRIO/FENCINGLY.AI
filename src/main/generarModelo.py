import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import optuna
from datetime import datetime
import os
import numpy as np
from tqdm import tqdm
from custom_transforms import transformacion_zoom, desplazar_posesX, desplazar_posesY, flip_poses
from CustomDataset import *

import torch.nn.functional as F

class Modelo(nn.Module):
    def __init__(self, lstm_units, dropout_rate_1, dropout_rate_2, dense_units, learning_rate, kernel_regularizer, num_lstm_layers, conv2d_out_channels, conv2d_kernel_size, attention_num_heads):
        super(Modelo, self).__init__()
        self.conv2d_left = nn.Conv2d(1, conv2d_out_channels, kernel_size=(conv2d_kernel_size, 1), padding=(conv2d_kernel_size//2, 0))
        self.conv2d_right = nn.Conv2d(1, conv2d_out_channels, kernel_size=(conv2d_kernel_size, 1), padding=(conv2d_kernel_size//2, 0))
        self.lstm_left = nn.LSTM(100 * 75, lstm_units, batch_first=True, num_layers=num_lstm_layers, bidirectional=True)
        self.lstm_right = nn.LSTM(100 * 75, lstm_units, batch_first=True, num_layers=num_lstm_layers, bidirectional=True)
        self.dropout1 = nn.Dropout(dropout_rate_1)
        self.attention_left = nn.MultiheadAttention(lstm_units * 2, num_heads=attention_num_heads, batch_first=True)
        self.attention_right = nn.MultiheadAttention(lstm_units * 2, num_heads=attention_num_heads, batch_first=True)
        self.lstm_combined = nn.LSTM(lstm_units * 4, lstm_units * 2, batch_first=True, num_layers=num_lstm_layers, bidirectional=True)
        self.dense1 = nn.Linear(lstm_units * 4, dense_units)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate_2)
        self.dense2 = nn.Linear(dense_units, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Verificar la forma de los datos de entrada
        assert x.dim() == 4, f"Expected input tensor to have 4 dimensions, but got {x.dim()}"
        assert x.size(1) == 100, f"Expected input tensor to have 100 frames, but got {x.size(1)}"
        assert x.size(2) == 2, f"Expected input tensor to have 2 poses, but got {x.size(2)}"
        assert x.size(3) == 75, f"Expected input tensor to have 75 keypoints, but got {x.size(3)}"

        # Separar los datos del tirador izquierdo y derecho
        left_data = x[:, :, 0, :].unsqueeze(1)
        right_data = x[:, :, 1, :].unsqueeze(1)
        #print("Left Data Shape:", left_data.shape)
        #print("Right Data Shape:", right_data.shape)

        # Aplicar convolución 2D y ReLU a cada tirador por separado
        left_conv = F.relu(self.conv2d_left(left_data))
        right_conv = F.relu(self.conv2d_right(right_data))
        #print("Left Conv Output Shape:", left_conv.shape)
        #print("Right Conv Output Shape:", right_conv.shape)

        # Redimensionar los tensores para aplicar LSTM
        left_conv = left_conv.permute(0, 3, 2, 1).contiguous().view(left_conv.size(0), -1, 100 * 75)
        right_conv = right_conv.permute(0, 3, 2, 1).contiguous().view(right_conv.size(0), -1, 100 * 75)
        #print("Left Conv Reshaped Shape:", left_conv.shape)
        #print("Right Conv Reshaped Shape:", right_conv.shape)

        left_lstm, _ = self.lstm_left(left_conv)
        right_lstm, _ = self.lstm_right(right_conv)
        #print("Left LSTM Output Shape:", left_lstm.shape)
        #print("Right LSTM Output Shape:", right_lstm.shape)

        # Aplicar atención temporal a cada tirador por separado
        left_attention, _ = self.attention_left(left_lstm, left_lstm, left_lstm)
        right_attention, _ = self.attention_right(right_lstm, right_lstm, right_lstm)
        #print("Left attention:", left_attention.shape)
        #print("Right attention:", right_attention.shape)

        # Concatenar las representaciones de ambos tiradores
        combined = torch.cat((left_attention, right_attention), dim=2)
        #print("Combined Shape:", combined.shape)

        # Aplicar LSTM a la combinación de representaciones
        combined_lstm, _ = self.lstm_combined(combined)
        #print("Combined LSTM Output Shape:", combined_lstm.shape)

        # Aplicar capas densas y funciones de activación
        x = self.dropout1(combined_lstm[:, -1, :])
        x = self.relu(self.dense1(x))
        x = self.dropout2(x)
        x = self.dense2(x)
        x = self.sigmoid(x)

        return x

def train_model(model, datos_entrenamiento, etiquetas_entrenamiento, hiperparametros, num_epochs, batch_size, patience=10, trial_number=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Dividir los datos en conjuntos de entrenamiento y validación
    X_train, X_val, y_train, y_val = train_test_split(datos_entrenamiento, etiquetas_entrenamiento, test_size=0.2, random_state=42)

    # Crear DataLoader personalizado para el conjunto de entrenamiento
    train_loader, val_loader = crear_dataloader(X_train, y_train, X_val, y_val, batch_size)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hiperparametros['learning_rate'])

    best_val_loss = float('inf')
    best_val_accuracy = 0.0
    epochs_without_improvement = 0
    
    # Crea un directorio para guardar los archivos de TensorBoard
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_dir = os.path.join("runs", f"{timestamp}_trial_{trial_number}")
    os.makedirs(tensorboard_dir, exist_ok=True)
    writer = SummaryWriter(tensorboard_dir)
    pbar = tqdm(range(num_epochs), desc="Training", unit="epoch")

    for epoch in pbar:
        # Entrenamiento
        model.train()
        train_loss = 0.0
        train_outputs = []
        train_targets = []
        for datos, etiquetas in train_loader:
            datos, etiquetas = datos.to(device), etiquetas.to(device)
            optimizer.zero_grad()
            outputs = model(datos)
            loss = criterion(outputs.squeeze(), etiquetas)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * datos.size(0)
            train_outputs.append(outputs.squeeze().detach().cpu().numpy())
            train_targets.append(etiquetas.detach().cpu().numpy())

        train_loss /= len(train_loader.dataset)
        train_outputs = np.concatenate(train_outputs)
        train_targets = np.concatenate(train_targets)
        train_auc = roc_auc_score(train_targets, train_outputs)

        # Registrar métricas en TensorBoard
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("AUC/train", train_auc, epoch)

        # Validación
        model.eval()
        val_loss = 0.0
        val_outputs = []
        val_targets = []
        with torch.no_grad():
            for datos_val, etiquetas_val in val_loader:
                datos_val, etiquetas_val = datos_val.to(device), etiquetas_val.to(device)
                outputs_val = model(datos_val)
                val_loss += criterion(outputs_val.squeeze(), etiquetas_val).item() * datos_val.size(0)
                val_outputs.append(outputs_val.squeeze().detach().cpu().numpy())
                val_targets.append(etiquetas_val.detach().cpu().numpy())

        val_loss /= len(val_loader.dataset)
        val_outputs = np.concatenate(val_outputs)
        val_targets = np.concatenate(val_targets)
        val_auc = roc_auc_score(val_targets, val_outputs)

        # Registrar métricas en TensorBoard
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("AUC/val", val_auc, epoch)

        # Mostrar métricas en la consola
        #tqdm.write(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}, Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")

        if val_auc > best_val_accuracy:
            best_val_accuracy = val_auc
            # Guardar el modelo con el mejor accuracy hasta ahora
            best_model = model.state_dict()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Restablecer el contador de épocas sin mejora en la pérdida
            epochs_without_improvement = 0
        else:
            # Incrementar el contador de epochs sin mejora en la pérdida
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            tqdm.write(f"Early stopping at epoch {epoch + 1} as there's no improvement in validation loss.")
            break

        # Actualizar el mensaje de tqdm con el mejor valor de val_accuracy hasta el momento
        pbar.set_postfix({"Best Val Accuracy": f"{best_val_accuracy:.8f}", "Best Val Loss": f"{best_val_loss:.8f}"})
    
    model.load_state_dict(best_model)
    return model, best_val_loss, best_val_accuracy

def optimize_hyperparameters(datos_entrenamiento, etiquetas_entrenamiento, datos_validacion, etiquetas_validacion, hiperparametros_ranges, num_trials, num_epochs, batch_size, patience=10):
    def objective(trial):
        hiperparametros = {
            'learning_rate': trial.suggest_float('learning_rate', *hiperparametros_ranges['learning_rate']),
            'lstm_units': trial.suggest_categorical('lstm_units', hiperparametros_ranges['lstm_units']),
            'dense_units': trial.suggest_categorical('dense_units', hiperparametros_ranges['dense_units']),
            'dropout_rate_1': trial.suggest_categorical('dropout_rate_1', hiperparametros_ranges['dropout_rate_1']),
            'dropout_rate_2': trial.suggest_categorical('dropout_rate_2', hiperparametros_ranges['dropout_rate_2']),
            'num_lstm_layers': trial.suggest_categorical('num_lstm_layers', hiperparametros_ranges['num_lstm_layers']),
            'kernel_regularizer': trial.suggest_float('kernel_regularizer', *hiperparametros_ranges['kernel_regularizer']),
            'conv2d_out_channels': trial.suggest_categorical('conv2d_out_channels', hiperparametros_ranges['conv2d_out_channels']),
            'conv2d_kernel_size': trial.suggest_categorical('conv2d_kernel_size', hiperparametros_ranges['conv2d_kernel_size']),
            'attention_num_heads': trial.suggest_categorical('attention_num_heads', hiperparametros_ranges['attention_num_heads']),
        }

        model = Modelo(**hiperparametros)
        
        _, _, val_accuracy = train_model(model, datos_entrenamiento, etiquetas_entrenamiento, hiperparametros, num_epochs, batch_size, patience, trial_number=trial.number)
        
        # Regresa también el número del trial
        return val_accuracy

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=num_trials)

    best_params = study.best_params
    best_model = Modelo(**best_params)
    best_model, best_val_loss, best_val_accuracy = train_model(best_model, datos_entrenamiento, etiquetas_entrenamiento, best_params, num_epochs, batch_size, patience)

    return best_model, best_val_loss, best_val_accuracy, best_params

