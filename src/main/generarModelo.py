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


def train_model(X_train, y_train, X_val, y_val, model_params, num_epochs=20, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_dataset = TensorDataset(X_train.view(X_train.shape[0], -1, X_train.shape[-1]), y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = TensorDataset(X_val.view(X_val.shape[0], -1, X_val.shape[-1]), y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    model = Modelo(**model_params).to(device)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=model_params['learning_rate'])
    
    writer = SummaryWriter()
    
    model.train()
    for epoch in tqdm(range(num_epochs), desc="Training", unit="epoch"):
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            predicted = torch.round(outputs.squeeze())
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_accuracy = correct / total
        
        writer.add_scalar('Train/Loss', epoch_loss, epoch)
        writer.add_scalar('Train/Accuracy', epoch_accuracy, epoch)
    
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predicted = torch.round(outputs.squeeze())
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        
        accuracy = correct / total
        writer.add_scalar('Validation/Accuracy', accuracy, epoch)
    
    writer.close()
    
    return model, accuracy

def entrenar_con_optuna(train_loader, val_loader, n_trials=100, num_epochs=20, patience=5):
    def objective(trial):
        learning_rate = trial.suggest_float('learning_rate', 1e-7, 1e-5, log=True)
        lstm_units = trial.suggest_int('lstm_units', 1100, 1300)
        dense_units = trial.suggest_int('dense_units', 1000, 1300)
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        model = Modelo(input_size=train_loader.dataset.tensors[0].shape[-1], lstm_units=lstm_units, dense_units=dense_units, dropout_rate=dropout_rate, learning_rate=learning_rate).to(device)
        
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        log_dir = f"./logs/optuna/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        writer = SummaryWriter(log_dir=log_dir)
        
        model.train()
        for epoch in tqdm(range(num_epochs), desc=f"Trial {trial.number}", unit="epoch"):
            running_loss = 0.0
            correct = 0
            total = 0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                predicted = torch.round(outputs.squeeze())
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
            
            epoch_loss = running_loss / len(train_loader.dataset)
            epoch_accuracy = correct / total
            
            writer.add_scalar('Train/Loss', epoch_loss, epoch)
            writer.add_scalar('Train/Accuracy', epoch_accuracy, epoch)
        
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                predicted = torch.round(outputs.squeeze())
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
            
            accuracy = correct / total
            writer.add_scalar('Validation/Accuracy', accuracy, epoch)
        
        writer.close()
        
        return accuracy

    
    study = optuna.create_study(direction='maximize')
    early_stopping_counter = 0
    best_score = -np.inf
    
    def callback(study, trial):
        nonlocal early_stopping_counter, best_score
        # Obtiene la mejor puntuación hasta el momento
        best_score = study.best_value
        if study.best_trial.number != trial.number:
            # Si no es el mejor ensayo actual, incrementa el contador
            early_stopping_counter += 1
        else:
            # Reinicia el contador si se encuentra un nuevo mejor ensayo
            early_stopping_counter = 0

        if early_stopping_counter >= patience:
            # Si se supera la paciencia, detén el estudio
            study.stop()

    study.optimize(objective, n_trials=n_trials, callbacks=[callback])
    
    log_dir = "./logs/optuna"
    guardar_logs(study, log_dir)

    return study


def entrenar_modelo(train_loader, test_loader, best_params, num_epochs=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = Modelo(input_size=train_loader.dataset.tensors[0].shape[-1], **best_params).to(device)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=best_params['learning_rate'])
    
    log_dir = f"./logs/fino/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=log_dir)
    
    model.train()
    for epoch in tqdm(range(num_epochs), desc="Final Training", unit="epoch"):
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            predicted = torch.round(outputs.squeeze())
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_accuracy = correct / total
        
        writer.add_scalar('Train/Loss', epoch_loss, epoch)
        writer.add_scalar('Train/Accuracy', epoch_accuracy, epoch)
    
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predicted = torch.round(outputs.squeeze())
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        
        accuracy = correct / total
        print(f"Accuracy on test data: {accuracy:.16f}")

        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predicted = torch.round(outputs.squeeze())
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        
        accuracy = correct / total
        print(f"Accuracy on train data: {accuracy:.16f}")

    writer.close()
    
    return model, accuracy


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = Modelo(input_size=train_loader.dataset.tensors[0].shape[-1], **best_params).to(device)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=best_params['learning_rate'])
    
    writer = SummaryWriter()
    
    model.train()
    for epoch in tqdm(range(num_epochs), desc="Final Training", unit="epoch"):
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            predicted = torch.round(outputs.squeeze())
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_accuracy = correct / total
        
        writer.add_scalar('Train/Loss', epoch_loss, epoch)
        writer.add_scalar('Train/Accuracy', epoch_accuracy, epoch)
    
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predicted = torch.round(outputs.squeeze())
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        
        accuracy = correct / total
        print(f"Accuracy on test data: {accuracy:.16f}")

        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predicted = torch.round(outputs.squeeze())
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        
        accuracy = correct / total
        print(f"Accuracy on train data: {accuracy:.16f}")

        
    
    writer.close()
    
    return model, accuracy

def guardar_logs(study, log_dir):
    
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    os.makedirs(log_dir, exist_ok=True)
    
def preparar_entrenamiento(datos_entrenamiento, etiquetas_entrenamiento, datos_validacion, etiquetas_validacion, batch_size):
    # Aplanar los datos de entrenamiento y validación
    datos_entrenamiento_flat = datos_entrenamiento.view(datos_entrenamiento.shape[0], datos_entrenamiento.shape[1], -1)
    datos_validacion_flat = datos_validacion.view(datos_validacion.shape[0], datos_validacion.shape[1], -1)

    # Convertir los datos a tensores de PyTorch
    datos_entrenamiento_tensor = torch.tensor(datos_entrenamiento_flat, dtype=torch.float32)
    etiquetas_entrenamiento_tensor = torch.tensor(etiquetas_entrenamiento, dtype=torch.float32)
    datos_validacion_tensor = torch.tensor(datos_validacion_flat, dtype=torch.float32)
    etiquetas_validacion_tensor = torch.tensor(etiquetas_validacion, dtype=torch.float32)

    # Crear datasets
    dataset_entrenamiento = TensorDataset(datos_entrenamiento_tensor, etiquetas_entrenamiento_tensor)
    dataset_validacion = TensorDataset(datos_validacion_tensor, etiquetas_validacion_tensor)

    # Crear DataLoader para el conjunto de entrenamiento
    train_loader = DataLoader(dataset_entrenamiento, batch_size=batch_size, shuffle=True)

    # Crear DataLoader para el conjunto de validación
    val_loader = DataLoader(dataset_validacion, batch_size=batch_size)
    
    return train_loader, val_loader
