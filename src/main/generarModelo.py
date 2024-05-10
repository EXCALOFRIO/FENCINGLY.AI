import numpy as np
import optuna
import tensorflow as tf
from sklearn.model_selection import train_test_split
from CustomDataset import *
import datetime
import os

def entrenar_modelo(modelo, train_loader, val_loader, patience, batch_size, epochs, log_dir, trial):
    if trial is not None:
        log_dir_trial = os.path.join(log_dir, "{}-{}".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), trial.number))
        
    else:
        log_dir_trial = os.path.join(log_dir, "{}-{}".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), "FINAL"))
        patience = 100000
        
    train_dataset_size = train_loader.cardinality().numpy()
    val_dataset_size = val_loader.cardinality().numpy()
    
    steps_per_epoch = train_dataset_size // batch_size if train_dataset_size > 0 else None
    validation_steps = val_dataset_size // batch_size if val_dataset_size > 0 else None
    
    # Definir el callback de TensorBoard
    
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir_trial, histogram_freq=1)
    
    modelo.fit(train_loader,
               epochs=epochs,
               steps_per_epoch=steps_per_epoch,
               validation_data=val_loader,
               validation_steps=validation_steps,
               callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience), tensorboard_callback])

def objetivo(trial, train_loader, val_loader, val_earlystopping_loader, hiperparametros_ranges, patience, batch_size, epochs, log_dir):
    params = {}
    for key, value in hiperparametros_ranges.items():
        if isinstance(value, list):
            params[key] = trial.suggest_categorical(key, value)
        elif isinstance(value, tuple) and len(value) == 2:
            params[key] = trial.suggest_float(key, value[0], value[1])
        else:
            raise ValueError("Invalid range format for parameter '{}': {}".format(key, value))

    modelo = tf.keras.Sequential([
        tf.keras.layers.Reshape((-1, 150), input_shape=(100, 2, 75)),
        tf.keras.layers.Masking(mask_value=0.),
        tf.keras.layers.LSTM(params['lstm_units'], return_sequences=True),
        tf.keras.layers.Dropout(params['dropout_rate']),
        tf.keras.layers.LSTM(params['lstm_units'] // 2),
        tf.keras.layers.Dropout(params['dropout_rate']),
        tf.keras.layers.Dense(params['dense_units'], activation='relu',
                            kernel_regularizer=tf.keras.regularizers.l2(params['kernel_regularizer'])),
        tf.keras.layers.Dropout(params['dropout_rate']),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    opt = tf.keras.optimizers.Adam(learning_rate=params['learning_rate'])

    modelo.compile(
        optimizer=opt,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    entrenar_modelo(modelo, train_loader, val_earlystopping_loader, patience, batch_size, epochs, log_dir=log_dir, trial=trial)

    return modelo.evaluate(val_loader)[1]

def optimizar_hiperparametros(datos_entrenamiento, etiquetas_entrenamiento, datos_validacion, etiquetas_validacion,
                              hiperparametros_ranges, patience, batch_size, epochs, num_trials=100, log_dir="logs"):
    
    datos_entrenamiento = np.array(datos_entrenamiento)
    etiquetas_entrenamiento = np.array(etiquetas_entrenamiento)
    
    datos_entrenamiento_early_stop, _, etiquetas_entrenamiento_early_stop, _ = train_test_split(
        datos_entrenamiento, etiquetas_entrenamiento, test_size=0.8, random_state=42)
       
    print('Creando dataloaders...')
    print('Datos entrenamiento:', datos_entrenamiento.shape)
    print('Datos validación:', datos_validacion.shape)
    print('Datos entrenamiento early stopping:', datos_entrenamiento_early_stop.shape)

    train_loader, val_loader, val_earlystopping_loader = crear_dataloader(datos_entrenamiento, etiquetas_entrenamiento, datos_validacion, etiquetas_validacion, datos_entrenamiento_early_stop, etiquetas_entrenamiento_early_stop, batch_size)

    
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objetivo(trial, train_loader, val_loader, val_earlystopping_loader, hiperparametros_ranges, patience, batch_size, epochs, log_dir=log_dir),
                   n_trials=num_trials)

    print('Mejores hiperparámetros:', study.best_params)
    print('Mejor precisión de validación:', study.best_value)

    best_params = study.best_params

    modelo_final = tf.keras.Sequential([
        tf.keras.layers.Reshape((-1, 150), input_shape=(100, 2, 75)),
        tf.keras.layers.Masking(mask_value=0.),
        tf.keras.layers.LSTM(best_params['lstm_units'], return_sequences=True),
        tf.keras.layers.Dropout(best_params['dropout_rate']),
        tf.keras.layers.LSTM(best_params['lstm_units'] // 2),
        tf.keras.layers.Dropout(best_params['dropout_rate']),
        tf.keras.layers.Dense(best_params['dense_units'], activation='relu',
                              kernel_regularizer=tf.keras.regularizers.l2(best_params['kernel_regularizer'])),
        tf.keras.layers.Dropout(best_params['dropout_rate']),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    opt_final = tf.keras.optimizers.Adam(learning_rate=best_params['learning_rate'])

    modelo_final.compile(
        optimizer=opt_final,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    entrenar_modelo(modelo_final, train_loader, val_loader, patience, batch_size, epochs, log_dir, trial=None)

    return modelo_final
