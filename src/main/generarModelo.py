import numpy as np
import optuna
import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from CustomDataset import *
from CustomDataset2 import *
import datetime
import os
from tqdm import tqdm
from tensorflow.keras import metrics
from tensorflow.keras.layers import LeakyReLU
from sklearn.metrics import f1_score


class CustomEarlyStopping(tf.keras.callbacks.Callback):
    def __init__(self, patience_accuracy=10, patience_loss=5):
        super(CustomEarlyStopping, self).__init__()
        self.patience_accuracy = patience_accuracy
        self.patience_loss = patience_loss
        self.best_accuracy = 0
        self.best_loss = np.Inf
        self.wait_accuracy = 0
        self.wait_loss = 0

    def on_epoch_end(self, epoch, logs=None):
        current_accuracy = logs.get('val_accuracy')
        current_loss = logs.get('val_loss')

        if current_accuracy > self.best_accuracy:
            self.best_accuracy = current_accuracy
            self.wait_accuracy = 0
        else:
            self.wait_accuracy += 1

        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.wait_loss = 0
        else:
            self.wait_loss += 1

        if self.wait_accuracy >= self.patience_accuracy or self.wait_loss >= self.patience_loss:
            self.model.stop_training = True

def entrenar_modelo(modelo, train_generator, val_generator, patience_accuracy, patience_loss, epochs, log_dir, trial):
    log_dir_trial = os.path.join(log_dir, "{}-{}".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), trial.number if trial is not None else "FINAL"))

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir_trial, histogram_freq=1)
    early_stopping_callback = CustomEarlyStopping(patience_accuracy=patience_accuracy, patience_loss=patience_loss)

    history = modelo.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        callbacks=[early_stopping_callback, tensorboard_callback]
    )

    return history

def crear_modelo(params):
    modelo = tf.keras.Sequential([
        tf.keras.layers.Reshape((-1, 200), input_shape=(200, 2, 50)),
        tf.keras.layers.Masking(mask_value=0.),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(params['lstm_units'], return_sequences=True, kernel_regularizer=get_regularizer(params['regularization_type'], params['regularization']))),
        tf.keras.layers.Dropout(params['dropout_rate_1']),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(params['lstm_units'], return_sequences=True, kernel_regularizer=get_regularizer(params['regularization_type'], params['regularization']))),
        tf.keras.layers.Dropout(params['dropout_rate_2']),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(params['lstm_units'], kernel_regularizer=get_regularizer(params['regularization_type'], params['regularization']))),
        tf.keras.layers.Dropout(params['dropout_rate_3']),
        tf.keras.layers.Dense(params['dense_units'], activation=params['activation'] if params['activation'] != 'leaky_relu' else LeakyReLU(alpha=0.01), kernel_regularizer=get_regularizer(params['regularization_type'], params['regularization'])),
        tf.keras.layers.Dropout(params['dropout_rate_4']),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    if params['optimizer'] == 'Adam':
        opt = tf.keras.optimizers.Adam(learning_rate=params['learning_rate'])
    elif params['optimizer'] == 'SGD':
        opt = tf.keras.optimizers.SGD(learning_rate=params['learning_rate'])

    modelo.compile(
        optimizer=opt,
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc'), F1Score()]
    )

    return modelo

def get_regularizer(regularization_type, regularization_rate):
    if regularization_type == 'l1':
        return tf.keras.regularizers.l1(regularization_rate)
    elif regularization_type == 'l2':
        return tf.keras.regularizers.l2(regularization_rate)
    else:
        raise ValueError("Invalid regularization type: {}".format(regularization_type))

def objetivo(trial, datos_entrenamiento, etiquetas_entrenamiento, datos_validacion, etiquetas_validacion, hiperparametros_ranges, patience_accuracy, patience_loss, batch_size, epochs, log_dir):
    params = {}
    for key, value in hiperparametros_ranges.items():
        if key == 'learning_rate':
            params[key] = trial.suggest_float(key, value[0], value[1], log=True)
        elif isinstance(value, list):
            params[key] = trial.suggest_categorical(key, value)
        elif isinstance(value, tuple) and len(value) == 2:
            params[key] = trial.suggest_float(key, value[0], value[1])
        else:
            raise ValueError("Invalid range format for parameter '{}': {}".format(key, value))

    # Add transformation_prob as a suggested parameter
    params['transformation_prob'] = trial.suggest_float('transformation_prob', 0.0, 0.5)

    modelo = crear_modelo(params)

    # Ensure that transformation_prob is used consistently
    train_generator, val_generator = crear_dataloader2(datos_entrenamiento, etiquetas_entrenamiento, datos_validacion, etiquetas_validacion, batch_size, transformation_prob=params['transformation_prob'])

    history = entrenar_modelo(modelo, train_generator, val_generator, patience_accuracy, patience_loss, epochs, log_dir=log_dir, trial=trial)

    val_accuracy = history.history.get('val_accuracy', [0])[-1]
    val_auc = history.history.get('val_auc', [0])[-1]
    val_f1 = history.history.get('val_f1_score', [0])[-1]  # Añadir el f1_score

    return val_accuracy  # Usar F1 score para la optimización de hiperparámetros


def optimizar_hiperparametros(datos_entrenamiento, etiquetas_entrenamiento, datos_validacion, etiquetas_validacion, hiperparametros_ranges, patience_accuracy, patience_loss, batch_size, epochs, num_trials=100, log_dir="logs"):
    print('Datos entrenamiento:', datos_entrenamiento.shape)
    print('Datos validación:', datos_validacion.shape)

    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objetivo(trial, datos_entrenamiento, etiquetas_entrenamiento, datos_validacion, etiquetas_validacion, hiperparametros_ranges, patience_accuracy, patience_loss, batch_size, epochs, log_dir=log_dir), n_trials=num_trials)

    print('Mejores hiperparámetros:', study.best_params)
    print('Mejor F1 score de validación:', study.best_value)

    best_params = study.best_params

    modelo_final = crear_modelo(best_params)

    train_generator, val_generator = crear_dataloader2(datos_entrenamiento, etiquetas_entrenamiento, datos_validacion, etiquetas_validacion, batch_size, transformation_prob=best_params['transformation_prob'])

    entrenar_modelo(modelo_final, train_generator, val_generator, patience_accuracy, patience_loss, epochs, log_dir, trial=None)

    return modelo_final


class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.false_positives = self.add_weight(name='fp', initializer='zeros')
        self.true_negatives = self.add_weight(name='tn', initializer='zeros')
        self.false_negatives = self.add_weight(name='fn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.bool)
        y_pred = tf.cast(y_pred > 0.5, tf.bool)

        self.true_positives.assign_add(tf.reduce_sum(tf.cast(y_true & y_pred, tf.float32)))
        self.false_positives.assign_add(tf.reduce_sum(tf.cast(~y_true & y_pred, tf.float32)))
        self.true_negatives.assign_add(tf.reduce_sum(tf.cast(~y_true & ~y_pred, tf.float32)))
        self.false_negatives.assign_add(tf.reduce_sum(tf.cast(y_true & ~y_pred, tf.float32)))

    def result(self):
        precision = self.true_positives / (self.true_positives + self.false_positives + 1e-9)
        recall = self.true_positives / (self.true_positives + self.false_negatives + 1e-9)
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-9)
        return f1_score

    def reset_state(self):
        for v in self.variables:
            v.assign(0)
