# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
#
# Renombre la columna "default payment next month" a "default"
# y remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las demas variables al intervalo [0, 1].
# - Selecciona las K mejores caracteristicas.
# - Ajusta un modelo de regresion logistica.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'type': 'metrics', 'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#

import os
import json
import gzip
import pickle

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import (
    precision_score,
    balanced_accuracy_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# Rutas 
TRAIN_PATH = "files/input/train_data.csv.zip"
TEST_PATH = "files/input/test_data.csv.zip"
MODEL_DIR = "files/models"
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl.gz")
OUTPUT_DIR = "files/output"
METRICS_PATH = os.path.join(OUTPUT_DIR, "metrics.json")


# Paso 1: Cargar y limpiar datos
def load_data():
    train = pd.read_csv(TRAIN_PATH, index_col=False, compression="zip")
    test = pd.read_csv(TEST_PATH, index_col=False, compression="zip")
    return train, test

def clean_data(df):
    df = df.rename(columns={"default payment next month": "default"})     # Renombre la columna "default payment next month" a "default".
    df = df.drop(columns=["ID"])                                          # Remueva la columna "ID".
    df = df[(df["EDUCATION"] != 0) & (df["MARRIAGE"] != 0)]               # Elimine los registros con informacion en 0                      
    df = df.dropna()                                                      # Elimine los registros con informacion no disponible.
    df.loc[df["EDUCATION"] > 4, "EDUCATION"] = 4                          # EDUCATION > 4 -> categoría 'others' (4)
    return df

# Paso 2: Dividir en x / y
def split_X_y(df):
    x = df.drop(columns=["default"])                                      # x: features
    y = df["default"]                                                     # y: target 
    return x, y

# Paso 3: Pipeline 
def build_pipeline():
    categorical_features = ["SEX", "EDUCATION", "MARRIAGE"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ],
        remainder=MinMaxScaler(),
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("select", SelectKBest(score_func=f_regression, k = 10)),
            ("model", LogisticRegression(max_iter=1000, random_state=42)),
        ],
    )
    return pipeline


# Paso 4: Optimización de los hiperparametros
def optimize_hyperparameters(pipeline, x_train, y_train):
    params = {
        "select__k": range(1, len(x_train.columns) + 1),
        "model__C": [0.1, 1, 10],
        "model__solver": ["liblinear", "lbfgs"],
    }

    tuned_model = GridSearchCV(
        estimator=pipeline,
        param_grid=params,
        cv=10,
        scoring="balanced_accuracy",
        n_jobs=-1,
        verbose=1,
    )

    tuned_model.fit(x_train, y_train)
    return tuned_model

# Paso 5: Guarde el modelo
def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, "wb") as f:
        pickle.dump(model, f)

# Paso 6 + 7: Métricas y Matriz de confusión
def compute_metrics(model, x, y, dataset_name):
    predictions = model.predict(x)

    precision = precision_score(y, predictions)
    balanced_acc = balanced_accuracy_score(y, predictions)
    recall = recall_score(y, predictions)
    f1 = f1_score(y, predictions)

    metrics = {
        "type": "metrics",
        "dataset": dataset_name,
        "precision": precision,
        "balanced_accuracy": balanced_acc,
        "recall": recall,
        "f1_score": f1,
    }

    cm = confusion_matrix(y, predictions)
    cm_info = {
        "type": "cm_matrix",
        "dataset": dataset_name,
        "true_0": {"predicted_0": int(cm[0, 0]), "predicted_1": int(cm[0, 1])},
        "true_1": {"predicted_0": int(cm[1, 0]), "predicted_1": int(cm[1, 1])},
    }

    return metrics, cm_info

def save_metrics(metrics_list, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in metrics_list:
            f.write(json.dumps(item) + "\n")


# Orquestador 
def main():
    # Cargar datos
    train_df, test_df = load_data()

    # Paso 1: limpieza
    train_clean = clean_data(train_df)
    test_clean = clean_data(test_df)

    # Paso 2: X / y
    X_train, y_train = split_X_y(train_clean)
    X_test, y_test = split_X_y(test_clean)

    # Paso 3: pipeline
    pipeline = build_pipeline()

    # Paso 4: GridSearchCV
    grid_search = optimize_hyperparameters(pipeline, X_train, y_train)

    # Paso 5: guardar el GridSearchCV completo 
    save_model(grid_search, MODEL_PATH)

    # Paso 6 y 7: métricas y matrices de confusión
    train_metrics, train_cm = compute_metrics(grid_search, X_train, y_train, dataset_name="train")
    test_metrics, test_cm = compute_metrics(grid_search, X_test, y_test, dataset_name="test")

    # Guardar métricas y matrices 
    all_metrics = [train_metrics, test_metrics, train_cm, test_cm]
    save_metrics(all_metrics, METRICS_PATH)


if __name__ == "__main__":
    main()