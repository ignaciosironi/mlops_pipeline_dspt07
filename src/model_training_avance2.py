
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, learning_curve
from sklearn.pipeline import Pipeline
from src.cargar_datos_avance1 import cargarDatos
from src.ft_engineering_avance2 import (
    TARGET, COLUMNAS_NUMERIC, COLUMNAS_CATEGORIC, COLUMNAS_ORDINAL,
    EXCLUIR_DE_X, _get_feature_columns, build_preprocessor,
)


def summarize_classification(y_true, y_pred):
    """Calcula indicadores básicos para una clasificación binaria."""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }


def build_model(classifier_fn, data_params, test_frac=0.2,
                save_path_modelo='models/model.joblib',
                save_path_data='models/test_data.joblib'):
    """Ajusta un pipeline y guarda tanto el modelo como el conjunto de prueba."""
    datos = data_params['dataset']
    objetivo = data_params['name_of_y_col']
    columnas = [c for c in data_params['names_of_x_cols']
                if c in datos.columns and c not in EXCLUIR_DE_X and c != objetivo]
    if not columnas:
        raise ValueError('No se encontraron variables predictoras.')
    X = datos[columnas].copy()
    y = datos[objetivo].copy()
    for columna in COLUMNAS_CATEGORIC + COLUMNAS_ORDINAL:
        if columna in X:
            X[columna] = X[columna].astype('string').replace({'': pd.NA}).astype(object)
            X[columna] = X[columna].where(X[columna].notna(), np.nan)

    estratificar = y if y.value_counts().min() >= 2 else None
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=test_frac, random_state=1234, stratify=estratificar,
    )
    grupos = _get_feature_columns(x_train)
    modelo = Pipeline([
        ('preprocess', build_preprocessor(*grupos)),
        ('model', classifier_fn),
    ])
    modelo.fit(x_train, y_train)
    resumen = {
        'train': summarize_classification(y_train, modelo.predict(x_train)),
        'test': summarize_classification(y_test, modelo.predict(x_test)),
    }
    minimo = int(y_train.value_counts().min())
    if minimo >= 2:
        cv = StratifiedKFold(n_splits=min(5, minimo), shuffle=True, random_state=1234)
        resultados = cross_validate(
            modelo, x_train, y_train, cv=cv,
            scoring=['accuracy', 'precision', 'recall', 'f1'], n_jobs=-1,
        )
        resumen['cross_validation'] = {
            clave.removeprefix('test_'): float(np.mean(valor))
            for clave, valor in resultados.items() if clave.startswith('test_')
        }
        tamaños, puntajes_train, puntajes_cv = learning_curve(
            modelo, x_train, y_train, cv=cv, scoring='recall',
            train_sizes=np.linspace(0.3, 1.0, 5), n_jobs=-1,
        )
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(tamaños, puntajes_train.mean(axis=1), 'o-', label='Entrenamiento')
        ax.plot(tamaños, puntajes_cv.mean(axis=1), 'o-', label='Validación')
        ax.set(xlabel='Cantidad de ejemplos', ylabel='Recall', title='Curva de aprendizaje')
        ax.legend()
        fig.tight_layout()
        plt.show()
        plt.close(fig)

    for ruta in (save_path_modelo, save_path_data):
        Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(modelo, save_path_modelo)
    joblib.dump((x_test, y_test), save_path_data)
    return resumen


if __name__ == '__main__':
    dataframe = cargarDatos()
    columnas = COLUMNAS_NUMERIC + COLUMNAS_CATEGORIC + COLUMNAS_ORDINAL
    configuracion = {
        'dataset': dataframe,
        'name_of_y_col': TARGET,
        'names_of_x_cols': columnas,
    }
    resultados = build_model(
        RandomForestClassifier(n_estimators=100, random_state=1234),
        configuracion,
    )
    print(pd.DataFrame({k: v for k, v in resultados.items()}).round(3))
