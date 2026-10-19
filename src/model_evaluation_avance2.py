
import io
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
)


def evaluation(model_path='models/model.joblib', data_path='models/test_data.joblib'):
    """Devuelve una imagen PNG con métricas y matriz de confusión."""
    modelo = joblib.load(model_path)
    X, y = joblib.load(data_path)
    predicciones = modelo.predict(X)
    metricas = {
        'accuracy': accuracy_score(y, predicciones),
        'precision': precision_score(y, predicciones, zero_division=0),
        'recall': recall_score(y, predicciones, zero_division=0),
        'f1': f1_score(y, predicciones, zero_division=0),
    }
    if hasattr(modelo, 'predict_proba') and y.nunique() == 2:
        probabilidades = modelo.predict_proba(X)
        indice_positivo = list(modelo.classes_).index(1)
        metricas['roc_auc'] = roc_auc_score(y, probabilidades[:, indice_positivo])

    resumen = pd.DataFrame([metricas]).round(3)
    print('Resultados sobre datos de prueba:')
    print(resumen.to_string(index=False))
    fig, ejes = plt.subplots(1, 2, figsize=(13, 5))
    sns.barplot(x=list(metricas), y=list(metricas.values()), ax=ejes[0])
    ejes[0].set(title='Indicadores de clasificación', ylabel='Puntuación', ylim=(0, 1))
    ejes[0].tick_params(axis='x', rotation=25)
    sns.heatmap(confusion_matrix(y, predicciones), annot=True, fmt='d',
                cmap='Blues', ax=ejes[1])
    ejes[1].set(title='Matriz de confusión', xlabel='Predicción', ylabel='Valor real')
    fig.tight_layout()
    imagen = io.BytesIO()
    fig.savefig(imagen, format='png', dpi=150)
    plt.close(fig)
    imagen.seek(0)
    return imagen


if __name__ == '__main__':
    resultado = evaluation()
    with open('models/evaluation.png', 'wb') as archivo:
        archivo.write(resultado.getvalue())
