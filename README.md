# Proyecto integrador: Predicción de riesgo de pago

Este proyecto consiste en el desarrollo de un modelo de *machine learning* para predecir si un cliente realizará el pago de un crédito a tiempo. El trabajo se realizó en cuatro avances, desde la preparación y el análisis de los datos hasta el entrenamiento, la evaluación, el monitoreo y el despliegue del modelo mediante una API y Docker. Para la visualizacio´n de la api se utiliza FastAPI y para la ejecución de los contenedores, docker desktop.

## Requisitos e instalación

Para ejecutar el proyecto se necesita Python, las librerías indicadas en `requirements.txt` y Docker Desktop para el despliegue en contenedores.

Desde la carpeta principal del proyecto, se crea y activa un entorno virtual a través de la terminal bash:

```bash
python -m venv venv
source venv/Scripts/activate
```

Luego se instalan las dependencias:

```bash
pip install -r requirements.txt
```

## Avance 1: Carga, EDA y ETL:

- **`cargar_datos.py`:** permite convertir la base original `Base_de_datos.xlsx` a `Base_de_datos.csv` y cargar los datos para trabajar con ellos.
- **`comprension_eda.ipynb`:** se utiliza para explorar y comprender la información. Se revisan los tipos de datos, valores faltantes, distribuciones y relaciones entre variables. También se realizan tareas de limpieza y transformación (ETL) para preparar los datos.

La variable que se busca predecir es `Pago_atiempo`, que indica si el cliente paga su crédito dentro del plazo previsto.

## Avance 2: Ingeniería de variables y entrenamiento

- **`ft_engineering.py`:** prepara las variables que se utilizan para entrenar el modelo, incluyendo las transformaciones necesarias y la separación de los datos de entrenamiento y prueba.
- **`model_training.py`:** entrena y compara modelos de clasificación para seleccionar el que presenta mejores resultados.

Para ejecutar el entrenamiento desde la raíz del proyecto:

```bash
python -m src.model_training_avance2
```

Al finalizar, se genera la carpeta `models/` con los archivos `model.joblib` (modelo entrenado) y `test_data.joblib` (datos de prueba guardados para su uso posterior).

## Avance 3: Evaluación y monitoreo

- **`model_evaluation.py`:** permite evaluar el rendimiento del modelo mediante métricas de clasificación.
- **`model_monitoring.py`:** permite revisar el comportamiento de las predicciones y detectar posibles cambios en los datos (data drift).
- **`streamlit_app.py`:** presenta una interfaz para visualizar predicciones, resultados y datos del monitoreo.

Para iniciar la aplicación de Streamlit:

```bash
python -m streamlit run streamlit_app.py
```

## Avance 4: API y despliegue con Docker

- **`model_deploy.py`:** carga el modelo entrenado y lo pone a disposición mediante una API desarrollada con FastAPI. El endpoint "/predict" permite realizar predicciones a partir de los datos enviados.
- **"Dockerfile":** contiene las instrucciones para construir una imagen con el código y las dependencias del proyecto.
- **".dockerignore":** indica qué archivos o carpetas no deben incluirse en la construcción de la imagen, como el entorno virtual y los archivos temporales.

### Ejecución local de FastAPI

Desde la carpeta principal, se inicia el servidor:

```bash
uvicorn src.model_deploy_avance4:app --reload
```

La documentación interactiva de la API se puede consultar en:

`http://127.0.0.1:8000/docs`

### Construcción de la imagen Docker

Con Docker Desktop abierto, se ejecuta:

```bash
docker build -t api-imagendocker-modulo5-avance4 .
```

### Creación y ejecución del contenedor

Primero se detiene el servidor local de FastAPI, si todavía está usando el puerto 8000. Luego se ejecuta:

```bash
docker run --name api-modelo-riesgo-pago -p 8000:8000 api-imagendocker-modulo5-avance4
```

Este comando crea e inicia el contenedor y permite acceder a la API desde `http://localhost:8000/docs`. Tanto la imagen como el contenedor pueden visualizarse en Docker Desktop, donde también se pueden consultar los registros de ejecución.

## Resumen

A lo largo de los cuatro avances se trabajó con la carga y preparación de los datos, el análisis exploratorio, el entrenamiento y la evaluación de modelos, el monitoreo de sus resultados y su despliegue. Finalmente, FastAPI permite utilizar el modelo para generar predicciones y sirve de interfaz para que el un usuario interactúe con el modelo entrenado. Docker facilita su ejecución en un entorno aislado y Streamlit ofrece una interfaz para visualizar e interactuar con los datos y resultados.

> **Nota:** En las explicaciones se utilizaron nombres de archivos simplificados. Los comandos incluyen los nombres reales de los módulos utilizados en este proyecto, como `model_training_avance2.py` y `model_deploy_avance4.py`.