"""Interactive prediction client and monitoring dashboard."""
import os
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from sklearn.model_selection import train_test_split
from src.model_deploy_avance4 import CreditRecord ##modificado
from src.model_monitoring_avance3 import (normalize_features, record_predictions, read_log,
                              measure_drift, save_drift_snapshot, read_history)

st.set_page_config(page_title='Advance 3 | Model monitoring', layout='wide')
st.title('📊 Payment prediction and model monitoring')
st.caption('API predictions • distribution drift • historical alerts')

@st.cache_data(show_spinner=False)
def load_dataset(csv_path: str):
    if csv_path and Path(csv_path).is_file():
        return pd.read_csv(csv_path)
    try:
        from src.cargar_datos_avance1 import cargarDatos
        return cargarDatos()
    except (ImportError, ModuleNotFoundError):
        raise FileNotFoundError('Provide a dataset CSV path, or make src.cargar_datos_avance1.cargarDatos available.')

with st.sidebar:
    st.header('Settings')
    api_url = st.text_input('Prediction API endpoint', os.getenv('PREDICTION_API_URL', 'http://localhost:8000/predict'))
    dataset_path = st.text_input('Dataset CSV path (optional)', os.getenv('DATASET_CSV', 'Base_de_datos.csv'))
    target = st.text_input('Target column', 'Pago_atiempo')
    sample_size = st.slider('Batch sample size', 5, 500, 100)
    psi_warning = st.number_input('PSI warning threshold', value=0.10, min_value=0.01, step=0.05)
    psi_critical = st.number_input('PSI critical threshold', value=0.25, min_value=float(psi_warning) + 0.01, step=0.05)

try:
    data = load_dataset(dataset_path)
    if target not in data:
        st.error(f'Target column {target!r} not found. Available columns: {list(data.columns)}')
        st.stop()
    features = data.drop(columns=[target])
    required_fields = list(CreditRecord.model_fields)
    missing_fields = [name for name in required_fields if name not in features.columns]
    if missing_fields:
        raise ValueError(f'Dataset does not contain required API features: {missing_fields}. Check feature engineering output.')
    features = features[required_fields].copy()
    labels = data[target]
    stratify = labels if labels.value_counts().min() >= 2 else None
    reference, candidate = train_test_split(features, test_size=0.2, random_state=42, stratify=stratify)
except Exception as error:
    st.error(f'Cannot load data: {error}')
    st.info('Place this app in your project root beside the src folder, or provide a CSV file path in the sidebar.')
    st.stop()


def request_predictions(batch: pd.DataFrame):
    payload = {'data': normalize_features(batch).to_dict(orient='records')}
    response = requests.post(api_url, json=payload, timeout=30)
    response.raise_for_status()
    result = response.json()
    if not isinstance(result, dict) or not isinstance(result.get('predictions'), list):
        raise ValueError('API must return JSON with a predictions list.')
    if len(result['predictions']) != len(batch):
        raise ValueError('API returned a different number of predictions than requested.')
    return result['predictions']

prediction_tab, drift_tab, history_tab = st.tabs(['🔮 Predictions', '📈 Data drift', '🕒 History & logs'])

with prediction_tab:
    st.subheader('Send data to the prediction API')
    mode = st.radio('Input method', ['Random sample', 'Edit one record'], horizontal=True)
    if mode == 'Random sample':
        size = min(sample_size, len(candidate))
        if size < sample_size:
            st.info(f'The evaluation subset contains only {size} records; the sample was reduced.')
        if 'sample_seed' not in st.session_state:
            st.session_state.sample_seed = 42
        if st.button('Shuffle sample'):
            st.session_state.sample_seed += 1
        batch = candidate.sample(n=size, random_state=st.session_state.sample_seed)
        st.dataframe(batch.head(15), use_container_width=True)
    else:
        st.caption('Edit the values below. The API will receive exactly one record.')
        edited = st.data_editor(features.head(1).copy(), num_rows='fixed', use_container_width=True)
        batch = edited
    if st.button('🚀 Predict and save to log', type='primary'):
        try:
            predictions = request_predictions(batch)
            record_predictions(batch, predictions)
            st.success(f'{len(predictions)} predictions saved to monitoring_log.csv')
            st.dataframe(pd.DataFrame({'prediction': predictions}), use_container_width=True)
        except Exception as error:
            st.error(f'Prediction failed: {error}')
            st.info('Check that the API is running and accepts POST /predict with {"data": [records]}.')

logs = read_log()
with drift_tab:
    st.subheader('Reference versus observed inputs')
    st.caption('Reference: training split (80%). Current: API prediction logs. No log means no observed production sample.')
    if logs.empty:
        st.info('First request predictions from the Predictions tab to collect current observations.')
    else:
        observed = logs.drop(columns=['prediction', 'timestamp', 'batch_id'], errors='ignore')
        drift = measure_drift(reference, observed, psi_warning, psi_critical)
        if drift.empty:
            st.warning('No common input columns were found for comparison.')
        else:
            a, b, c = st.columns(3)
            a.metric('Observed rows', len(observed))
            b.metric('Warning variables', int((drift.status == 'Warning').sum()))
            c.metric('Critical variables', int((drift.status == 'Critical').sum()))
            if (drift.status == 'Critical').any():
                st.error('Critical drift: review affected variables, data quality and model performance; consider retraining after validation.')
            elif (drift.status == 'Warning').any():
                st.warning('Potential drift: inspect affected variables and continue monitoring.')
            else:
                st.success('No PSI thresholds exceeded in the observed sample.')
            st.dataframe(drift.style.format({'psi': '{:.3f}', 'ks_pvalue': '{:.4f}', 'js_divergence': '{:.4f}', 'chi2_pvalue': '{:.4f}'}, na_rep='—'), use_container_width=True)
            st.plotly_chart(px.bar(drift, x='feature', y='psi', color='status', title='PSI by feature'), use_container_width=True)
            feature = st.selectbox('Compare distribution for', drift.feature.tolist())
            if drift.set_index('feature').loc[feature, 'type'] == 'numeric':
                a = pd.to_numeric(reference[feature], errors='coerce')
                b = pd.to_numeric(observed[feature], errors='coerce')
                plot_data = pd.concat([pd.DataFrame({'value': a, 'source': 'Reference'}), pd.DataFrame({'value': b, 'source': 'Observed'})])
                st.plotly_chart(px.histogram(plot_data, x='value', color='source', histnorm='probability density', barmode='overlay', title=feature), use_container_width=True)
            else:
                counts = pd.concat([reference[feature].fillna('(missing)').astype(str).value_counts(normalize=True).rename('Reference'), observed[feature].fillna('(missing)').astype(str).value_counts(normalize=True).rename('Observed')], axis=1).fillna(0)
                st.plotly_chart(px.bar(counts.rename_axis('category').reset_index().melt(id_vars='category', var_name='source', value_name='fraction'), x='category', y='fraction', color='source', barmode='group', title=feature), use_container_width=True)
            if st.button('💾 Save drift snapshot'):
                save_drift_snapshot(drift)
                st.success('Snapshot saved to drift_history.csv')
            st.caption('PSI 0.10/0.25 are conventional heuristic thresholds; p-values require adequate sample sizes and multiple-testing care. Drift does not prove accuracy degradation.')

with history_tab:
    if logs.empty:
        st.info('No prediction history yet.')
    else:
        st.metric('Predictions logged', len(logs))
        numeric_predictions = pd.to_numeric(logs['prediction'], errors='coerce')
        if numeric_predictions.notna().any():
            st.metric('Mean numeric prediction', f'{numeric_predictions.mean():.3f}')
        if 'timestamp' in logs:
            timeline = logs.copy()
            timeline['timestamp'] = pd.to_datetime(timeline['timestamp'], errors='coerce', utc=True)
            timeline['prediction_numeric'] = numeric_predictions
            series = timeline.dropna(subset=['timestamp', 'prediction_numeric']).set_index('timestamp').resample('1h')['prediction_numeric'].mean().reset_index()
            if not series.empty:
                st.plotly_chart(px.line(series, x='timestamp', y='prediction_numeric', title='Mean predictions over time'), use_container_width=True)
        st.dataframe(logs.tail(100), use_container_width=True)
        st.download_button('Download prediction log', logs.to_csv(index=False), 'monitoring_log.csv', 'text/csv')
    history = read_history()
    st.subheader('Saved drift snapshots')
    if history.empty:
        st.info('Save at least two drift snapshots to inspect changes over time.')
    else:
        history['checked_at'] = pd.to_datetime(history['checked_at'], errors='coerce', utc=True)
        st.plotly_chart(px.line(history, x='checked_at', y='psi', color='feature', markers=True, title='PSI trend by feature'), use_container_width=True)
        st.dataframe(history.tail(100), use_container_width=True)
        st.download_button('Download drift history', history.to_csv(index=False), 'drift_history.csv', 'text/csv')
