
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

TARGET = 'Pago_atiempo'
COLUMNAS_NUMERIC = [
    'capital_prestado', 'plazo_meses', 'edad_cliente', 'salario_cliente',
    'total_otros_prestamos', 'cuota_pactada', 'puntaje', 'puntaje_datacredito',
    'cant_creditosvigentes', 'huella_consulta', 'saldo_mora', 'saldo_total',
    'saldo_principal', 'saldo_mora_codeudor', 'creditos_sectorFinanciero',
    'creditos_sectorCooperativo', 'creditos_sectorReal',
    'promedio_ingresos_datacredito',
]
COLUMNAS_CATEGORIC = ['tipo_laboral', 'tendencia_ingresos']
COLUMNAS_ORDINAL = ['tipo_credito']
EXCLUIR_DE_X = ['fecha_prestamo']


def _get_feature_columns(data):
    """Devuelve columnas disponibles, agrupadas por su tratamiento."""
    disponibles = set(data.columns)
    return tuple(
        [nombre for nombre in grupo if nombre in disponibles]
        for grupo in (COLUMNAS_NUMERIC, COLUMNAS_CATEGORIC, COLUMNAS_ORDINAL)
    )


def build_preprocessor(numeric_cols, categorical_cols, ordinal_cols):
    """Construye transformaciones que se ajustan únicamente con entrenamiento."""
    pasos = []
    if numeric_cols:
        pasos.append(('numericas', SimpleImputer(strategy='median'), numeric_cols))
    if categorical_cols:
        nominal = make_pipeline(
            SimpleImputer(strategy='most_frequent'),
            OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False),
        )
        pasos.append(('nominales', nominal, categorical_cols))
    if ordinal_cols:
        ordinal = make_pipeline(
            SimpleImputer(strategy='most_frequent'),
            OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1),
        )
        pasos.append(('ordinales', ordinal, ordinal_cols))
    if not pasos:
        raise ValueError('No hay variables disponibles para el preprocesamiento.')
    return ColumnTransformer(pasos, remainder='drop', verbose_feature_names_out=False)
