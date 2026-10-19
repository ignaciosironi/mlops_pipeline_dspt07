from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXCEL_PATH = PROJECT_ROOT / "Base_de_datos.xlsx"
CSV_PATH = PROJECT_ROOT / "Base_de_datos.csv"


def convertir_excel_csv():
    """Convert the original Excel dataset to CSV without modifying the Excel file."""
    df = pd.read_excel(EXCEL_PATH)
    df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
    return CSV_PATH


def cargarDatos():
    """Load the original dataset from CSV, creating it if necessary."""
    if not CSV_PATH.exists():
        convertir_excel_csv()
    return pd.read_csv(CSV_PATH)
