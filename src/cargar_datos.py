import os
import pandas as pd
def cargarDatos():
    #Obtiene la ruta del directorio donde se encuentra este script (src/)
    script_dir = os.path.dirname(__file__)
    # Construye la ruta completa al archivo .xlsx
    file_path = os.path.join(script_dir, 'Base_de_datos.xlsx')
    # lee archivo excel usando la ruta completa
    df = pd.read_excel(file_path)
    return df