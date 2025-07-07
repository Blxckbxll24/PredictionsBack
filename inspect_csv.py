import pandas as pd

# Cargar el CSV
try:
    df = pd.read_csv('Datos_Completos.csv', encoding='utf-8', sep=',')
    df.columns = df.columns.str.strip()  # Eliminar espacios en los nombres de las columnas
    print("Columnas en el CSV:", df.columns.tolist())

    # Verificar valores no numéricos
    expected_numeric = ['age', 'avg_daily_usage_hours', 'sleep_hours_per_night', 'affects_academic_performance']
    for col in expected_numeric:
        if col in df.columns:
            non_numeric = df[pd.to_numeric(df[col], errors='coerce').isna() & df[col].notna()]
            if not non_numeric.empty:
                print(f"Valores no numéricos en '{col}':\n", non_numeric[[col]].head())
            else:
                print(f"No hay valores no numéricos en '{col}'")
        else:
            print(f"Columna '{col}' no encontrada en el CSV")
    
    # Mostrar primeras filas para inspección
    print("\nPrimeras 5 filas del CSV:\n", df.head())
except Exception as e:
    print(f"Error al cargar el CSV: {e}")