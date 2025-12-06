import pandas as pd
import matplotlib.pyplot as plt

# Ruta al archivo
ruta_archivo = r"D:\Datacuentas\Data\FRED\Non Farm Payrolls\NFP.csv"

# Leer el archivo
df = pd.read_csv(ruta_archivo)

# Convertir 'observation_date' a formato datetime
df['observation_date'] = pd.to_datetime(df['observation_date'])

# Ordenar por fecha
df = df.sort_values('observation_date')

# Graficar
plt.figure(figsize=(14, 6))
plt.plot(df['observation_date'], df['PAYEMS'], label='Non-Farm Payrolls (PAYEMS)')
plt.title('Evolución del Non-Farm Payrolls desde 1939')
plt.xlabel('Fecha')
plt.ylabel('Número de empleos (miles)')
plt.legend()
plt.grid(True)
plt.show()