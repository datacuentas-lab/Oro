import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from datetime import datetime, timedelta

# Paso 1: Cargar datos históricos diarios del oro (archivo CSV)
df = pd.read_csv("oro_historico_diario.csv", parse_dates=["Date"])
df = df.sort_values("Date")

# Paso 2: Cálculo de retornos logarítmicos para análisis estocástico
df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
df.dropna(inplace=True)

# Parámetros de movimiento browniano geométrico
drift = df['log_return'].mean()
volatility = df['log_return'].std()

# Horizonte de simulación: 2 meses (42 días hábiles aprox)
T = 42
dt = 1  # paso diario
S0 = df['Close'].iloc[-1]  # precio actual oro

# Simulación Monte Carlo de precio futuro con movimiento browniano geométrico
np.random.seed(42)
simulations = 1000
price_paths = np.zeros((T, simulations))
price_paths[0] = S0

for t in range(1, T):
    Z = np.random.standard_normal(simulations)
    price_paths[t] = price_paths[t-1] * np.exp((drift - 0.5 * volatility**2) * dt + volatility * np.sqrt(dt) * Z)

# Cálculo de retornos diarios simulados para cada camino
simulated_returns = np.log(price_paths[1:] / price_paths[:-1])

# Volatilidad diaria simulada: desviación estándar de los retornos simulados por día
volatilidad_diaria_simulada = np.std(simulated_returns, axis=1)

# Obtener índices de los 5 días con mayor volatilidad diaria, ordenados de mayor a menor
top5_indices = np.argsort(volatilidad_diaria_simulada)[-5:][::-1]
top5_volatilidades = volatilidad_diaria_simulada[top5_indices]

# Fecha base (día de hoy)
fecha_base = datetime(2025, 10, 28)

print("Los 5 días con mayor volatilidad simulada (ordenados de mayor a menor) son:")
for i, idx in enumerate(top5_indices):
    fecha = fecha_base + timedelta(days=int(idx) + 1)  # Convertir idx a int para timedelta
    fecha_str = fecha.strftime("%d-%m-%Y")
    print(f"{fecha_str}: Volatilidad simulada = {volatilidad_diaria_simulada[idx]:.6f}")

# Visualización con fechas en eje x
fechas_simulacion = [fecha_base + timedelta(days=i) for i in range(1, T)]

plt.figure(figsize=(10, 5))
plt.plot(fechas_simulacion, volatilidad_diaria_simulada, label="Volatilidad Diaria Simulada")
plt.scatter([fechas_simulacion[i] for i in top5_indices], top5_volatilidades, color='red', label="Top 5 Mayor Volatilidad")
plt.title("Volatilidad Diaria Estimada en Simulaciones de Precio del Oro")
plt.xlabel("Fecha")
plt.ylabel("Volatilidad (Desviación estándar del retorno logarítmico)")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()