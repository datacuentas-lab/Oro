[6:51, 3/12/2025] Ares: import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import yfinance as yf

# Parámetros y fechas
fecha_inicio = "2010-01-01"
fecha_hoy = datetime(2025, 10, 28)
fecha_fin_prediccion = datetime(2025, 12, 31)

# Descargar datos históricos ajustados de Yahoo Finance
ticker = "GC=F"  # Futuros de oro
data = yf.download(ticker, start=fecha_inicio, end=fecha_hoy.strftime("%Y-%m-%d"), auto_adjust=True)
data = data.reset_index()

# Calcular retornos logarítmicos diarios
data['log_return'] = np.log(data['Close'] / data['Close'].shift(1))
data.dropna(inplace=True)

# Estimación de parámetros para movimiento browniano geométrico
drift = data['log_return'].mean()
volatility = data['log_return'].std()

# Precio actual del oro (último disponible)
S0 = data['Close'].iloc[-1]

# Horizonte de simulación (días calendario)
diferencia_dias = (fecha_fin_prediccion - fecha_hoy).days
dias_simulacion = diferencia_dias

# Simulación Monte Carlo
simulations = 1000
price_paths = np.zeros((dias_simulacion, simulations))
price_paths[0] = S0

np.random.seed(42)
for t in range(1, dias_simulacion):
    Z = np.random.standard_normal(simulations)
    price_paths[t] = price_paths[t-1] * np.exp((drift - 0.5 * volatility**2) + volatility * Z)

# Cálculo del precio esperado para cada día
precio_esperado = price_paths.mean(axis=1)

# Crear rango de fechas para la tabla y gráfica
fechas_prediccion = [fecha_hoy + timedelta(days=i) for i in range(1, dias_simulacion + 1)]

# Crear DataFrame con la predicción diaria
tabla_prediccion = pd.DataFrame({
    "Fecha": fechas_prediccion,
    "Precio_Esperado_USD": precio_esperado
})

# Configurar pandas para imprimir toda la tabla sin truncar
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(tabla_prediccion)

# Gráfica solo desde hoy hasta fin de año
precio_min = np.percentile(price_paths, 5, axis=1)
precio_max = np.percentile(price_paths, 95, axis=1)

plt.figure(figsize=(12, 6))
plt.plot(fechas_prediccion, precio_esperado, label="Precio Esperado (Predicción Monte Carlo)")
plt.fill_between(fechas_prediccion, precio_min, precio_max, color='gray', alpha=0.3, label="Intervalo de Confianza 90%")
plt.title("Predicción del Precio del Oro desde Hoy hasta Fin de Año (Monte Carlo)")
plt.xlabel("Fecha")
plt.ylabel("Precio en USD")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
[6:57, 3/12/2025] Ares: Volatilidad histórica del oro
[6:58, 3/12/2025] Ares: import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt