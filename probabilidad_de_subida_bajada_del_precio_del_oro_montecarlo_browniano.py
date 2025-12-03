[6:43, 3/12/2025] Ares: import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

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

# Paso 5: Función Black-Scholes para valor de opciones europeas
def black_scholes(S, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("Tipo de opción inválida: use 'call' o 'put'.")
    return price

# Parámetros para opciones (ejemplo)
strike_price = S0  # opción at-the-money
risk_free_rate = 0.04  # tasa libre de riesgo anual (ejemplo)

call_price = black_scholes(S0, strike_price, T/252, risk_free_rate, volatility, "call")
put_price = black_scholes(S0, strike_price, T/252, risk_free_rate, volatility, "put")

# Visualizaciones

plt.figure(figsize=(14,8))

# Gráfico de simulaciones Monte Carlo
plt.subplot(2, 1, 1)
plt.plot(price_paths[:, :20], lw=1)
plt.title("Simulación Monte Carlo - Movimiento Browniano Geométrico del Precio del Oro para 2 meses")
plt.xlabel("Días")
plt.ylabel("Precio Oro (USD)")

# Curva gaussiana para probabilidad de retornos
plt.subplot(2, 1, 2)
x = np.linspace(-0.05, 0.05, 200)
plt.plot(x, norm.pdf(x, drift, volatility), 'r-', lw=2, label="Distribución Normal de retornos")
plt.axvline(0, color='black', linestyle='--')
plt.title("Probabilidad de Subida/Bajada en Precio del Oro")
plt.xlabel("Retorno Diario Logarítmico")
plt.ylabel("Densidad de probabilidad")
plt.legend()

plt.tight_layout()
plt.show()

print(f"Parámetros estimados:")
print(f"  Drift (media retornos diarios): {drift:.6f}")
print(f"  Volatilidad (std retornos diarios): {volatility:.6f}")
print(f"  Precio actual oro: ${S0:.2f}")
print(f"  Precio opción Call aprox. 2 meses (Strike = {strike_price:.2f}): ${call_price:.2f}")
print(f"  Precio opción Put aprox. 2 meses (Strike = {strike_price:.2f}): ${put_price:.2f}")