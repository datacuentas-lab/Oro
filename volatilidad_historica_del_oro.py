import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ticker = "GC=F"
data = yf.download(ticker, start="2010-01-01", progress=False, auto_adjust=True)

data['log_return'] = np.log(data['Close'] / data['Close'].shift(1))

window_size = 21
data['volatilidad_rolling'] = data['log_return'].rolling(window=window_size, min_periods=1).std()

data = data.loc[~data['volatilidad_rolling'].isna()]

vol_min = 0.010
vol_max = 0.020
vol_rango = data[(data['volatilidad_rolling'] >= vol_min) & (data['volatilidad_rolling'] <= vol_max)]

print(f"Fechas con volatilidad entre {vol_min} y {vol_max}:")
for fecha, fila in vol_rango.iterrows():
    print(f"{fecha.date()}: Volatilidad = {fila['volatilidad_rolling'].iloc[0]:.6f}")

fig, ax = plt.subplots(figsize=(14,6))
ax.plot(data.index, data['volatilidad_rolling'], label='Volatilidad móvil diaria (21 días)')
ax.scatter(vol_rango.index, vol_rango['volatilidad_rolling'], color='red', label=f'Volatilidad entre {vol_min} y {vol_max}')

ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
fig.autofmt_xdate()

ax.set_title(f"Volatilidad diaria del precio del oro con fechas en rango [{vol_min}, {vol_max}]")
ax.set_xlabel("Fecha")
ax.set_ylabel("Volatilidad (desviación estándar retorno logarítmico)")
ax.legend()
ax.grid(True)
plt.show()