import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Código punto 2: descarga datos históricos diarios del oro y guarda en CSV
oro = yf.Ticker("GC=F")
data_historica = oro.history(period="max", interval="1d")
data_historica.to_csv("oro_historico_diario.csv")

# Cargar el archivo CSV descargado con pandas
df = pd.read_csv("oro_historico_diario.csv", parse_dates=["Date"])

# Graficar el precio de cierre del oro en el tiempo
plt.figure(figsize=(12, 6))
plt.plot(df["Date"], df["Close"], label="Precio de cierre")
plt.title("Precio Histórico Diario del Oro")
plt.xlabel("Fecha")
plt.ylabel("Precio (USD)")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()