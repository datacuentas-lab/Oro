import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import pearsonr
import seaborn as sns

def analizar_correlacion_oro_nfp():
    # =========================================================================
    # 1. ANÁLISIS DE VOLATILIDAD DEL ORO (Tu primer código adaptado)
    # =========================================================================
    print("Descargando y procesando datos del oro...")
    
    ticker = "GC=F"
    data_oro = yf.download(ticker, start="2000-01-01", progress=False, auto_adjust=True)
    
    # Calcular volatilidad
    data_oro['log_return'] = np.log(data_oro['Close'] / data_oro['Close'].shift(1))
    window_size = 21
    data_oro['volatilidad_rolling'] = data_oro['log_return'].rolling(window=window_size, min_periods=1).std()
    data_oro = data_oro.dropna()
    
    # Filtrar volatilidad en el rango deseado
    vol_min = 0.010
    vol_max = 0.020
    vol_rango = data_oro[(data_oro['volatilidad_rolling'] >= vol_min) & 
                         (data_oro['volatilidad_rolling'] <= vol_max)].copy()
    
    print(f"Encontradas {len(vol_rango)} fechas con volatilidad entre {vol_min} y {vol_max}")

    # =========================================================================
    # 2. CARGAR DATOS NFP (Tu segundo código adaptado)
    # =========================================================================
    print("\nCargando datos del Non-Farm Payrolls...")
    
    # Simulamos datos NFP (en tu caso usarías tu archivo CSV)
    # Para este ejemplo, crearemos datos sintéticos del NFP
    fecha_inicio = "2000-01-01"
    fecha_fin = "2024-01-01"
    
    # Generar fechas mensuales (NFP se publica mensualmente)
    fechas_mensuales = pd.date_range(start=fecha_inicio, end=fecha_fin, freq='MS')
    
    # Crear datos sintéticos del NFP (tendencia creciente + estacionalidad + ruido)
    np.random.seed(42)
    tendencia = np.linspace(130000, 155000, len(fechas_mensuales))
    estacionalidad = 5000 * np.sin(2 * np.pi * np.arange(len(fechas_mensuales)) / 12)
    ruido = np.random.normal(0, 3000, len(fechas_mensuales))
    
    nfp_sintetico = tendencia + estacionalidad + ruido
    
    data_nfp = pd.DataFrame({
        'observation_date': fechas_mensuales,
        'PAYEMS': nfp_sintetico
    })
    
    print(f"Cargados {len(data_nfp)} registros del NFP")

    # =========================================================================
    # 3. PREPARAR DATOS PARA CORRELACIÓN
    # =========================================================================
    print("\nPreparando datos para análisis de correlación...")
    
    # Crear serie mensual de volatilidad (promedio mensual)
    data_oro['mes'] = data_oro.index.to_period('M')
    volatilidad_mensual = data_oro.groupby('mes')['volatilidad_rolling'].mean().reset_index()
    volatilidad_mensual['fecha'] = volatilidad_mensual['mes'].dt.to_timestamp()
    
    # Combinar datos
    data_combinada = pd.merge(volatilidad_mensual, data_nfp, 
                             left_on='fecha', right_on='observation_date', 
                             how='inner')
    
    # Crear variable binaria para periodos de baja volatilidad
    data_combinada['baja_volatilidad'] = ((data_combinada['volatilidad_rolling'] >= vol_min) & 
                                         (data_combinada['volatilidad_rolling'] <= vol_max)).astype(int)
    
    # Calcular variación mensual del NFP
    data_combinada['nfp_cambio_mensual'] = data_combinada['PAYEMS'].pct_change() * 100
    
    print(f"Datos combinados: {len(data_combinada)} observaciones mensuales")

    # =========================================================================
    # 4. ANÁLISIS DE CORRELACIÓN
    # =========================================================================
    print("\n" + "="*50)
    print("ANÁLISIS DE CORRELACIÓN")
    print("="*50)
    
    # Correlación directa entre volatilidad y NFP
    correlacion, p_valor = pearsonr(data_combinada['volatilidad_rolling'], 
                                   data_combinada['PAYEMS'])
    
    print(f"Correlación volatilidad vs nivel NFP: {correlacion:.4f}")
    print(f"Valor p: {p_valor:.4f}")
    
    # Correlación entre volatilidad y cambio mensual del NFP
    data_temp = data_combinada.dropna()
    if len(data_temp) > 0:
        correlacion_cambio, p_valor_cambio = pearsonr(data_temp['volatilidad_rolling'], 
                                                     data_temp['nfp_cambio_mensual'])
        print(f"Correlación volatilidad vs cambio NFP: {correlacion_cambio:.4f}")
        print(f"Valor p: {p_valor_cambio:.4f}")
    
    # Análisis de periodos de baja volatilidad vs NFP
    print(f"\nPeriodos de baja volatilidad: {data_combinada['baja_volatilidad'].sum()} meses")
    
    if data_combinada['baja_volatilidad'].sum() > 0:
        nfp_baja_vol = data_combinada[data_combinada['baja_volatilidad'] == 1]['PAYEMS']
        nfp_alta_vol = data_combinada[data_combinada['baja_volatilidad'] == 0]['PAYEMS']
        
        print(f"NFP promedio en baja volatilidad: {nfp_baja_vol.mean():.0f}")
        print(f"NFP promedio en alta volatilidad: {nfp_alta_vol.mean():.0f}")
        print(f"Diferencia: {nfp_baja_vol.mean() - nfp_alta_vol.mean():.0f}")

    # =========================================================================
    # 5. VISUALIZACIONES
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Gráfico 1: Serie temporal comparativa
    ax1 = axes[0, 0]
    ax1.plot(data_combinada['fecha'], data_combinada['volatilidad_rolling'], 
             color='orange', label='Volatilidad Oro', linewidth=2)
    ax1.set_ylabel('Volatilidad Oro', color='orange')
    ax1.tick_params(axis='y', labelcolor='orange')
    ax1.legend(loc='upper left')
    
    ax1b = ax1.twinx()
    ax1b.plot(data_combinada['fecha'], data_combinada['PAYEMS'], 
              color='blue', alpha=0.7, label='NFP')
    ax1b.set_ylabel('Non-Farm Payrolls', color='blue')
    ax1b.tick_params(axis='y', labelcolor='blue')
    ax1b.legend(loc='upper right')
    ax1.set_title('Volatilidad del Oro vs Non-Farm Payrolls')
    
    # Gráfico 2: Dispersión correlación
    ax2 = axes[0, 1]
    scatter = ax2.scatter(data_combinada['volatilidad_rolling'], data_combinada['PAYEMS'],
                         c=data_combinada['fecha'].dt.year, cmap='viridis', alpha=0.6)
    ax2.set_xlabel('Volatilidad Oro')
    ax2.set_ylabel('Non-Farm Payrolls')
    ax2.set_title(f'Correlación: {correlacion:.4f}')
    plt.colorbar(scatter, ax=ax2, label='Año')
    
    # Gráfico 3: Distribución por condición de volatilidad
    ax3 = axes[1, 0]
    condiciones = ['Alta Volatilidad', 'Baja Volatilidad']
    valores_nfp = [nfp_alta_vol.mean(), nfp_baja_vol.mean()]
    bars = ax3.bar(condiciones, valores_nfp, color=['lightcoral', 'lightgreen'])
    ax3.set_ylabel('NFP Promedio')
    ax3.set_title('NFP Promedio por Nivel de Volatilidad')
    
    # Añadir valores en las barras
    for bar, valor in zip(bars, valores_nfp):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100, 
                f'{valor:.0f}', ha='center', va='bottom')
    
    # Gráfico 4: Heatmap de correlaciones
    ax4 = axes[1, 1]
    variables_corr = ['volatilidad_rolling', 'PAYEMS', 'nfp_cambio_mensual', 'baja_volatilidad']
    corr_matrix = data_combinada[variables_corr].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax4)
    ax4.set_title('Matriz de Correlaciones')

    plt.tight_layout()
    plt.show()

    # =========================================================================
    # 6. ANÁLISIS ADICIONAL CON ÍNDICE
    # =========================================================================
    print("\n" + "="*50)
    print("ANÁLISIS CON ÍNDICE NORMALIZADO")
    print("="*50)
    
    # Crear índice normalizado para mejor comparación
    data_combinada['volatilidad_norm'] = (data_combinada['volatilidad_rolling'] - 
                                         data_combinada['volatilidad_rolling'].min()) / \
                                        (data_combinada['volatilidad_rolling'].max() - 
                                         data_combinada['volatilidad_rolling'].min())
    
    data_combinada['nfp_norm'] = (data_combinada['PAYEMS'] - 
                                 data_combinada['PAYEMS'].min()) / \
                                (data_combinada['PAYEMS'].max() - 
                                 data_combinada['PAYEMS'].min())
    
    # Correlación con datos normalizados
    correlacion_norm, p_valor_norm = pearsonr(data_combinada['volatilidad_norm'], 
                                             data_combinada['nfp_norm'])
    
    print(f"Correlación con datos normalizados: {correlacion_norm:.4f}")
    print(f"Valor p: {p_valor_norm:.4f}")
    
    # Resumen ejecutivo
    print("\n" + "="*50)
    print("RESUMEN EJECUTIVO")
    print("="*50)
    
    if abs(correlacion) > 0.3:
        direccion = "positiva" if correlacion > 0 else "negativa"
        print(f"✓ Se encontró una correlación {direccion} moderada")
    else:
        print("✓ No se encontró una correlación significativa")
    
    if data_combinada['baja_volatilidad'].sum() > 0:
        diferencia = nfp_baja_vol.mean() - nfp_alta_vol.mean()
        if abs(diferencia) > 1000:
            tendencia = "mayor" if diferencia > 0 else "menor"
            print(f"✓ El NFP tiende a ser {tendencia} en periodos de baja volatilidad")

    return data_combinada

# Ejecutar el análisis
if _name_ == "_main_":
    resultados = analizar_correlacion_oro_nfp()