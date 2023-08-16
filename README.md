<p align=center><img src=https://d31uz8lwfmyn8g.cloudfront.net/Assets/logo-henry-white-lg.png><p>

<h1 align=center>CRYPTO PROJECT</h1>

Estudio analítico de tendencias de 10 Crypto Monedas.

<h1 align=center>LIMPIEZA DE DATOS</h1> 

**CryptoCurrency List:**

1.) Bitcoin (BTC).

2.) VeChain (VET).

3.) Cardano (ADA).

4.) Polkadot (DOT).

5.) The Sandbox (SAND).

6.) ApeCoin (APE).

7.) Chiliz (CHZ).

8.) Dash (DASH).

9.) Ethereum (ETH).

10.) Tether (USDT).

En esta primera etapa del proyecto empezamos a realizar una limpieza de datos ya que la función que extraje directamente de la CoinGecko API fue la siguiente /coin/{id}/market_chart, gracias a esos datos y con ayuda de CoinMarketCap, pude sacar la deducción de las monedas que escogí y explicar mi preferencia al   momento de seleccionarlas.


Utilizando las librerías que ya conocemos pude extraer la información y a adecuar mi código de tal manera que la limpieza de cada dato fuera clara al momento de verlas reflejadas en las graficas:

``` python
import requests
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pycoingecko import CoinGeckoAPI
 ```
- Para resumir el proceso de la limpieza de datos empezare a mostrar los valores "Timestamps" los cuales son una combinación numérica en donde pueden ser remplazados por fechas y así poder llevarlo a los análisis exploratorios que nos ayudaran a hacer las predicciones de valoración de cada moneda:

1. Monedas de mayor capitalización:
   ```python
   top_10_coins = get_top_10_market_cap_coins(coin_list)
      for symbol, data in top_10_coins.items():
        print(f"{data['Name']} ({symbol}): Market Cap - {data['Market cap']}")
   ```
```python
Bitcoin (btc): Market Cap - 567954616232
Ethereum (eth): Market Cap - 219495626810
Tether (usdt): Market Cap - 83105960408
Cardano (ada): Market Cap - 9862024376
Polkadot (dot): Market Cap - 6079168173
VeChain (vet): Market Cap - 1279500512
The Sandbox (sand): Market Cap - 762546546
ApeCoin (ape): Market Cap - 673494492
Chiliz (chz): Market Cap - 495205335
Dash (dash): Market Cap - 336678828
```

2. Obtener Timestamp de cada criptomoneda:
``` python
top_10_coins = get_top_10_market_cap_coins(coin_list)
timestamps = {}

for symbol, coin_data in top_10_coins.items():
    coin_id = coin_data["ID"]
    timestamps[symbol] = get_timestamps(coin_id)
```
3. Obtenemos el Data Frame Limpio:
 <h1 align=center>Crear un DataFrame con los datos históricos de todas las criptomonedas</h1>

```python
top_10_coins = get_top_10_market_cap_coins(coin_list)
dataframes = [get_historical_data(coin_data['ID']).assign(Coin = symbol) for symbol, coin_data in top_10_coins.items()]

# Concatenar los DataFrames en uno solo
final_df = pd.concat(dataframes)

# Aplicar formato de moneda a las columnas "Price", "Market Cap" y "Volume"
#currency_columns = ['Price', 'Market Cap', 'Volume']
#for col in currency_columns:
   # final_df[col] = final_df[col].apply(lambda x: "${:,.2f}".format(round(x, 2)))

# Mostrar los datos de cada criptomoneda en un DataFrame individual
for symbol in top_10_coins.keys():
    coin_df = final_df[final_df['Coin'] == symbol]
    display(coin_df)

    # Guardar el DataFrame en un archivo CSV
    final_df.to_csv('datos_criptomonedas.csv', index = False)
```

- En esta finalización de los datos obtenidos los cuales se pueden ver mejor explicados en mi documento "Analisis_Exploratorio(EDA).ipynb", gracias a la extracción de datos procedemos a la creación del EDA.


<h1 align=center>ANALISIS EXPLORATORIO EDA</h1>

1. Realicé una Matriz de correlación entre variables por cada una de las monedas seleccionadas en donde me muestra el Volumen, Market Cap y el Precio, en esos valores y colores el mapa de calor te indicarán la fuerza y la dirección de la relación entre pares de variables. Los valores cercanos a 1 indican una correlación positiva fuerte (a medida que una variable aumenta, la otra también tiende a aumentar), mientras que valores cercanos a -1 indican una correlación negativa fuerte (a medida que una variable aumenta, la otra tiende a disminuir). Valores cercanos a 0 indican una correlación débil o nula.

```python
def plot_correlation_matrix(dataframe, coin):
    #Filtrar los datos solo para la moneda en especifico:
    coin_df = dataframe[dataframe["Coin"] == coin]

    #Seleccionar solo las columnas numéricas para la matriz de correlación
    numeric_columns = coin_df.select_dtypes(include=[float, int]).columns
    correlation_matrix = coin_df[numeric_columns].corr()

    #Establecer el estilo de las visualizaciones de Seaborn
    sns.set(style="white")

    #Crear una matriz de correlación
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", center=0, linewidths=.5)
    plt.title(f'Matriz de Correlación entre Variables Numéricas - {coin}')
    plt.show()

#Llamar a la función con tu DataFrame 'final_df' y la moneda deseada
plot_correlation_matrix(final_df, 'btc')
plot_correlation_matrix(final_df, 'eth')
plot_correlation_matrix(final_df, 'usdt')
plot_correlation_matrix(final_df, 'ada')
plot_correlation_matrix(final_df, 'dot')
plot_correlation_matrix(final_df, 'vet')
plot_correlation_matrix(final_df, 'sand')
plot_correlation_matrix(final_df, 'ape')
plot_correlation_matrix(final_df, 'chz')
plot_correlation_matrix(final_df, 'dash')
```
2 . Seguido realice un grafico de tendencias el cual nos es útil para observar datos que se muestran a lo largo del tiempo. Este tipo de gráficos es muy útil para observar patrones, cambios y tendencias en cada uno de los datos reflejados, de informaciones que se recopilan entre días, meses anos etc.

```python
from sklearn.preprocessing import MinMaxScaler

def plot_combined_normalized_trends(dataframe):
    #Copiar el DataFrame original para no modificar los datos originales
    df = dataframe.copy()

    #Crear una instancia del escalador MinMaxScaler
    scaler = MinMaxScaler()

    #Seleccionar solo las columnas de precio de cierre para normalizar
    prices = df.pivot(index='Timestamp', columns='Coin', values='Price')
    normalized_prices = scaler.fit_transform(prices)

    #Crear un DataFrame con los precios normalizados
    normalized_df = pd.DataFrame(normalized_prices, columns=prices.columns, index=prices.index)

    #Establecer el estilo de las visualizaciones de Seaborn
    sns.set(style="whitegrid")

    #Crear un gráfico de tendencia combinado con todas las criptomonedas con precios normalizados
    plt.figure(figsize=(10, 6)) 
    for symbol in normalized_df.columns:
        plt.plot(normalized_df.index, normalized_df[symbol], label=symbol)

    plt.title('Tendencia de Precios de Cierre Normalizados - Todas las Criptomonedas')
    plt.xlabel('Fecha')
    plt.ylabel('Precio Normalizado')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.show()

#Llamar a la función con tu DataFrame 'final_df'
plot_combined_normalized_trends(final_df)
```
3. Finalizamos nuestro EDA con una caja Boxplot o Outliers lo utilice mas que todo para identificación de los datos atípicos, la dispersión de datos y visualización de la mediana.

```python
def plot_individual_normalized_boxplots(dataframe):
    #Copiar el DataFrame original para no modificar los datos originales
    df = dataframe.copy()

    #Normalizar los datos dividiendo por el valor máximo de la columna "Price"
    df['Price'] = df.groupby('Coin')['Price'].transform(lambda x: x / x.max())

    #Establecer el estilo de las visualizaciones de Seaborn
    sns.set(style="whitegrid")

    #Crear un gráfico de caja para precios de cierre de cada criptomoneda normalizados
    plt.figure(figsize=(10, 8))
    sns.boxplot(data=df, x="Coin", y="Price")
    plt.title("Grafico de Caja - Precios de cierre de Criptomonedas Normalizados")
    plt.xlabel("Criptomonedas")
    plt.ylabel("Precio de cierre normalizado")
    plt.xticks(rotation=45)
    plt.ylim(0, 1)  # Ajustar el rango de valores en el eje y de 0 a 1
    plt.tight_layout()

    plt.show()

#Llamar a la función con tu DataFrame 'final_df'
plot_individual_normalized_boxplots(final_df)
```
<h1 align=center> KPI'S</h1>

En la continua presentación que hare, con información reflejada directamente de los Dashboard, procedo a explicar el porque de mi selección de los KPI en donde los relacione conjunto a las monedas escogidas, quiero resaltar antes de empezar la explicación que cada una de estas monedas escogidas fueron netamente por investigaciones previas realizadas y proyectos por cada una de ellas que las hacen atractivas para mi criterio, estos análisis reflejados son una recomendación, no estoy diciendo que a partir de este momento empiecen a invertir en estas monedas. Todos los ejemplos reflejados hacen parte de un estudio realizado por mi.

1. CRECIMIENTO MENSUAL PROMEDIO.
2. VOLUMEN DE NEGOCIACION.
3. VOLATILIDAD.
4. RETORNO DE INVERSION.






   
