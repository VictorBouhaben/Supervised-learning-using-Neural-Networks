#
#   Víctor Bouhaben
#   25/05/2024
#   Contexto: TFG Ingeniería Informática EUPT
#
#   Programa para calcular la precición de un modelo de H2O,
#   compara la columna con el valor predicho con la columna con 
#   el valor real.
#

import pandas as pd

# Cargar el archivo CSV
df = pd.read_csv('C:\\Users\\Bowy\\Desktop\\resultados.csv')

columna1 = df['predict']
columna2 = df['output']

aciertos = 0

# Comparar las dos columnas
comparacion = columna1 == columna2
aciertos = comparacion.sum()

print("Precisión: " + str((aciertos / len(df) *100)) + " %")