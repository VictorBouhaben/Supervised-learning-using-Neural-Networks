#
#   Víctor Bouhaben Barrera
#   25/05/2024
#   Contexto: TFG Ingeniería Informática EUPT
#
#   Programa para realizar modificaciones en el dataset para crear errores
#

import csv
import random

# Archivo del que modificar los datos
archivo_sin_cambios = 'PyTorch-10.000-sinCambios.csv'

# Archivo en el que guardar los datos modificados
archivo_con_cambios = ''

# Función para cambiar un dígito aleatorio por su contrario
def cambiar_digito(numero):
    if numero == '0':
        return '1'
    elif numero == '1':
        return '0'
    else:
        return numero

# Leer el archivo CSV y almacenar los datos en una lista
with open(archivo_sin_cambios, 'r') as file:
    reader = csv.reader(file, delimiter=';')
    datos = list(reader)

# Iterar sobre cada fila y cambiar un dígito aleatorio
for fila in datos:
    c = 5
    probabilidad = random.randint(0,c)
    posicion_seleccionada = random.randint(0, 35-1)
    valor_original = fila[posicion_seleccionada]
    listaPosiciones = []

    if probabilidad == 0:
        pass
    elif probabilidad == 1:        
        fila[posicion_seleccionada] = cambiar_digito(valor_original)
    elif probabilidad == 2:
        fila[posicion_seleccionada] = cambiar_digito(valor_original)
        
        posicion_seleccionada1 = random.randint(0, 35-1)

        if posicion_seleccionada == posicion_seleccionada1:
            posicion_seleccionada1 = random.randint(0, 35-1)

        valor_original = fila[posicion_seleccionada1]
        fila[posicion_seleccionada1] = cambiar_digito(valor_original) 
    elif probabilidad == 3:
        fila[posicion_seleccionada] = cambiar_digito(valor_original)
        
        for i in range(1, c-2):
            posicion_seleccionada = random.randint(0, 35-1)

            if posicion_seleccionada in listaPosiciones:
                posicion_seleccionada = random.randint(0, 35-1)

            if posicion_seleccionada in listaPosiciones:
                posicion_seleccionada = random.randint(0, 35-1)

            if posicion_seleccionada in listaPosiciones:
                posicion_seleccionada = random.randint(0, 35-1)

            valor_original = fila[posicion_seleccionada]
            fila[posicion_seleccionada] = cambiar_digito(valor_original)
    elif probabilidad == 4:
        fila[posicion_seleccionada] = cambiar_digito(valor_original)
        
        for i in range(1, c-1):
            posicion_seleccionada = random.randint(0, 35-1)

            if posicion_seleccionada in listaPosiciones:
                posicion_seleccionada = random.randint(0, 35-1)

            if posicion_seleccionada in listaPosiciones:
                posicion_seleccionada = random.randint(0, 35-1)

            if posicion_seleccionada in listaPosiciones:
                posicion_seleccionada = random.randint(0, 35-1)

            valor_original = fila[posicion_seleccionada]
            fila[posicion_seleccionada] = cambiar_digito(valor_original)
    else:
        fila[posicion_seleccionada] = cambiar_digito(valor_original)
        
        for i in range(1, c):
            posicion_seleccionada = random.randint(0, 35-1)

            if posicion_seleccionada in listaPosiciones:
                posicion_seleccionada = random.randint(0, 35-1)

            if posicion_seleccionada in listaPosiciones:
                posicion_seleccionada = random.randint(0, 35-1)

            if posicion_seleccionada in listaPosiciones:
                posicion_seleccionada = random.randint(0, 35-1)

            valor_original = fila[posicion_seleccionada]
            fila[posicion_seleccionada] = cambiar_digito(valor_original)

# Guardar los datos modificados de vuelta al archivo CSV
with open(archivo_con_cambios, 'w', newline='') as file:
    writer = csv.writer(file, delimiter=';')
    writer.writerows(datos)