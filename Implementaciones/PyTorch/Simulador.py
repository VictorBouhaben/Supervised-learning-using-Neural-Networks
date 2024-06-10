#
#   Víctor Bouhaben Barrera
#   25/05/2024
#
#   Simulador:
#       - Define y entrena red neuronal
#       - Realiza predicción y compara
#       - Muestra resultados
#
#   Contexto: 
#       - TFG Ingeniería Informática EUPT
#       - Realizado con la finalidad de uso en la enseñanza
#

import string
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import time

# Fichero dataset a importar
dataset = 'PyTorch-10.000-5cambios.csv'

hidden_layer = True     # Capa oculta
hidden_neurons = 128    # Número de neuronas de la capa oculta
lr=0.01                 # Learning rate
epochs = 10             # Número de epochs
batch_size = 300        # Tamaño de batch
cuda = False            # Uso de GPU
val_ratio = 0.2         # Porcentaje para el conjunto de test/validación

# Carga del DataSet
dataset = np.loadtxt(dataset, delimiter=';')
inputs = dataset[:,0:35]
labels = dataset[:,35:]

# Pasa a tensores los datos importados
inputs = torch.tensor(inputs, dtype=torch.float32)
labels = torch.tensor(labels, dtype=torch.float32)

# Función para dividir los datos en conjuntos de entrenamiento y validación manualmente
def train_val_split(inputs, labels, val_ratio):
    dataset_size = len(inputs)
    indices = list(range(dataset_size))
    split = int(np.floor(val_ratio * dataset_size))
    
    np.random.shuffle(indices)
    
    train_indices, val_indices = indices[split:], indices[:split]
    
    train_inputs, val_inputs = inputs[train_indices], inputs[val_indices]
    train_labels, val_labels = labels[train_indices], labels[val_indices]
    
    return train_inputs, val_inputs, train_labels, val_labels

# Dividir los datos
train_inputs, val_inputs, train_labels, val_labels = train_val_split(inputs, labels, val_ratio)

# DEFINICIÓN DEL MODELO
if hidden_layer:
    # Define el modelo con capa
    model = nn.Sequential(
        nn.Linear(35, hidden_neurons),
        nn.ReLU(),
        nn.Linear(hidden_neurons, 26),
        nn.Sigmoid()
    )
else:
    # Define el modelo sin capa
    model = nn.Sequential(
        nn.Linear(35, 26),
        nn.Sigmoid()
    )

# Función de pérdida
loss_fn   = nn.BCELoss()  # Entropía cruzada binaria

# Optimizador
# Usa logaritmo RMSProp
# Se le pasa como parámetro el learning rate
optimizer = optim.RMSprop(model.parameters(), lr=lr) 

losses = np.array([])

training_time = time.time()

# INICIO ENTRENAMIENTO
# Comprueba si se usa GPU
if cuda == True:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Se está usando: ", device)
    # Pasa el modelo y los datos a la GPU
    model.to(device)
    train_inputs = Variable(train_inputs.to(device))
    val_inputs = Variable(val_inputs.to(device))
    train_labels = Variable(train_labels.to(device))
    val_labels = Variable(val_labels.to(device))
else:
    print("Se está usando: cpu")

# Pasa los datos por el modelo un número de veces definido en la 
# variable "epochs"
for epoch in range(epochs):

    # Se le pasa al modelo un número de datos definido en la 
    # variable "batch_size"
    for i in range(0, len(train_inputs), batch_size):
        train_inputs_batch = train_inputs[i:i+batch_size]
        labels_pred = model(train_inputs_batch)
        train_labels_batch = train_labels[i:i+batch_size]

        # Calcula la pérdida (rendimiento) del modelo
        loss = loss_fn(labels_pred, train_labels_batch) 

        # Reinicia los gradientes
        optimizer.zero_grad()

        # Calcula los gradientes
        loss.backward()

        # El optimizador actualiza los parámetros del modelo
        optimizer.step()

    if epoch % 10 == 0 or epoch == epochs-1:
        print(f'Finished epoch {epoch}, latest loss {loss}')
    
    losses = np.append(losses, loss.item())

# Calcula tiempo de entrenamiento
training_time = time.time() - training_time

# FIN ENTRENAMIENTO

# Validación o test
with torch.no_grad():
    # Realiza una predicción
    val_labels_pred = model(val_inputs)

#Compara la predicción con los datos reales
accuracy = (val_labels_pred.round() == val_labels).float().mean()

print(f"Tiempo de entrenamiento {training_time}")
print(f"Accuracy {accuracy*100} %")

# Muestra tabla de loss
plt.plot(losses)
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.show(block=False)

# CONVIERTE A NÚMERO DE LA LETRA EN EL ALFABETO
# Crear el diccionario de mapeo
index_to_letter = {i: letter for i, letter in enumerate(string.ascii_lowercase)}

# Encontrar el índice del valor '1' de las etiquetas
def convert_to_number(tensor):
    letters_numbers = []
    for i in range(len(tensor)):
        index = torch.argmax(tensor[i]).item()
        index += 1
        letters_numbers.append(index)
    return letters_numbers

val_labels = convert_to_number(val_labels)
val_labels_pred = convert_to_number(val_labels_pred)

val_labels = np.array(val_labels)
val_labels_pred = np.array(val_labels_pred)

# Muestra tabla de regresión lineal y r2
fig, ax = plt.subplots()
ax.scatter(val_labels, val_labels_pred)
ax.plot([0, 27], [0, 27], 'k--', lw=4)
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')

# Línea de regresión
val_labels, val_labels_pred = val_labels.reshape(-1,1), val_labels_pred.reshape(-1,1)
ax.plot(val_labels, LinearRegression().fit  (val_labels, val_labels_pred).predict(val_labels))
ax.set_title('R2: ' + str(r2_score(val_labels, val_labels_pred)))
print('R2: ' + str(r2_score(val_labels, val_labels_pred)))
plt.show()