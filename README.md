# Aprendizaje supervisado usando Redes Neuronales

Este proyecto contiene el desarrollo de Redes Neuronales para la clasificación de caracteres codificados mediante matrices 7x5 píxeles, donde un valor de 1 representa el color negro y un valor 0 representa el color blanco.

El software se ha desarrollado en el ámbito académico y con la finalidad de ser usado también en dicho ámbito, para garantizar y facilitar el aprendizaje de los alumnos, sirviendo de simulador para ver el funcionamiento de las Redes Neuronales.

Para el desarrollo del trabajo, se han creado distintos conjuntos de datos para realizar el entrenamiento de la red neuronal artificial creada, un Perceptrón Multicapa (MLP) creado con la librería de Python denominada PyTorch y comparado con otros modelos creados en la plataforma H2O.

Este repositorio también contiene las pruebas realizadas en ambos entornos y los resultados conseguidos.


## [Datasets](./Datasets)
Este subdirectorio contiene los ficheros con los conjuntos de datos o datasets necesarios para el entrenamiento del modelo en ambos entornos.

Los datasets de dividen principalmente en dos conjuntos, en función del número de observaciones, que pueden ser 5.000 o 10.000. Cada uno de estos conjuntos dispone de diferentes tipos de datasets, en los que se ha variado el número de errores generados en ellos.

Por lo que los datasets quedarían de la siguiente manera en ambos entornos:
  - Dataset de 5.000 observaciones con hasta 2 errores.
  - Dataset de 5.000 observaciones con hasta 3 errores.
  - Dataset de 5.000 observaciones con hasta 5 errores.
  - Dataset de 10.000 observaciones con hasta 2 errores.
  - Dataset de 10.000 observaciones con hasta 3 errores.
  - Dataset de 10.000 observaciones con hasta 5 errores.

## [Implementaciones](./Implementaciones)
Este subdirectorio contiene la implementación del simulador creado en PyTorch, una guía de la implementación llevada a cabo en H2O, junto con un script para el cáculo de la precisión de las predicciones en la plataforma H2O, y un script para la generación de errores en los dataset.

## [Resultados](./Resultados)
Este subdirectorio contiene los resultados obtenidos de las pruebas realizadas con ambos entornos.
