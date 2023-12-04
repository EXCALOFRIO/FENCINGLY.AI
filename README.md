# FencingAI - Proyecto de Arbitraje de Esgrima con IA

[README](README_EN.md) English version.

Este proyecto se centra en el desarrollo de una inteligencia artificial capaz de arbitrar videos de esgrima. La IA se entrenará para reconocer y evaluar movimientos, técnicas y reglas de esgrima, permitiendo una evaluación objetiva de las acciones en los videos.

## Índice

1. [Agradecimientos](#agradecimientos)
2. [Objetivos del Proyecto](#objetivos-del-proyecto)
3. [Tecnologías Utilizadas](#tecnologías-utilizadas)
4. [Licencia](#licencia)

## Agradecimientos

Queremos expresar mi más sincero agradecimiento a [Scott (Shalom) Dubinsky](https://www.linkedin.com/in/sdubinsky/), quien generosamente proporcionó los videos de la base de datos de [fencingdatabase.com](https://www.fencingdatabase.com). Sin su apoyo y contribución, este proyecto no habría sido posible.

## Objetivos del Proyecto

El objetivo principal de este proyecto es desarrollar un sistema de IA que pueda:

- Evaluar de manera objetiva las acciones de los esgrimistas de acuerdo con las reglas establecidas.

## Tecnologías Utilizadas

El proyecto se basa en diversas tecnologías y herramientas, que incluyen, entre otras:

- Redes Neuronales Convolucionales (CNN) para la detección y clasificación de movimientos.
- Procesamiento de imágenes y video para el análisis de fotogramas.
- Aprendizaje automático para la evaluación de las acciones y el arbitraje.

## Problemas y Soluciones

#### Problema 1: Detecciones Erróneas de Personas o Artefactos

- Al usar OpenPose, se detectaron personas o artefactos irrelevantes. Se implementó un filtro basado en la dispersión de puntos para eliminar detecciones inapropiadas.

 <img src="https://github.com/EXCALOFRIO/FENCINGLY.AI/blob/main/outputs/V0_V1.gif?raw=true" width="600">

#### Problema 2: Detecciones con Número Insuficiente de Puntos

- OpenPose a veces generaba detecciones con pocos puntos. Se aplicó un filtro para descartar detecciones con menos de 15 puntos, mejorando la precisión.

 <img src="https://github.com/EXCALOFRIO/FENCINGLY.AI/blob/main/outputs/V1_V2.gif?raw=true" width="600">
 
##### Ambos filtros mejoraron significativamente la calidad de las detecciones en el proyecto.

## Licencia

Este proyecto se distribuye bajo la licencia [Apache License Version 2.0, January 2004](LICENSE). Consulta el archivo `LICENSE` para obtener más detalles.
