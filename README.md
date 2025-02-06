# Creación de un Modelo de Lenguaje para la Extracción de Terminología en Español  

## Descripción del Proyecto  
Este repositorio es el código del Trabajo de Fin de Grado (TFG) **Creación de un Modelo de Lenguaje para la Extracción de Terminología en Español** en la Universidad Politécnica de Madrid. El objetivo es adaptar y extender el código base del proyecto [PromptRank](https://github.com/NKU-HLT/PromptRank) para realizar predicciones en textos en español en lugar de inglés.  

El proyecto emplea el modelo **mT5** en lugar de **T5**, lo que permite manejar múltiples idiomas, incluyendo español. Como consecuencia, el tiempo de ejecución es mayor debido a la mayor complejidad de mT5 en comparación con T5.

Actualmente se está evaluando el modelo con una versión traducida al español de la base de datos SemEval2017. Esta base de datos se encuentra bajo el fichero /data. En el que encontramos los documentos en el /docsutf8 y las frases clave anotadas de estos documentos en /keys.

## Requisitos  
Todos los requisitos necesarios para ejecutar este proyecto están listados en el archivo `requirements.txt`. Asegúrate de instalarlos antes de comenzar:  
```bash
pip install -r requirements.txt

##Ejecución

Para ejecutar el modelo y extraer las frases clave con los valores por defecto, simplemente ejecuta el siguiente comando en la terminal de VS Code o cualquier otra terminal compatible:
```bash
py.exe .\main.py

Los resultados de la ejecución se guardarán en el fichero PromptRankLib.log.
