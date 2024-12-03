# Creación de un Modelo de Lenguaje para la Extracción de Terminología en Español  

## Descripción del Proyecto  
Este repositorio es el código del Trabajo de Fin de Grado (TFG) **Creación de un Modelo de Lenguaje para la Extracción de Terminología en Español** en la Universidad Politécnica de Madrid. El objetivo es adaptar y extender el código base del proyecto [PromptRank](https://github.com/NKU-HLT/PromptRank) para realizar predicciones en textos en español en lugar de inglés.  

El proyecto emplea el modelo **mT5** en lugar de **T5**, lo que permite manejar múltiples idiomas, incluyendo español. Como consecuencia, el tiempo de ejecución es mayor debido a la mayor complejidad de mT5 en comparación con T5.  

## Requisitos  
Todos los requisitos necesarios para ejecutar este proyecto están listados en el archivo `requirements.txt`. Asegúrate de instalarlos antes de comenzar:  
```bash
pip install -r requirements.txt
