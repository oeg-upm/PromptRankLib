# Creación de un Modelo de Lenguaje para la Extracción de Terminología en Español  

## Descripción del Proyecto  
Este repositorio es el código del Trabajo de Fin de Grado (TFG) **Creación de un Modelo de Lenguaje para la Extracción de Terminología en Español** en la Universidad Politécnica de Madrid. El objetivo es adaptar y extender el código base del proyecto [PromptRank](https://github.com/NKU-HLT/PromptRank) para realizar predicciones en textos en español en lugar de inglés.  

El proyecto emplea el modelo **mT5** en lugar de **T5**, lo que permite manejar múltiples idiomas, incluyendo español. Como consecuencia, el tiempo de ejecución es mayor debido a la mayor complejidad de mT5 en comparación con T5.

Actualmente se está evaluando el modelo con una versión traducida al español de la base de datos SemEval2017. Esta base de datos se encuentra bajo el fichero /data. En el que encontramos los documentos en el /docsutf8 y las frases clave anotadas de estos documentos en /keys.

## Requisitos  
Todos los requisitos necesarios para ejecutar este proyecto están listados en el archivo `requirements.txt`. Asegúrate de instalarlos antes de comenzar:  
```bash
pip install -r requirements.txt
```

## Ejecución
Para ejecutar el modelo y extraer las frases clave con los valores por defecto, simplemente ejecuta el siguiente comando en la terminal de VS Code o cualquier otra terminal compatible:
```bash
py.exe .\main.py
```
Los resultados de la ejecución se guardarán en el fichero PromptRankLib.log.

## Parámetros de entrada
Si deseas modificar los parámetros del modelo, puedes hacerlo pasando argumentos en la línea de comandos. A continuación se detallan los principales parámetros disponibles:

--regular_expresion / --no-regular_expresion: Habilita (True) o deshabilita (False) el uso de expresiones regulares para la extracción de candidatos.

--greedy: Método de extracción de candidatos con expresión regular. Opciones disponibles: FIRST, LONGEST, COMBINED, NONE.

--title_graph_candidates_extraction: Título del gráfico generado para la extracción de candidatos.

--batch_size: Tamaño del lote (batch size) para evaluar el modelo. Valor por defecto: 128.

--encoder_header: Texto que precederá a la entrada en el codificador. Valor por defecto: "Texto:".

--prompt: Prompt que precederá al candidato en la entrada del modelo. Valor por defecto: "Este texto habla principalmente de ".

--max_len: Longitud máxima permitida por el tokenizador al codificar el texto. Valor por defecto: 512.

--model_version: Versión del modelo mT5 a utilizar. Opciones disponibles: base, small, large.

--length_factor: Factor de longitud para favorecer candidatos más largos o más cortos. Valor por defecto: 1.6.

--position_factor: Hiperparámetro que regula la penalización por posición. Valor por defecto: 1.2e8.

--enable_pos: Activa (True) o desactiva (False) la penalización por posición. Valor por defecto: False.

## Ejemplo de ejecución con parámetros personalizados:

Si deseas ejecutar el modelo con un batch_size de 64, utilizando el modelo large, y con el Prompt 'Este texto trata de ' puedes hacerlo con el siguiente comando:

```bash
py.exe .\main.py --batch_size 64 --model_version large --prompt 'Este texto trata de '
```

En la carpeta /vscode hay disponible un archivo launch.json para debuggear el modelo con diferentes valores de los parámetros de entrada.
