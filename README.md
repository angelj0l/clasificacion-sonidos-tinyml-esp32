# clasificacion-sonidos-tinyml-esp32
Repositorio con el código fuente y codigo de entrenamiento para la clasificación de ruido urbano con Edge Computing en ESP32.

Clasificación de Sonido Urbano con TinyML: Pipeline de Entrenamiento y Cuantización
Este repositorio contiene el notebook principal de Jupyter utilizado para el desarrollo, entrenamiento y optimización de un modelo de aprendizaje automático convolucional (CNN) diseñado para clasificar entornos acústicos (Música, Tráfico y Neutro). El objetivo final de este flujo de trabajo es exportar un modelo altamente eficiente para su despliegue en microcontroladores de arquitectura embebida (específicamente la familia ESP32) utilizando TensorFlow Lite for Microcontrollers.

📋 Estructura del Flujo de Trabajo (Jupyter Notebook)
El notebook documenta un proceso iterativo de ensayo y error, optimización y evaluación técnica. A continuación, se detalla el paso a paso del código incluido:

1. Exploración de Datos y Visualización Dimensional (t-SNE & PCA)
Las primeras celdas del notebook están dedicadas a la validación visual de los espectrogramas Log-Mel almacenados en formato binario (.bin).

Se cargan los vectores aplanados (7826 puntos de datos por archivo).

Se implementa una reducción de dimensionalidad combinando PCA (para compresión inicial) y t-SNE (para proyección a 2D).

Se genera un gráfico interactivo utilizando plotly para observar la separabilidad espacial de las tres clases acústicas.

2. Limpieza del Dataset (Depuración de Anomalías)
Dado que los datos fueron capturados en campo, se implementa una etapa de "caza de intrusos":

Se entrena una CNN preliminar rápida (10 épocas) con un subconjunto balanceado de datos.

El modelo evalúa la totalidad del dataset buscando errores de etiquetado (archivos donde la red se equivoca con una confianza superior al 75%).

Se despliega un script interactivo con IPython.display.Audio que permite escuchar los audios sospechosos y decidir en tiempo real si se eliminan (b), se cambian de clase (c) o se mantienen (m). Esta limpieza eliminó 24 archivos ruidosos y reubicó 5.

3. Corrección Crítica de Memoria (C++ vs. Python)
Durante el desarrollo se identificó y corrigió un bug fundamental en la lectura de las matrices de audio exportadas por el hardware:

El problema: Inicialmente, Python interpretaba los datos binarios de forma lineal (reshape(N_FRAMES, N_MELS)), lo que mezclaba el eje temporal con el frecuencial al no coincidir con el almacenamiento en memoria plana de C++.

La solución: Se ajustó la lectura para coincidir con la estructura original del array en C (float mel[N_MELS][N_FRAMES]), aplicando posteriormente una transposición (.T) de la matriz y añadiendo la dimensión de canal para obtener el formato (301, 26, 1) esperado por la red neuronal.

4. Entrenamiento de la Arquitectura Definitiva en float32
Tras balancear las clases (791 muestras por categoría) mediante undersampling y corregir la forma de los tensores, se procedió al entrenamiento formal:

Escalado: Se implementó una normalización dividiendo los valores de entrada por 15.0 para evitar la inestabilidad del gradiente (Loss inicial elevado).

Arquitectura: Se utilizó una CNN secuencial de dos capas convolucionales con MaxPooling2D, seguida de una capa densa de 64 neuronas.

Ajuste Fino: Tras varias pruebas iterativas, se incorporó un Dropout ligero (20%) en la capa densa y se extendió el entrenamiento a 70 épocas.

Resultados de Línea Base: El modelo original en coma flotante alcanzó una exactitud (accuracy) en validación superior al 93%, generando su respectiva matriz de confusión y reporte de clasificación para establecer la línea base de comparación. El artefacto se guardó como modelo_clasificacion_audio.h5.

5. Cuantización a Enteros de 8 bits (INT8)
Para garantizar la viabilidad del despliegue en dispositivos con memoria RAM limitada (PSRAM), el modelo se sometió a un proceso de Full Integer Quantization:

Se utilizó TFLiteConverter forzando las operaciones, entradas y salidas al formato INT8 (tf.int8).

Se generó un Dataset Representativo utilizando las muestras de entrenamiento (previamente escaladas por 15.0) para calibrar correctamente los rangos dinámicos de los pesos y activaciones.

El modelo resultante se exportó como modelo_audio_int8.tflite.

6. Validación de Degradación (Simulación INT8)
Para asegurar que la compresión algorítmica no destruyera el rendimiento del sistema:

Se extrajeron los parámetros de cuantización (Scale y Zero Point) directamente del intérprete de TFLite.

Se reconstruyó la lógica de inferencia cuantizando manualmente un lote de validación y procesándolo a través del modelo de 8 bits.

Hallazgo clave: Las métricas de desempeño y la matriz de confusión del modelo INT8 resultaron idénticas a las del modelo en float32. La cuantización posterior al entrenamiento mantuvo márgenes de decisión lo suficientemente amplios como para no alterar ni una sola predicción en el conjunto de prueba (93.47% de exactitud).

7. Depuración de Inferencia (Hardware in the Loop)
La etapa final del notebook incluye pruebas de inyección directa de volcados de memoria (Dumps) provenientes del hardware:

Se leen archivos .txt planos (esp32_dump.txt, esp32_dump_2.txt) que contienen las matrices crudas en formato INT8 generadas por el preprocesamiento de la señal en el microcontrolador.

Estos tensores se pasan por el intérprete de TFLite en Python para diagnosticar discrepancias de resultados entre la inferencia del computador y la del dispositivo físico.

Esto permitió confirmar que el preprocesamiento en el nodo físico y el consumo del modelo estaban perfectamente alineados, arrojando probabilidades coherentes (ej. 99.6% Neutro) sin desbordamientos de enteros (overflow).
