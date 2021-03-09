# Predicción Calidad del Aire

Repositorio del proyecto Inteligencia artificial para el cuidado del medio ambiente y la salud de la población: modelo predictivo para episodios críticos de contaminación en el aire para una zona industrial de Chile.

El modelo esta desarrollado en Keras con un backend de Tensorflow-GPU y posee las siguientes caracteristicas:

1. Modelo LSTM + capas densas, siguiendo la literatura reciente para abordar problemas de predicción multivariado de series de tiempo.

1. La salida del modelo contempla la discretización de una variable continua en *n_classes* neuronas, donde cada neurona devuelve una probabilidad de resultado(softmax).

1. Archivos de configuracion que permiten cambiar de forma dinamica estructura de red, y datos de entrada/salida

1. Los diferentes variables de entrada pueden no estar sincronizadas temporalmente, y el codigo puede hacer un *re-sample* y sincronizar los datos para generar los lag y boost.

1. Los dataset de entrenamiento y testing se dividen en bloques random no correlacionados temporalmente en escalas del lag/boost para evitar sesgos/overfit.

1. El entrenamiento con GPUs es ~10x mas rápido que en CPU(i7CPU vs TitanX).


# Contenido y Descripción

## /Config
Archivos de configuración utilizados para entrenamiento, testing e inferencia.
Estan en formato JSON y se deben generar 3 archivos minimo:

* data.json: contiene los archivo de datos de entrada y sus campos que luego seran sincronizados t convertidos en *chunks* de datos continuos en un archivo de salida HDF5(formato que permite leer datos mucho mas rápido y es práctico en situaciones de múltiples realizaciones). Tambien, se define la variable temporal y el formato de transformacion en DateTime.

* model.json: parametros de la estructura del modelo, rutas escritura resultados, lag, boost, y rangos de clases. 

* training.json: contiene la ruta del archivo HDF5 con datos sincronizados, variables especificas de entrenamiento, y los campos utilizados como entrada y el campo a predecir en categorias. Tambien, se puede incluir una normalizacion a cada variable que se ejecuta antes del entrenamiento.


## /Data/

* csv/ series de tiempo con variables de entrada, pueden ser múltiples archivos, y el sampling rate puede ser arbitrario y asincrono.

* hdf5/ datos consolidados y sincronizados en bloques o *chunks* para utilizar en los entrenamientos.

## /src

* experiment_train.py: código entrenamiento

* experiment_test.py: código testing

* generator_inference_filetest.py: código generico inferencias 

### nowcastlib/

* data_handlers.py; funciones de manipulación de datos de series de tiempo, generacion de bloques, sincronización.

* dnn_models_f1.py: funciones de construcción de modelos, usando función de pérdida custom "f1 macro"

* dnn_models.py: funciones de construcción de modelos, usando función de pérdida "cross-entropy"


## ./

### environment_tf_keras.yml
Configuracion de librerias para ambiente conda. Se utiliza tf_gpu (nightly version) para acelerar entrenamientos si es que hay CUDA y recursos GPU NVIDIA, de lo contrario funciona con CPU sin necesidad de configuración adicional.

### run_experiments.sh
Shell script linux para entrenar y evaluar multiples modelos para diferentes parametros.

