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

* build_dataset.py: Toma 2 o mas archivos .csv con con datos de series de tiempo, los sincroniza, genera bloques continuos(chunks) y almacena en formato .hdf5.

### nowcastlib/

* data_handlers.py; funciones de manipulación de datos de series de tiempo, generacion de bloques, sincronización.

* dnn_models_f1.py: funciones de construcción de modelos, usando función de pérdida custom "f1 macro"

* dnn_models.py: funciones de construcción de modelos, usando función de pérdida "cross-entropy"


## ./

### environment_tf_keras.yml
Configuracion de librerias para ambiente conda. Se utiliza tf_gpu (nightly version) para acelerar entrenamientos si es que hay CUDA y recursos GPU NVIDIA, de lo contrario funciona con CPU sin necesidad de configuración adicional.

### run_experiments.sh
Shell script linux para entrenar y evaluar multiples modelos para diferentes parametros.

# Paso a Paso Entrenamiento modelo

Procedimiento para entrenar y validar un modelo.

## Preparación de datos

1. Tener al menos 2 archivos .csv con información de las series de tiempo, con al menos 2 columnas cada uno. 
Cada archivo debe tener una de sus columnas representando el tiempo en un formato datetime preferentemente, el *sampling rate* no debe ser uniforme ni el mismo para los diferentes archivos. El resto de las columnas contienen los features o variables que consideramos y por ahora pueden ser numericas. No se esperan valores NaN o  Null. Sin embargo, si se permiten *gaps* o discontinuidades en el *sampling*.

1. Se debe generar un archivo de configuracion *data.json* que maneja las variables de lectura y conversiond de datos, usar como referencia *config/sma_uai_alldata.json*, en donde se deben definir los *data_sources* para cada archivo .csv con los nombres de columna que se considerarán(*field_list*), y la columna que representa el tiempo y su formato. Luego, en *chunk_config* se define el archivo de salida que contendra los datos en binario y convertidos en *chunks* con bloques continuos de datos consolidados y sincronizados de series de tiempo, en base a las variables *min_chunk_duration_sec*, *max_delta_between_blocks_sec*, and *sample_spacing* definidas en detalle en *src/build_dataset.py*

1. Ejecutar *python src/build_dataset.py data.json* que convertira los datos para su uso por el entrenamiento. Esto se debe realizar una sola vez si los multiples entrenamientos no implican cambios en la fuente de datos.

## Entrenamiento

1. Se debe generar un archivo de configuracion *model.json*, ver como referencia *model_config.json* que contiene variables especificas de la estructura de red, directorio de salida, e informacion para convertir la variable de salida en una variable categorica de n clases.

* results_path, experiment, name: rutas de salida
* lag: cuantos pasos en el pasado consideraremos para la prediccion, el tiempo dependera del *sample_spacing*.
* boost: pasos en el futuro a considerar para la predicción.
* dropout: rango recomendado 0.5 - 0.9, mientras mas bajo el entrenamiento demorará mas en converger.
* lstm_neurons: para ambas capas LSTM, recomendado 32-256 
* fcn_neurons: neuronas de capas densas, recomendado 32-256 
* fcn_layers: capas densas, recomendado 2-16
* n_class: numero clases a convertir la variable de salida. *Hardcoded* solo para 2,3 y 4 clases. 
* normalized_ranges_2class: lista con rangos que separan 2 clases despues de la normalizacion definida en la configuracion de entrenamiento(*target_field*).
* normalized_ranges_3class": lista con rangos para 3 clases
* normalized_ranges_4class": lista con rangos para 4 clases

1. Tambien debemos generar un archivo *training.json* con información especifica del entrenamiento a ejecutar, ver como referencia *training_config.json*.

* input (data_path,filename): ruta de salida
* stride: Es el espacio a avanzar para generar un nuevo *sample* de entrenamiento, siguiendo el mismo concepto usado en convoluciones. i.e un stride de 2 generará la mitad de muestras que usando un stride de 1.
* validation_fraction: (dejar en 0.01) no estamos haciendo cross validation.
* testing_fraction: Fraccion relativa de todos datos, para separar en el test set, el resto será el training set. Recomendado 0.1 a 0.3 dependiendo del volumen de datos.
* n_steps_boost_avg: pasos para promediar el valor maximo de la variable categorica en la ventana de tiempo = boost. Si es 1 implica que simplemente se ocupa el valor maximo encontrado.
* n_steps_lag_avg: no utilizado, dejar en 1.
* normalized_target_variance_range: maxima varianza permitida en un bloque de datos despues de la normalización. Esto es para encontrar y descartar bloques con *glitches* o errores/discontinuidades evidentes en los datos.
* target_field: nombre de campo a predecir el cual se convierte en variable categorica en la configuración del modelo.
* training_fields: se incorporan todos los nombres de columnas a incluir en el entrenamiento
* normalization: para cada variable se define la tupla [min,max] donde la variable v se transforma en (v-min)/(max-min)
* allowed_dyn_range: rango permitido de las variables despues de la normalización.
* batch_size: Mejora la convergencia y el ruido en la progresion de la evolucion de la pérdida(loss). Recomendado 16-64
* number_of_epochs: numero de epocas a entrenar, una epoca es recorrer todos los datos una vez. deberia entrenarse hasta ver que la funcion de perdida converge a un valor estable. Recomendado > 10.
* learning_rate: Tasa de aprendizaje, si es grande el entrenamiento es mas rapido pero la funcion de perdida es ruidosa y podria no converger, o converger fuera de lo optimo, muy bajo podria no mejorar el loss. Recomendado 0.00001 - 0.01
* beta_1 y 2: parametros de momento para ADAM
* class_weight_2class, class_weight_3class, class_weight_4class: Pesos asociados a la funcion de costo para cada clase, es util para mejorar la precision de clases subrepresentadas, donde el desbalance de clases es mayor a 1:10.
  
1. Correr *python experiment_train.py*, donde debemos definir los 3 archivos de configuracion dentro del codigo, ds_conf_path, tr_conf_path, y md_conf_path.

1. se generarán archivos de peso en la carpeta de salida *weights/saved-mode_XX.h5* donde *XX* es el peso para cada iteracion, y se generarán *number_of_epochs* archivos de pesos. Tambien en la salida se genera una imagen "model_loss.png" con la funcion de pérdida. Finalmente, tambien se genera un archivo *report.txt* donde se evalua el ultimo peso generado con el testing set. 

## Testing

Se ejecuta *python experiment_test.py* que tiene la misma entrada que el script de entrenamiento. Lo que hace es recorrer todos los archivos de peso generados en el entrenamiento y encontrar el que entrega el mejor macro f1-score total, tanto global, como para el caso en donde ninguna clase sea nula. Este archivo de peso deberia utilizarse para futuras inferencias del modelo entrenado.


