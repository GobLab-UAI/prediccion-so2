from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import tensorflow as tf

import os
import sys
import json
import pandas
import numpy
import random

# PARA USAR FUNCION DE PERDIDA F1 SE CARGA dnn_models_f1
from nowcastlib.dnn_models_f1 import build_categorical_model

# PARA USAR FUNCION DE PERDIDA CLASICA CROSS_ENTROPY SE CARGA dnn_models
#from nowcastlib.dnn_models_f1 import build_categorical_model


from nowcastlib.data_handlers import MaxCategorical

from keras.callbacks import ModelCheckpoint

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


import matplotlib.pyplot as plt

# Esto reduce el consumo de memoria GPU, recordar que si no hay GPU igual deberia correr el codigo
def configure_tf():
    '''function to make tensorflow don't take up all GPU memory
    '''
    # experimental, configure GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print("gpu:",gpus)
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

# Definicion de todos los parametros
def configure_handler(ds_config, tr_config, md_config):
    '''
    '''    
    basepath = tr_config["input"]["data_path"]
    hdf_path = os.path.join(basepath, tr_config["input"]["filename"])
    
    ds_config = tr_config['dataset']
    stride = ds_config['stride']
    max_blocks = int(ds_config.get('max_blocks', -1))
    n_boost_avg = int(ds_config['n_steps_boost_avg'])
    n_lag_avg = int(ds_config['n_steps_lag_avg'])
    
    training_fields = ds_config['training_fields']
    n_variables = len(training_fields)
    target_field = ds_config['target_field']
    val_frac = ds_config['validation_fraction']
    test_frac = ds_config['testing_fraction']
    norm_info = ds_config['normalization']
    norm_variance_range = ds_config['normalized_target_variance_range']
    allowed_dyn_range = ds_config['allowed_dyn_range']

    
    number_class = int(md_config['n_class'])

    if number_class == 2:
        ranges = md_config['normalized_ranges_2class']
    elif number_class == 3:
        ranges = md_config['normalized_ranges_3class']
    elif number_class == 4:
        ranges = md_config['normalized_ranges_4class']
    else:
        raise ValueError("number of classes is not supported")

    lag = int(md_config['lag'])
    boost = int(md_config['boost'])

    handler = MaxCategorical(hdf_path, training_fields, target_field)
    handler.set_stride(stride)
    handler.set_trainval_frac(1.0 - test_frac)
    handler.set_lag(lag)
    handler.set_boost(boost)
    handler.set_ranges(ranges)
    handler.set_boost_avg_steps(n_boost_avg)
    handler.set_normalization(norm_info)
    handler.set_allowed_dyn_range(allowed_dyn_range[0], allowed_dyn_range[1])
    handler.set_allowed_std_range(norm_variance_range[0], norm_variance_range[1])
    print('building dataset')
    handler.build_dataset(
        filter_by_std=True)
    print('done.')    
    return handler
    
def configure_model(ds_config, tr_config, md_config):
    '''
    '''
    ds_config = tr_config['dataset']
    training_fields = ds_config['training_fields']
    n_variables = len(training_fields)
    
    number_class = int(md_config['n_class'])
    if number_class == 2:
        ranges = md_config['normalized_ranges_2class']
    elif number_class == 3:
        ranges = md_config['normalized_ranges_3class']
    elif number_class == 4:
        ranges = md_config['normalized_ranges_4class']
    else:
        raise ValueError("number of classes is not supported")

    lag = int(md_config['lag'])
    boost = int(md_config['boost'])
    lstm_neurons = int(md_config['lstm_neurons'])
    fcn_neurons = int(md_config['fcn_neurons'])
    fcn_layers = int(md_config['fcn_layers'])
    fcn_dropout = float(md_config['dropout'])
    
    train_config = tr_config['training']
    n_epochs = int(train_config['number_of_epochs'])
    batch_size = int(train_config['batch_size'])
    learning_rate = float(train_config['learning_rate'])
    beta_1 = float(train_config['beta_1'])
    beta_2 = float(train_config['beta_2'])

    # define model 
    compiled_model = build_categorical_model(lag, n_variables, number_class, lstm_neurons, fcn_neurons, fcn_layers, 
    fcn_dropout, learning_rate, beta_1, beta_2)
    
    compiled_model.summary()
    return compiled_model

if __name__ == '__main__':
    '''
    '''
    configure_tf()
    # Semillas fijas para reproducir resultados
    numpy.random.seed(2)
    random.seed(2)
    
    # parametros de entrada
    ds_conf_path = 'config/sma_uai_alldata.json'
    tr_conf_path = 'config/training_config.json'
    md_conf_path = 'config/model_config.json'
    
    # load configurations
    master_config = dict()
    for path in [ds_conf_path, tr_conf_path, md_conf_path]:
        config_file = open(path, 'r')
        config = json.loads(config_file.read())
        config_file.close()
        master_config.update(config)
            
    ds_conf = master_config["dataset"]
    tr_conf = master_config["training"]
    md_conf = master_config["model"]

    # read in positional parameters
    estacion = sys.argv[1]
    n_class = sys.argv[2]
    boost = sys.argv[3]

    # change values of the json corresponding to the loop
    md_conf['name'] = str(md_conf['experiment']) + '/' + str(estacion) + '_c' + str(n_class) + '_h' + str(boost)
    md_conf['boost'] = str(boost)
    md_conf['n_class'] = str(n_class) 
    tr_conf['dataset']['target_field'] = str(estacion) + '_so2_ppb'

    # create output folder. abort if folder exists
    output_path = os.path.join(md_conf['results_path'], md_conf['name'])
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
        print("created folder : ", output_path)
    else:
        print(output_path, "folder already exists.")
        #sys.exit(-1)

    # create model-checkpoint folder. overwrite weights if already exists
    weights_path = os.path.join(output_path, "weights")
    if not os.path.isdir(weights_path):
        os.makedirs(weights_path)
        print("created folder : ", weights_path)
    else:
        print(weights_path, "folder already exists.")

    ncast_model = configure_model(ds_conf, tr_conf, md_conf)
    data_handler = configure_handler(ds_conf, tr_conf, md_conf)
    tkeys = data_handler.get_testing_keys()
    
    # define model checkpoint
    weights_filename = os.path.join(weights_path, "saved-model_{epoch:02d}.h5")
    checkpoint = ModelCheckpoint(
        weights_filename, 
        monitor='loss', 
        verbose=1,
        save_best_only=False, 
        save_weights_only=True,
        mode='min')    
    
    # get training set
    training_set_x, training_set_y = data_handler.get_training_set()
    train_x1 = training_set_x[0]
    train_y1 = training_set_y[0]

    # get testing set
    testing_set_x, testing_set_y = data_handler.get_testing_set()
    test_x1 = testing_set_x[0]
    test_y1 = testing_set_y[0]

    #cuenta de elementos para entrenar por clase.
    nclass=train_y1.shape[1]
    nobj=train_y1.shape[0]
    ccla=numpy.zeros(nclass)
    print("train count classes:::")
    for i in range(nobj):
        for j in range(nclass):
            ccla[j]=ccla[j]+train_y1[i][j] 
    print(ccla)
    nclass=test_y1.shape[1]
    nobj=test_y1.shape[0]
    ccla=numpy.zeros(nclass)
    print("test count classes:::")
    for i in range(nobj):
        for j in range(nclass):
            ccla[j]=ccla[j]+test_y1[i][j]
    print(ccla)

    number_class = int(md_conf['n_class'])
    list_n_class = [i for i in range(number_class)]
    if number_class == 2:
        class_weight = tr_conf['training']['class_weight_2class']
        class_weight = dict(zip(list_n_class, list(class_weight.values())))
    elif number_class == 3:
        class_weight = tr_conf['training']['class_weight_3class']
        class_weight = dict(zip(list_n_class, list(class_weight.values())))
    elif number_class == 4:
        class_weight = tr_conf['training']['class_weight_4class']
        class_weight = dict(zip(list_n_class, list(class_weight.values())))
    else:
        raise ValueError("number of classes is not supported")

    # train model
    # class weights define el peso que se debe asignar a los elementos de cada clase
    # en el calculo de la loss function. es bastante sensible y determina la convergencia
    # por ejemplo si el numero de elementos de clase 1 es 10 veces mayor que la clase 2,
    # y classweights=(1,1) resultaria en precision ~90% clase 1 y muy bajo la clase 2
    # si ahora para el mismo entrenamiento seleccionamos
    # classweithds=(1,10) resultara probablemente en algo asi como precision clase1=70%, precision clase2=60%
    # para desproporciones muy grandes entre clases de 2 o mas ordenes de magnitud, este balance no funciona muy bien
    # se recomienda no usar un desbalance mayor que 50-60 aunque la desproporcion sea mucho mayor, sino nunca
    # bajaria la loss function y quizas no se observaria convergencia.
    train_history = ncast_model.fit(
        train_x1, train_y1,
        batch_size=int(tr_conf['training']['batch_size']), 
        epochs=int(tr_conf['training']['number_of_epochs']), 
        callbacks=[checkpoint],
        shuffle=True,
        class_weight = class_weight)

    # save training history
    loss_plot_path = os.path.join(output_path, "model_loss.png")
    loss = train_history.history['loss']
    import matplotlib.pyplot as plt
    plt.plot(loss)
    plt.savefig(loss_plot_path)



    # aca en realidad puede ser mas interesante mirar la prediccion contra train_x1 
    # para saber que tan bien aprendio a modelar el problema y si hay hay overfit!!!

    # test model
    pred_y1= ncast_model.predict(test_x1)
    rep = classification_report(test_y1.argmax(axis=1), pred_y1.argmax(axis=1))
    str_rep = "{}".format(rep)
    print(str_rep)
    report_path = os.path.join(output_path, "report.txt")
    report_file = open(report_path, 'w')
    report_file.write(str_rep)
    report_file.close()


    #conf_matrix = confusion_matrix(test_y1.argmax(axis=1), pred_y1.argmax(axis=1))
    #print(conf_matrix)

    '''        
    # save output as npz file
    numpy.savez(os.path.join(tr_conf['output']['path'], "test_and_loss.npz"),
        model_loss=loss,
        testing_keys=data_handler.get_testing_keys(),
        training_keys=data_handler.get_training_keys(),
        test_x = test_x,
        test_y = test_y)
    '''
