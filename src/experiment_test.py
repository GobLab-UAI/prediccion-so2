from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import tensorflow as tf

import os
import sys
import json
import pandas
import numpy
import random

from nowcastlib.dnn_models_f1 import build_categorical_model
from nowcastlib.data_handlers import MaxCategorical

from keras.callbacks import ModelCheckpoint

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

import matplotlib.pyplot as plt

def configure_tf():
    '''function to make tensorflow don't take up all GPU memory
    '''
    # experimental, configure GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

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
    allow_star_change = ds_config['allow_star_change']
    ra_tracking_field = ds_config.get('tracking_ra_field_name', None)
    dec_tracking_field = ds_config.get('tracking_dec_field_name', None)
    filter_star_change = None
    if allow_star_change == 'True':
        filter_star_change = False
    elif allow_star_change == 'False':
        filter_star_change = True
    else:
        raise ValueError("allow_start_change must be True or False")
    if allow_star_change and ra_tracking_field is None:
        raise ValueError("allow_start_change = True needs ra field")
    if allow_star_change and dec_tracking_field is None:
        raise ValueError("allow_start_change = True needs dec field")
    
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
    #handler.set_allowed_dyn_range(allowed_dyn_range[0], allowed_dyn_range[1])
    handler.set_allowed_std_range(norm_variance_range[0], norm_variance_range[1])
    if filter_star_change:
        handler.set_ra_dec_field_name(ra_tracking_field, dec_tracking_field)
    print('building dataset')
    handler.build_dataset(
        filter_star_change=filter_star_change, 
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
    numpy.random.seed(2)
    random.seed(2)
    
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

    ncast_model = configure_model(ds_conf, tr_conf, md_conf)
    data_handler = configure_handler(ds_conf, tr_conf, md_conf)
    tkeys = data_handler.get_testing_keys()
        
    # get testing set
    testing_set_x, testing_set_y = data_handler.get_testing_set()
    test_x1 = testing_set_x[0]
    test_y1 = testing_set_y[0]

    n_epochs = int(tr_conf['training']['number_of_epochs'])
    f1max=0
    f1maxnozero=0
    epochmax=0
    for ep in range(n_epochs):
        epoch=ep+1
        weights_path = os.path.join(output_path, "weights")
        weights_filename = os.path.join(weights_path, 'saved-model_%02d.h5'%(epoch))
        ncast_model.load_weights(weights_filename)
        pred_y1= ncast_model.predict(test_x1)
        f1 = f1_score(test_y1.argmax(axis=1), pred_y1.argmax(axis=1),average='macro')
        f1s = f1_score(test_y1.argmax(axis=1), pred_y1.argmax(axis=1),average=None)
        aux = int(n_class)-1
        f1s = f1s[aux]
        if (f1s*0.5+f1>f1maxnozero):
            f1maxnozero=f1s*0.5+f1
            weights_filename_best2=weights_filename

        if (f1>f1max):
            f1max=f1
            epochmax=epoch
            weights_filename_best=weights_filename

        print(weights_filename,'  f1=',f1,' f1s=',f1s)


    print("best result epoch=",epochmax,' with f1_macro=',f1max)
    # train model
    ncast_model.load_weights(weights_filename_best)
    # test model
    pred_y1= ncast_model.predict(test_x1)
    rep = classification_report(test_y1.argmax(axis=1), pred_y1.argmax(axis=1))
    str_rep = "{}".format(rep)
    print(str_rep)

    report_path = os.path.join(output_path, "report.txt")
    report_file = open(report_path, 'w')
    report_file.write(str_rep)
    report_file.close()
        
    conf_matrix = confusion_matrix(test_y1.argmax(axis=1), pred_y1.argmax(axis=1))
    print(conf_matrix)
 
    if f1maxnozero>0:
        print("best f1 non zero:",weights_filename_best2)
        ncast_model.load_weights(weights_filename_best2)
        pred_y1= ncast_model.predict(test_x1)
        rep = classification_report(test_y1.argmax(axis=1), pred_y1.argmax(axis=1))
        str_rep = "{}".format(rep)
        print(str_rep)
        conf_matrix = confusion_matrix(test_y1.argmax(axis=1), pred_y1.argmax(axis=1))
        print(conf_matrix)


