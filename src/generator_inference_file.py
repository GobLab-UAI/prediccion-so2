from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import tensorflow as tf

import pickle

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
    boost_out = int(md_config['boost_out'])
    
    handler = MaxCategorical(hdf_path, training_fields, target_field)
    handler.set_stride(stride)
    handler.set_trainval_frac(1.0 - test_frac)
    handler.set_lag(lag)
    handler.set_boost(boost)
    handler.set_boost_out(boost_out)
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
    
    # create model-checkpoint folder. overwrite weights if already exists
    file_path = os.path.join("./inference_files")
    if not os.path.isdir(file_path):
        os.makedirs(file_path)
        print("created folder : ", file_path)
    else:
        print(file_path, "folder already exists.")

    data_handler = configure_handler(ds_conf, tr_conf, md_conf)
    tkeys = data_handler.get_testing_keys()
        
    # get training set
    training_set_x, training_set_y = data_handler.get_training_set()
    train_x1 = training_set_x[0]
    train_y1 = training_set_y[0]

    # get testing set
    testing_set_x, testing_set_y = data_handler.get_testing_set()
    test_x1 = testing_set_x[0]
    test_y1 = testing_set_y[0]
    
    # Save train set for future inferences
    outfile = '_inference_c' + str(n_class) + '_h' + str(boost) + '.pkl'
    with open('inference_files/trainx'+ outfile, 'wb') as handle:
        pickle.dump(train_x1, handle)
    with open('inference_files/trainy'+ outfile, 'wb') as handle:
        pickle.dump(train_y1, handle)

    # Save test set for future inferences
    outfile = '_inference_c' + str(n_class) + '_h' + str(boost) + '.pkl'
    with open('inference_files/testx'+ outfile, 'wb') as handle:
        pickle.dump(test_x1, handle)
    with open('inference_files/testy'+ outfile, 'wb') as handle:
        pickle.dump(test_y1, handle)
        
