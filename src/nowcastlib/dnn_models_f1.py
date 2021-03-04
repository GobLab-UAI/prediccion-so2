from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.optimizers import Adam

"""
import tensorflow as tf
def focal_loss(gamma=2., alpha=4.):

    gamma = float(gamma)
    alpha = float(alpha)

    def focal_loss_fixed(y_true, y_pred):
            #y_true {tensor} -- ground truth labels, shape of [batch_size, num_cls]
            #y_pred {tensor} -- model's output, shape of [batch_size, num_cls]

            #gamma {float} -- (default: {2.0})
            #alpha {float} -- (default: {4.0})

        epsilon = 1.e-9
        y_true = tf.convert_to_tensor(y_true, tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, tf.float32)

        model_out = tf.add(y_pred, epsilon)
        ce = tf.multiply(y_true, -tf.math.log(model_out))
        weight = tf.multiply(y_true, tf.pow(tf.subtract(1., model_out), gamma))
        fl = tf.multiply(alpha, tf.multiply(weight, ce))
        reduced_fl = tf.reduce_max(fl, axis=1)
        return tf.reduce_mean(reduced_fl)
    return focal_loss_fixed
"""
import tensorflow as tf
import keras.backend as K
def f1(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)

def f1_loss(y_true, y_pred):
    
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return 1 - K.mean(f1)



def build_continuous_model(lag, n_inputs, 
    lstm_neurons=64, 
    fcn_neurons=512, fcn_layers=8, fcn_dropout=0.5,
    learning_rate=1e-4, beta_1=0.9, beta_2=0.999):
    '''
    keyword arguments default to best model
    '''
    use_dropout = fcn_dropout > 0
    inx1 = Input(shape=(lag, n_inputs))
    x = LSTM(lstm_neurons, return_sequences=True, activation='tanh')(inx1)
    x = LSTM(lstm_neurons, return_sequences=False, activation='tanh')(x)
    x = Flatten()(x)
    for _ in range(fcn_layers):
        x = Dense(fcn_neurons, activation='relu')(x)
        if use_dropout:
            x = Dropout(fcn_dropout)(x) 
    p = Dense(1, name='rng_pred', activation='linear')(x)
    
    model = Model(inputs=[inx1,], outputs=[p,])
    model.compile(
        loss='mse', 
        optimizer=Adam(lr=learning_rate, beta_1=beta_1, beta_2=beta_2),
        metrics='mape')
    return model

def build_categorical_model(lag, n_inputs, n_ranges, lstm_neurons, fcn_neurons, fcn_layers, fcn_dropout,
    learning_rate, beta_1, beta_2):
    '''
    keyword arguments default to best model
    '''
    use_dropout = fcn_dropout > 0
    inx1 = Input(shape=(lag, n_inputs))
    x = LSTM(lstm_neurons, return_sequences=True, activation='tanh')(inx1)
    x = LSTM(lstm_neurons, return_sequences=False, activation='tanh')(x)
    x = Flatten()(x)
    for _ in range(fcn_layers):
        x = Dense(fcn_neurons, activation='relu')(x)
        if use_dropout:
            x = Dropout(fcn_dropout)(x) 
    p = Dense(n_ranges, name='rng_pred', activation='softmax')(x)
    
    model = Model(inputs=[inx1,], outputs=[p,])
    model.compile(
        #loss='categorical_crossentropy', 
        #loss=focal_loss(alpha=1), #para probar focal loss
        loss=f1_loss,
        optimizer=Adam(lr=learning_rate, beta_1=beta_1, beta_2=beta_2),
        #metrics='categorical_accuracy')
        metrics=['accuracy',f1])
    return model

def build_trend_model(lag, n_inputs, n_ranges_up, n_ranges_down,
    lstm_neurons=64, 
    fcn_neurons=512, fcn_layers=8, fcn_dropout=0.5,
    learning_rate=1e-4, beta_1=0.9, beta_2=0.999):
    '''
    keyword arguments default to best model
    '''
    use_dropout = fcn_dropout > 0
    inx1 = Input(shape=(lag, n_inputs))
    x = LSTM(lstm_neurons, return_sequences=True, activation='tanh')(inx1)
    x = LSTM(lstm_neurons, return_sequences=False, activation='tanh')(x)
    x = Flatten()(x)
    for _ in range(fcn_layers):
        x = Dense(fcn_neurons, activation='relu')(x)
        if use_dropout:
            x = Dropout(fcn_dropout)(x) 
    # range up
    xu = Dense(32, activation='relu')(x)
    xu = Dense(32, activation='relu')(xu)
    pu = Dense(n_ranges_up, activation='softmax', name='pu')(xu)
    # range down
    xd = Dense(32, activation='relu')(x)
    xd = Dense(32, activation='relu')(xd)
    pd = Dense(n_ranges_down, activation='softmax', name='pd')(xd)

    model = Model(inputs=[inx1,], outputs=[pu, pd])
    model.compile(
            loss='categorical_crossentropy', 
            optimizer=Adam(lr=learning_rate, beta_1=beta_1, beta_2=beta_2),
            metrics='categorical_accuracy')
    return model
