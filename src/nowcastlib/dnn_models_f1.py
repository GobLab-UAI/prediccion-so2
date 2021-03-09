# Colaboradores: Pedro Fluxá, Roberto González


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.optimizers import Adam



# ---- Funcion de perdida manual por f1 macro -----
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

# --------------------------------------------------


def build_categorical_model(lag, n_inputs, n_ranges, lstm_neurons, fcn_neurons, fcn_layers, fcn_dropout,
    learning_rate, beta_1, beta_2):

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
        # clasico loss para categorical
        # loss='categorical_crossentropy', 
        # loss f1 manual
        loss=f1_loss,
        optimizer=Adam(lr=learning_rate, beta_1=beta_1, beta_2=beta_2),
        # clasico para categorical
        # metrics='categorical_accuracy')
        # para loss f1
        metrics=['accuracy',f1])
    return model

