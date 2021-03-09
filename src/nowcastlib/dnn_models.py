# Colaboradores: Pedro Fluxá, Roberto González


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.optimizers import Adam


# Estructura clasica 2 capas lstm, y 8 capas densas
def build_categorical_model(lag, n_inputs, n_ranges, 
    lstm_neurons=128, #64 def 
    fcn_neurons=256, fcn_layers=8, fcn_dropout=0.5,
    learning_rate=0.00005, beta_1=0.9, beta_2=0.999):

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
        loss='categorical_crossentropy', 
        optimizer=Adam(lr=learning_rate, beta_1=beta_1, beta_2=beta_2),
        metrics='categorical_accuracy')
    return model

