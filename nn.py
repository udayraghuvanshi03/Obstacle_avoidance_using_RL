from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import LSTM
from tensorflow.keras.callbacks import Callback


class LossHistory(Callback):
    def on_train_begin(self, logs=None):
        if logs is None:
            logs = {}
        self.losses = []

    def on_batch_end(self, batch, logs=None):
        if logs is None:
            logs = {}
        self.losses.append(logs.get('loss'))


def neural_net(num_sensors, params, load=''):
    model = Sequential()

    model.add(Dense(
        params[0], kernel_initializer='lecun_uniform', input_shape=(num_sensors,)
    ))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(params[1], kernel_initializer='lecun_uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(3, kernel_initializer='lecun_uniform'))
    model.add(Activation('linear'))

    rms = RMSprop()
    model.compile(loss='mse', optimizer=rms)

    if load:
        model.load_weights(load)

    return model
