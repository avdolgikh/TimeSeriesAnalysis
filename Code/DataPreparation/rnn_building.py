import json
import numpy as np
import keras
import tensorflow

from keras.layers import Input, Dense, RNN, GRU, SimpleRNN, LSTM
from keras.models import Model

import matplotlib.pyplot as plt

INPUT_LENGTH = 7
RECURRENT_LAYER_LENGTH = 5
N_ORIGINAL_POINTS = 1000
NOISE = 0 #0.05
N_TRAIN_POINTS = 500
N_PREDICTED_POINTS = 10000
N_EPOCHS = 30

def build_henon():
    x = [.0]
    y = [.0]
    a = 1.4
    b = .3
    for i in range(N_ORIGINAL_POINTS - 1):
        x.append(1 - a * x[i]**2 + y[i])
        y.append(b * x[i])

    return np.array(x), np.array(y)
    
def get_train_set(x):
    # add noise:
    x = x + np.random.randn(N_ORIGINAL_POINTS) * NOISE

    input = []
    output = []
    for i in range(N_TRAIN_POINTS):
        input.append([x[i : (i+INPUT_LENGTH)]])
        output.append(x[i+INPUT_LENGTH])    
    input = np.array(input)
    output = np.array(output)
    return input, output, x


if __name__ == '__main__':
    x, y = build_henon()
    
    input_data, output_data, x_noised = get_train_set(x)

    # https://keras.io/layers/recurrent/
    inputs = Input(shape=(None, INPUT_LENGTH))
    #rnn = SimpleRNN(RECURRENT_LAYER_LENGTH, activation="sigmoid", use_bias=True)(inputs)
    rnn = LSTM(RECURRENT_LAYER_LENGTH, activation="sigmoid", recurrent_activation="sigmoid",
                use_bias=True, dropout=0.0, recurrent_dropout=0.01)(inputs)

    # TODO: Add Embedding layer?
    #rnn2 = LSTM(RECURRENT_LAYER_LENGTH, activation="sigmoid", recurrent_activation="sigmoid",
    #            use_bias=True, dropout=0.0, recurrent_dropout=0.01)(rnn)

    #rnn = GRU(RECURRENT_LAYER_LENGTH, activation="sigmoid", recurrent_activation="sigmoid", use_bias=True,
    #            dropout=0.0, recurrent_dropout=0.0)(inputs)
    output = Dense(1, activation="linear", use_bias=True)(rnn)

    model = Model(inputs=inputs, outputs=output)

    # https://keras.io/optimizers/
    # "rmsprop" is usually a good choice for recurrent neural networks.
    # "adadelta"
    # "nadam". Much like Adam is essentially RMSprop with momentum, Nadam is Adam RMSprop with Nesterov momentum.
    optimizer = 'sgd'
    model.compile(loss=keras.losses.mean_squared_error,
                  optimizer=optimizer,
                  metrics=[keras.metrics.mean_squared_error])
    # SGD. loss is big (0.0132). But the shape is very good!!! Not blured. Just shifted a bit.

    print(model.summary())

    np.random.seed(19)
    tensorflow.set_random_seed(19)
    model.fit(input_data, output_data, epochs=N_EPOCHS, verbose=1)

    # TODO: save the model!

    x_pred = input_data[0][0] # the first seven data measures
    for i in range(N_PREDICTED_POINTS):
        x_pred = np.append(x_pred, model.predict(np.array([[x_pred[i:(i+INPUT_LENGTH)]]]))[0][0])
    

    figure = plt.figure(figsize=(15, 11), dpi=200)
    title = "RNN_Henon_LSTM_sigmoid_recdropout0.01_noise{}_train{}_epochs{}_rnd19".format(NOISE, N_TRAIN_POINTS, N_EPOCHS)
    plt.title(title)
    plt.xlabel("x(t)")
    plt.ylabel("x(t+1)")
    #plt.scatter(x_noised[:-1], x_noised[1:], c='#3F00FF', s=2)
    plt.scatter(x[:-1], x[1:], c='black', s=2)
    plt.scatter(x_pred[:-1], x_pred[1:], c='red', s=2)
    plt.grid(True)
    plt.show()
    #figure.savefig(r'..\..\..\Docs\{}.png'.format(title), bbox_inches='tight')