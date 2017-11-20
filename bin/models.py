import logging

import keras
from keras import Model
from keras.layers import Dense


def gen_stupid_ff_network(input_shape, output_shape):
    # Number of words in the word lookup index
    embedding_input_dim = input_shape

    # Number of output labels
    output_shape = output_shape

    logging.info('Creating stupid_ff_network. input shape: {}, output_shape: {}'.format(input_shape, output_shape))

    # Create architecture


    sequence_input = keras.Input(shape=(input_shape,))

    x = Dense(16, activation='linear')(sequence_input)
    x = Dense(64, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    preds = Dense(units=output_shape, activation='softmax')(x)



    # Compile architecture
    classification_model = Model(sequence_input, preds)
    classification_model.compile(loss='categorical_crossentropy',
                                 optimizer='rmsprop',
                                 metrics=['acc'])

    return classification_model