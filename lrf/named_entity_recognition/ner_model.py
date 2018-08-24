import keras
from keras.models import Model
from keras.layers.convolutional import Conv2D
from keras.initializers import RandomUniform
from keras.layers.recurrent import LSTM
import pandas as pd
import numpy as np
import sys
from os import path
# sys.path.append(path.abspath('../../keras-contrib'))
sys.path.append(path.abspath('/home/onespace/ankita/keras-contrib'))
from keras_contrib.layers import crf
import matplotlib.pyplot as plt
import tensorflow as tf
from lrf.configs import lrf_config
from lrf.named_entity_recognition import ner_preprocessing
from lrf.utilities import constants, utility


############## Matrix initialization
def initialize_matrix(dim, m,n,initialization_typ='random_uniform'):
    upper_lim = np.sqrt(3 / dim)
    lower_lim = upper_lim * (-1)

    if initialization_typ == 'random_uniform':
        initialized_matrix = np.random.uniform(low=lower_lim, high=upper_lim, size=(m, n))

    return initialized_matrix


################# Model
def tweet_ner_model(tweets_info):

    ### Get The orthographic Sentences
    cmplt_word_list = tweets_info['words']
    cmplt_ortho_word_list = tweets_info['orthographic_words']
    cmplt_BIOU_result = tweets_info['']


    flattened_wrdlst = [val for sublist in cmplt_word_list for val in sublist]
    flattened_ortho_wrdlst = [val for sublist in cmplt_ortho_word_list for val in sublist]

    unique_wrds = set(flattened_wrdlst)
    unique_ortho_wrds = list(set(flattened_ortho_wrdlst))

    glove_dict = utility.getGloveVec(unique_wrds)

    char_dict = lrf_config.get_char_dict()
    ortho_char_dict = lrf_config.get_orthohraphic_char_dict()


    ############################# Initializations of Embedding Matrices:

    #### Initialization of Actual Word Character Embedding : DIM : Dim_of_chars x Dim_needed_for_char_embedding : 94 x 30
    ## Random Uniform Initialization

    char_embed_matrix = initialize_matrix(dim = constants.CHAR_EMBED_DIM, m = constants.CHAR_EMBED_DIM, n=constants.CHAR_ONEHOT_DIM, initialization_typ ='random_uniform')

    #### Initialization of Orthographic Word Character Embedding : DIM : Dim_of_ortho_chars x Dim_needed_for_ortho_char_embedding : 4 x 30
    ## Random Uniform Initialization

    char_o_embed_matrix = initialize_matrix(dim=constants.CHAR_O_EMBED_DIM, m=constants.CHAR_O_EMBED_DIM,
                                            n=constants.CHAR_O_ONEHOT_DIM, initialization_typ ='random_uniform')

    #### Initialization of Orthographic Word Embedding : DIM : Dim_of_unique_Ortho_words x Dim_of_glove_vec : n x 200
    ## Random Uniform Initialization

    word_o_embed_matrix = initialize_matrix(dim=constants.GLOVE_DIM, m=constants.GLOVE_DIM,
                                            n=len(unique_ortho_wrds), initialization_typ='random_uniform')


    ############################ Actual Model for Processing

    comprehensive_input = []

    for ind_tweet,tweet in enumerate(cmplt_word_list):
        ortho_tweet = cmplt_ortho_word_list[ind_tweet]

        for ind_word,word in enumerate(tweet):
            ortho_word = ortho_tweet[ind_word]

            #########################################################
            ## Part 1: Finding Char Embedding of any word:
            char_labels = [char_dict[c] for c in list(word)]
            char_onehot = keras.utils.to_categorical(char_labels, num_classes=constants.CHAR_ONEHOT_DIM)

            char_embed_inp = np.matmul(char_embed_matrix,np.transpose(char_onehot))

            out_1 = Conv2D(filters=constants.NUM_OF_FILTERS, kernel_size=(
            constants.CHAR_EMBED_DIM, constants.WINDOW_SIZE), padding='same', activation=constants.LAYER_1_ACTIV, kernel_initializer=RandomUniform, bias_initializer=RandomUniform)(char_embed_inp)

            #########################################################
            ## Part 2: Finding Word Embedding of word: Glove
            high_dim = np.sqrt(3 / constants.GLOVE_DIM)
            low_dim = (-1) * high_dim

            out_2 = np.transpose(glove_dict.get(word)) if word in glove_dict else np.random.uniform(low=low_dim,
                                                                                                    high=high_dim,
                                                                                                    size=(
                                                                                                        constants.GLOVE_DIM,
                                                                                                        1))
            #########################################################
            ## Part 3: Finding Char Embedding of orthographic word

            ortho_char_labels = [ortho_char_dict[c] for c in list(ortho_word)]
            ortho_char_onehot = keras.utils.to_categorical(ortho_char_labels, num_classes=constants.CHAR_O_ONEHOT_DIM)

            ortho_char_embed_inp = np.matmul(char_o_embed_matrix, np.transpose(ortho_char_onehot))

            out_3 = Conv2D(filters=constants.NUM_OF_FILTERS,
                           kernel_size=(constants.CHAR_O_EMBED_DIM, constants.WINDOW_SIZE), padding='same',
                           activation=constants.LAYER_2_ACTIV, kernel_initializer=RandomUniform,
                           bias_initializer=RandomUniform)(ortho_char_embed_inp)

            #########################################################
            ## Part 4: Finding Word Embedding of orthographic word

            word_onehot = keras.utils.to_categorical(unique_ortho_wrds.index(ortho_word))
            ortho_word_inp = np.matmul(np.transpose(word_o_embed_matrix),word_onehot)
            out_4 =  Conv2D(filters=constants.NUM_OF_FILTERS, kernel_size=(
            constants.WORD_O_EMBED_DIM, constants.WINDOW_SIZE), padding='same', activation=constants.LAYER_3_ACTIV, kernel_initializer=RandomUniform, bias_initializer=RandomUniform)(ortho_word_inp)

            comprehensive_input = tf.keras.backend.stack((out_1,out_2,out_3,out_4),axis=0)

            # comprehensive_input.append(np.concatenate((out_1,out_2,out_3,out_4)))


        LSTM_NUM_NODES = len(comprehensive_input)

        lstm_out = keras.layers.Bidirectional(LSTM(units=LSTM_NUM_NODES, return_sequences=True, activation='hard_sigmoid', use_bias=True, kernel_initializer=RandomUniform, dropout=0.0))(comprehensive_input)

        comprehensive_model = crf.CRF(constants.NUM_OF_TAGS)
        out = comprehensive_model(lstm_out)


def plot_results(hist):
    plt.style.use("ggplot")
    plt.figure(figsize=(12,12))
    plt.plot(hist["acc"])
    plt.plot(hist["val_acc"])
    plt.show()

def assign_label_seq():
    train_data,validation_data,test_data = ner_preprocessing.get_ner_data()
    model = Model(train_data,train_data['tags'])

    model.compile(optimizer='rmsprop', loss=crf.loss_function, metrics=[crf.accuracy])

    model.summary()

    ## Fitting the model
    history = model.fit(train_data, train_data['tags'],batch_size=32, epochs=5,validation_data=(validation_data,validation_data['tags']))

    hist = pd.DataFrame(history.history)

    plot_results(hist)




