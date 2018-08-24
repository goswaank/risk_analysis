import keras
import numpy as np
from lrf.configs import lrf_config
from lrf.utilities import constants, utility
import os
import re

## Global Initializations
locations = lrf_config.get_locations()
ner_ref_location = '/home/onespace/ankita/risk_analysis/reference_data/ner_data'



## Converting word or sentences to Orthographic Sentences
def get_orthographic_sentences(word):
    word = re.sub(r'([A-Z])', 'C', word)
    word = re.sub(r'([a-z])', 'c', word)
    word = re.sub(r'([1-9])', 'n', word)
    word = re.sub(r"([!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~])", 'p', word)

    return word


## Getting the data
def get_labeled_data(file_name):
    data_path = os.path.join(ner_ref_location,file_name)

    with open(data_path,'r') as f:
        data = f.readlines()

    ## Per Tweet Lists
    tweet_word_tmp = []
    tweet_tag_tmp = []
    orthographic_words = []

    ## Global Lists
    tweets = {}

    tweets['sentence'] = []
    tweets['words'] = []
    tweets['tags'] = []
    tweets['orthographic_words'] = []
    for item in data:

        if item.isspace() == False:
            word,tag = item.split('	')
            tweet_word_tmp.append(word)
            tweet_tag_tmp.append(tag)

            orthographic_words.append(get_orthographic_sentences(word))
        else:

            ## Storing in global tweet dictionary
            tweets['sentence'].append(' '.join(tweet_word_tmp))
            tweets['words'].append(tweet_word_tmp)
            tweets['tags'].append(tweet_tag_tmp)
            tweets['orthographic_words'].append(orthographic_words)

            ## Emptying the list for new tweet
            tweet_word_tmp = []
            tweet_tag_tmp = []

    return tweets

def vectorize_unique_words(data):
    all_wordlists = data['words']
    flattened_wordlist = [val for sublist in all_wordlists for val in sublist]

    unique_words = set(flattened_wordlist)

    upper_lim = np.sqrt(3 / constants.CHAR_EMBED_DIM)
    lower_lim = -1*upper_lim

    char_dict = utility.get_char_dict()

    all_obs = []
    all_targets = []

    for word in unique_words:
        word_len = len(word)
        initialized_word_matrix = np.random.uniform(low=lower_lim, high=upper_lim, size=(
        constants.CHAR_EMBED_DIM, word_len))

        word_matrix = []
        for c in word:
            word_matrix.append(char_dict[c])

        all_targets.append(keras.utils.to_categorical(word_matrix, num_classes=constants.CHAR_EMBED_DIM))
        all_obs.append(initialized_word_matrix)

    return all_obs,all_targets

# def get_char_embeddings(train_data,train_target,validation_data,validation_target):
#
#     ## Defining the Model
#     model = Sequential()
#     model.add(Convolution2D(filters=constants.NUM_OF_FILTERS,kernel_size=char_embed_dim,strides=3,data_format='channels_last',padding='same',kernel_initializer=RandomUniform, bias_initializer=RandomUniform,activation='sigmoid'))
#     model.add(MaxPooling2D(pool_size=(2,2)))
#     model.add(Flatten())
#     model.add(Dense(100))
#     model.add(Dropout(0.5))
#     model.add(Dense(94))
#     model.add(Activation('sigmoid'))
#     model.compile(loss='categorical_crossentropy',optimizer='adamhd',metrics=['accuracy'])
#
#
#     ## Fitting the Model
#     model.fit(x = train_data,y = train_target, validation_data=(validation_data, validation_target))


def get_ner_data():
    train_data = get_labeled_data('ner_train')
    validation_data = get_labeled_data('ner_dev')
    test_data = get_labeled_data('ner_test')

    return train_data,validation_data,test_data

## Main Function
if __name__=='__main__':

    train_data,validation_data, test_data = get_ner_data()

    # train_data, train_target = vectorize_unique_words(train_data)
    # validation_data, validation_target = vectorize_unique_words(validation_data)
    # char_embeddings, train_acc, test_acc = get_char_embeddings(train_data,train_target,validation_data,validation_target)
