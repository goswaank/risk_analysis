from lrf import utility, lrf_config
from tqdm import tqdm
import spacy

### Actual Imports from here
import pandas as pd
import numpy as np
import os
from lrf import tweet_former, constants
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report
from keras.callbacks import ModelCheckpoint

##########################################################################
def read_data(data_location, file_name):

    ## Data paths and locations
    data_abs_path = os.path.join(data_location,file_name)

    ## Delimiter
    delim = '	'

    ## Reading the Data
    with open(data_abs_path,'r') as fp:
       raw_data = fp.readlines()

    tweet_id = 0

    tweets_data = []

    for row in raw_data:
        if row.isspace() == False:
            word, ner_tag = row.strip('\n').split(delim)
            word_data = [tweet_id,word,None,ner_tag]
            tweets_data.append(word_data)
        else:
            tweet_id += 1

    tweet_dataframe = pd.DataFrame(data=tweets_data,columns=['tweet_id', 'word', 'pos_tag', 'ner_tag'])

    return tweet_dataframe

################################################################
def prepare_tokens(elem_list,extras):

    ind_step = len(extras)

    token_dict = {elem: i+ind_step for i,elem in enumerate(elem_list)}

    for ind,item in enumerate(extras):
        token_dict[item] = ind

    inv_token_dict = {i:w for w,i in token_dict.items()}

    return token_dict,inv_token_dict

################################################################
def get_char_sequences(tweets,char2ind):
    '''
    :param tweets:
    :param char2ind:
    :return: Character sequence for each word in each sentence

    Ex: Sentence : "I am an owl"
    then for each word: ->
    word_seq.append(char2ind['I']) --> tweet_seq.append([2])
    word_seq.append(char2ind['a']) -> word_seq.append(char2ind['m']) --> tweet_seq.append([3,67])
    word_seq.append(char2ind['a']) -> word_seq.append(char2ind['n']) --> tweet_seq.append([3,45])
    word_seq.append(char2ind['o']) -> word_seq.append(char2ind['w']) -> word_seq.append(char2ind['l']) --> tweet_seq.append([34,56,53])

    => X_seq.append([[2],[3,67],[3,45],[34,45,53]])  ===> return X_char : once done for all the tweets
    '''
    X_char = []

    for tweet in tweets:
        tweet_seq = []
        for i in range(constants.MAX_SENT_LEN):
            word_seq = []
            for j in range(constants.MAX_WORD_LEN):
                try:
                    word_seq.append(char2ind.get(tweet[i][0][j]))
                except:
                    word_seq.append(char2ind.get("PAD"))
            tweet_seq.append(word_seq)
        X_char.append(tweet_seq)

    return X_char

################################################################
def plot_graph(plot_typ="ggplot", fig_size = (12,12), fields = ["acc","val_acc","loss","val_loss"], lrate=0.3, epochs=10,test_type='validation',model_type='lstm_bilstm',activation='sigmoid',optimizer='adam'):

    import matplotlib.pyplot as plt
    plt.style.use(plot_typ)
    plt.figure(figsize=fig_size)

    legend_list = []

    for field in fields:
        plt.plot(hist[field])
        legend_list.append(field)

    plt.title('model accuracy and loss')
    plt.ylabel('accuracy/loss')
    plt.xlabel('epoch')
    plt.legend(legend_list, loc='upper left')
    plt.show()
    print(os.path.join(plot_location, model_type+ activation+'_LR_'+str(lrate)+'_EP_'+str(epochs)+'_'+test_type + '.png'))
    outpath = os.path.join(plot_location, model_type+ activation+'_LR_'+str(lrate)+'_EP_'+str(epochs)+'_'+test_type + '.png')
    plt.savefig(outpath)


################################################################
def get_char_embedding_model():

    ## Imports
    from keras.models import Model, Input
    from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Conv1D
    from keras.layers import Bidirectional, concatenate, SpatialDropout1D, GlobalMaxPool1D

    ## Apparently the trick here is to wrap the parts that should be applied to characters in a TimeDistributed so that characters in a layer apply the same layers to every character sequence

    ## Returns a Tensor
    word_in = Input(shape=(constants.MAX_SENT_LEN,))

    ## To find word embedding
    emb_word = Embedding(input_dim=n_words+2, output_dim=20, input_length=constants.MAX_SENT_LEN, mask_zero=True)(word_in)

    ## To find character embedding for characters of that word
    char_in = Input(shape=(constants.MAX_SENT_LEN,constants.MAX_WORD_LEN,))
    emb_char = TimeDistributed(Embedding(input_dim=n_chars+2, output_dim=10, input_length=constants.MAX_WORD_LEN, mask_zero=True))(char_in)

    ## Character CNN to get the word encoding by characters
    # char_encoding = TimeDistributed(Conv1D())
    char_encoding = TimeDistributed(LSTM(units=20, return_sequences=False, activation=constants.ACTIVATION,recurrent_dropout=0.5))(emb_char)

    ## main LSTM
    x = concatenate([emb_word,char_encoding])
    x = SpatialDropout1D(0.3)(x)

    main_lstm = Bidirectional(LSTM(units=50, return_sequences=True, recurrent_dropout=0.6,activation=constants.ACTIVATION))(x)

    out = TimeDistributed(Dense(n_tags+1, activation="softmax"))(main_lstm)

    model = Model([word_in,char_in], out)

    return model

################################################################
def get_validation_data(test_type):
    if test_type == 'validation':
        data = ([X_word_validation, np.array(X_char_validation).reshape((len(X_char_validation), max_sent_len, max_word_len))],
                               np.array(y_validation).reshape(len(y_validation),max_sent_len, 1))
    elif test_type == 'test':
        data = ([X_word_test, np.array(X_char_test).reshape((len(X_char_test), max_sent_len, max_word_len))],
                               np.array(y_test).reshape(len(y_test),max_sent_len, 1))
    return data

################################################################
def get_optimizer(optimizer,learning_rate=0.001):
    from keras import optimizers
    if optimizer == 'adam':
        return optimizers.Adam(lr=learning_rate)
    elif optimizer == 'adamax':
        return optimizers.Adamax(lr=learning_rate)
    elif optimizer == 'adagrad':
        return optimizers.Adagrad(lr=learning_rate)
    elif optimizer == 'rmsprop':
        return optimizers.RMSprop(lr=learning_rate)
    elif optimizer == 'adadelta':
        return optimizers.Adadelta(lr=learning_rate)
    elif optimizer == 'sgd':
        return optimizers.SGD(lr=learning_rate)


################################################################
if __name__=='__main__':

    ## Data Paths and Locations
    plot_location = '/home/onespace/home/onespace/ankita/risk_analysis/plots/lrf_ner'
    data_location = '/home/onespace/ankita/risk_analysis/reference_data/ner_data'
    train_file_name = "ner_train"
    validation_file_name = "ner_dev"
    test_file_name = "ner_test"

    ## Reading the Data
    train_dataframe = read_data(data_location,train_file_name)
    validation_dataframe = read_data(data_location, validation_file_name)
    test_dataframe = read_data(data_location, test_file_name)

    ## Collecting Uniqe Words
    words = list(set(train_dataframe['word'].values).union(set(validation_dataframe['word'].values)))
    n_words = len(words)

    ## Collecting Unique Tags
    tags = list(set(train_dataframe['ner_tag'].values).union(set(validation_dataframe['ner_tag'].values)))
    n_tags = len(tags)

    ## Getting Full Tweets
    train_tf = tweet_former.TweetFormer(train_dataframe)
    validation_tf = tweet_former.TweetFormer(validation_dataframe)
    test_tf = tweet_former.TweetFormer(test_dataframe)

    ## Getting the Tweets
    train_tweets = train_tf.tweets
    validation_tweets = validation_tf.tweets
    test_tweets = test_tf.tweets

    ## Preparing the tokens
    ## Here we use extra padding because we have to use mask_zero parameter of embedding layer to ignore with value zero
    extras = ['PAD','UNK']
    word2ind, ind2word = prepare_tokens(words,extras)
    extras = ['PAD']
    tag2ind, ind2tag = prepare_tokens(tags,extras)

    ## Max Len of a sentence and a word - this is to ensure that we don't run into disappearing gradient issues in Neural Networks ## Random Choice
    max_sent_len = constants.MAX_SENT_LEN
    max_word_len = constants.MAX_WORD_LEN

    ## To perform this padding we use keras pad_sequences
    X_word_train = [[word2ind[w[0]] for w in tweet] for tweet in train_tweets]
    X_word_validation = [[word2ind[w[0]] for w in tweet] for tweet in validation_tweets]
    X_word_test = [[word2ind[w[0]] if w[0] in word2ind else word2ind["UNK"] for w in tweet] for tweet in test_tweets]

    ## Padding of the sentence
    '''
    maxlen = maximum length of all the sequencs encountered
    sequences = input that is list of positive integers to be padded
    value = the value with which we need to pad
    padding = whether to do the padding before or after the sequence encountered
    truncating = if the sequence is larger than the maxlen then truncation needs to happen from beginning or end of the tweet
    '''
    X_word_train = pad_sequences(maxlen=max_sent_len, sequences=X_word_train, value=word2ind["PAD"],padding='post',truncating='post')
    X_word_validation = pad_sequences(maxlen=max_sent_len, sequences=X_word_validation, value=word2ind["PAD"], padding='post',
                                 truncating='post')
    X_word_test = pad_sequences(maxlen=max_sent_len, sequences=X_word_test, value=word2ind["PAD"],
                                      padding='post',
                                      truncating='post')

    ## Padding the words:
    chars = set([c for w in words for c in w])
    n_chars = len(chars)

    ## Prepare char index dicts
    extras = ['PAD','UNK']
    char2ind, ind2char = prepare_tokens(chars,extras)


    ## Getting the list of characters for each word in each sentence:

    X_char_train = get_char_sequences(train_tweets,char2ind)
    X_char_validation = get_char_sequences(validation_tweets,char2ind)
    X_char_test = get_char_sequences(test_tweets, char2ind)

    ######## Making tag sequences
    y_train = [[tag2ind[w[2]] for w in word] for word in train_tweets]
    y_validation = [[tag2ind[w[2]] for w in word] for word in validation_tweets]
    y_test = [[tag2ind[w[2]] for w in word] for word in test_tweets]

    ####### Padding the tag sequences
    y_train = pad_sequences(maxlen=max_sent_len, sequences=y_train, value=tag2ind["PAD"], padding='post', truncating='post')
    y_validation = pad_sequences(maxlen=max_sent_len, sequences=y_validation, value=tag2ind["PAD"], padding='post',
                            truncating='post')
    y_test = pad_sequences(maxlen=max_sent_len, sequences=y_test, value=tag2ind["PAD"], padding='post',
                            truncating='post')

    ######## Character Embedding Model
    char_embed_model = get_char_embedding_model()

    ## Compilation Parameters

    from keras import optimizers
    # char_embed_model.compile(optimizer=get_optimizer(constants.OPTIM,constants.LR), loss="sparse_categorical_crossentropy", metrics=["acc"])
    char_embed_model.compile(optimizer=optimizers.Adam(lr=0.001),
                             loss="sparse_categorical_crossentropy", metrics=["acc"])
    ## Print Model Summary
    char_embed_model.summary()


    ## Saving the Model using checkpoints - after every 25 epochs
    # model_file_path = '/home/onespace/ankita/risk_analysis/reference_data/ner_model/ner_best_model.hdf5'
    # checkpoint = ModelCheckpoint(model_file_path, monitor="val_loss",verbose=1,save_best_only=False, mode='min',period=constants.CALLBACK_INTERVAL)
    # callbacks_list = [checkpoint]

    ## Model Hyperparameters
    batch_size=constants.BATCH_SIZE
    epochs=constants.EPOCHS
    test_type=constants.TEST_TYPE

    ## Fitting the model
    # history = char_embed_model.fit([X_word_train, np.array(X_char_train).reshape((len(X_char_train), max_sent_len, max_word_len))],
    #                            np.array(y_train).reshape(len(y_train),max_sent_len, 1),callbacks=callbacks_list,
    #                            batch_size=batch_size, epochs = epochs, validation_data=get_validation_data(test_type), verbose = 1)

    history = char_embed_model.fit(
        [X_word_train, np.array(X_char_train).reshape((len(X_char_train), max_sent_len, max_word_len))],
        np.array(y_train).reshape(len(y_train), max_sent_len, 1),
        batch_size=batch_size, epochs=10, validation_split = 0.1, verbose=1)

    ## Save Model

    char_embed_model.save('/home/onespace/ankita/risk_analysis/reference_data/ner_model/char_embed_model')

    ## Create a dataframe out of the history
    hist = pd.DataFrame(history.history)

    ## plot a graph for the data
    plot_graph(plot_typ="ggplot", fig_size = (12, 12), fields = constants.plot_fields, lrate = constants.LR, epochs = epochs, test_type = 'validation', model_type = 'lstm_bilstm', activation = constants.ACTIVATION,optimizer=constants.OPTIM)
    plot_graph(plot_typ="ggplot", fig_size=(12, 12), fields=['loss','val_loss'], lrate=constants.LR, epochs=epochs,
               test_type='validation', model_type='lstm_bilstm_loss_', activation=constants.ACTIVATION,
               optimizer=constants.OPTIM)
    plot_graph(plot_typ="ggplot", fig_size=(12, 12), fields=['acc','val_acc'], lrate=constants.LR, epochs=epochs,
               test_type='validation', model_type='lstm_bilstm_acc_', activation=constants.ACTIVATION,
               optimizer=constants.OPTIM)
    ## Extra Code to look into some of the predictions
    ## Predicting the Validation Data
    y_pred = char_embed_model.predict([X_word_validation, np.array(X_char_validation).reshape((len(X_char_validation),max_sent_len, max_word_len))])

    # y_pred = y_pred.reshape(y_pred.shape[0] * y_pred.shape[1], y_pred.shape[2])
    # y_validation = y_validation.reshape(y_validation.shape[0] * y_validation.shape[1])
    # from keras import backend as K
    # import keras
    # n_values = np.max(y_validation) + 1
    # y_validation = np.eye(n_values)[y_validation]
    # x = K.eval(keras.metrics.categorical_accuracy(K.variable(y_validation), K.variable(y_pred)))
    # print(np.sum(x))
    # print(len(x))
    # # exit(0)

    for i,elem in enumerate(y_pred[0]):
        print(i,' ---- ',elem,' +++++ ',np.argmax(elem))
    # classification_report(y_validation,y_pred)
    exit(0)
    p = np.argmax(y_pred,axis=-1)

    print(p)
    print(y_validation)
    for i, elem in enumerate(y_validation):
        print(elem, ' : ', p[i])

    classification_report(y_validation,p)
    ## Looking into those predictions
    i = 1
    p = np.argmax(y_pred[i], axis = -1)
    print('{:15}||{:5}||{}'.format("Word","True","Pred"))
    print(30 * '=')

    for w,t,pred in zip(X_word_validation[i], y_validation[i], p):
        if w != 0:
            print("{:15}: {:5} {}".format(ind2word[w], ind2tag[t], ind2tag[pred]))

