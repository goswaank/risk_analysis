from lrf import lrf_config,datasets,utility,classifier
import os
import json
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from lrf import replacers,removers
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from sklearn.preprocessing import normalize
import numpy as np
from sklearn.feature_selection import chi2,SelectKBest
from scipy import sparse
from sklearn.model_selection import KFold
############################ GLOBAL INITIALIZATIONS

def pre_process_lst(str):
    str = str.lower()
    str = removers.rm_url(str)
    str = removers.rm_at_user(str)
    str = removers.rm_repeat_chars(str)
    str = removers.rm_hashtag_symbol(str)
    str = removers.rm_time(str)

    return str

############################# PREPROCESS FUNCTION

def pre_process(str):

    tt = TweetTokenizer()

    str = removers.rm_hashtag_symbol(str)
    str = removers.rm_at_user(str)
    str = removers.rm_url(str)
    str = removers.rm_time(str)

    word_bag = tt.tokenize(str)

    ## Extra Features
    sentiment_score = []
    intensifier_presence = 0
    diminisher_presence = 0
    lexicon_features = [0]*len(lrf_config.get_sentiment_map())

    expanded_wrd_list = []

    for word in word_bag:
        if word.lower() in sentiment_data:
            mean,variance = sentiment_data.get(word.lower())
            senti_score = mean/(0.01+variance)
            sentiment_score.append(senti_score)

        correct_word, is_intensifier, is_diminisher = replacers.RepeatReplacer().intensifier_and_spell_chck(word)

        if is_intensifier == True:
            intensifier_presence = 1
        if is_diminisher == True:
            diminisher_presence = 1

        if correct_word in abbreviation_data:
            correct_word = abbreviation_data.get(correct_word)

        expanded_wrd_list.append(correct_word)

    if sentiment_score != []:
        sentiment_mean = np.mean(sentiment_score)
    else:
        sentiment_mean = 0


    microblogging_features = [intensifier_presence,diminisher_presence,sentiment_mean]

    sentence = replacers.AntonymReplacer().replace_negations(expanded_wrd_list)

    final_word_list = []

    for word_case in sentence:
        word = word_case.lower()
        if word not in en_stopwords:
            if word in mpqa_dataset:
                mpqa_sentiment_score = mpqa_dataset.get(word)
                lexicon_features[0] = lexicon_features[0] + mpqa_sentiment_score[0]
                lexicon_features[1] = lexicon_features[1] + mpqa_sentiment_score[1]
                lexicon_features[2] = lexicon_features[2] + mpqa_sentiment_score[2]

            final_word_list.append(word)

    summed_val =sum(lexicon_features)

    if summed_val != 0:
        lexicon_features = [elem/summed_val for elem in lexicon_features]

    return ' '.join(final_word_list), microblogging_features, lexicon_features

########################## MAIN FUNCTION
def main():


    locations = lrf_config.get_locations()

    ref_data_dir = locations['REF_DATA_PATH']

    x_filename = 'sentiment_data/tweets.txt'
    y_filename = 'sentiment_data/labels.txt'

    ##load and process samples
    print('start loading and process samples...')

    tweets = []
    microblog_features = []
    lexicon_features = []
    tweets_lst = []

    with open(os.path.join(ref_data_dir,x_filename)) as f:

        for i, line in enumerate(f):

            tweet_obj = json.loads(line.strip(), encoding='utf-8')

            # Twitter Text contents
            content = tweet_obj['text'].replace("\n", " ")

            tweets_lst.append(pre_process_lst(content))


            postprocessed_tweet,microblogging_features, mpqa_sentiment_score = pre_process(content)

            tweets.append(postprocessed_tweet)

            microblog_features.append(microblogging_features)

            lexicon_features.append(mpqa_sentiment_score)


    lexicon_features = np.asarray(lexicon_features)
    microblog_features = np.asarray(microblog_features)

    tf_idf_vectorizer = utility.get_tf_idf_vectorizer(tweets_lst, ngram_range=2)

    transformed_data_rahul = tf_idf_vectorizer.fit_transform(tweets_lst)
    #
    # tf_idf_vectorizer = utility.get_tf_idf_vectorizer(tweets,ngram_range=2)
    #
    # transformed_data_mine = tf_idf_vectorizer.fit_transform(tweets)

    with open(os.path.join(ref_data_dir,y_filename)) as f:
        y_data = f.readlines()

    y_data = [y.strip('\n') for y in y_data]
    y_data = np.asarray(y_data)
    num_of_features = 50
    accuracy_in_each_turn = []
    while num_of_features <= 3000:
        X_new = SelectKBest(chi2, k=num_of_features).fit_transform(transformed_data_rahul, y_data)


        extended_features_1 = np.append(X_new.toarray(),lexicon_features,axis=1)
        extended_features_2 = np.append(extended_features_1,microblog_features,axis=1)

        sentiment_map = lrf_config.get_sentiment_map()
        inv_sentiment_map = {str(v): k for k, v in sentiment_map.items()}

        X_data = X_new.toarray()


        kf = KFold(n_splits=5)
        kf.get_n_splits(X_data)
        train_list = []
        test_list = []

        for train_index, test_index in kf.split(X_data):
            X_train = X_data[train_index]
            Y_train = y_data[train_index]
            X_test = X_data[test_index]
            Y_test = y_data[test_index]

            Y_pred, train_acc, test_acc = classifier.classify('svc',X_train=X_train,Y_train=Y_train,X_test=X_test,Y_test=Y_test,class_map=inv_sentiment_map,is_X_text=False)

            # print('_______________________________________________________')
            # print(train_acc)
            # print(test_acc)
            train_list.append(train_acc)
            test_list.append(test_acc)

        # print('Train_Acc : ',np.mean(train_acc))
        # print('Test_Acc : ', np.mean(test_acc))
        accuracy_in_each_turn.append([np.mean(train_acc),np.mean(test_acc)])

    for elem in accuracy_in_each_turn:
        print(elem)


if __name__=='__main__':
    mpqa_dataset = datasets.get_mpqa_data()
    sentiment_data = datasets.get_sentiment_data()
    abbreviation_data = datasets.get_twitter_abbreviations_data()
    en_stopwords = set(stopwords.words('english'))
    main()