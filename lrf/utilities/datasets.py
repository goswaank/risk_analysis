import os
import pandas as pd
import re
import numpy as np
from multiprocessing import Pool
from nltk.corpus import stopwords
import itertools
from lrf.configs import lrf_config as lc
from collections import defaultdict

################ Get MPQA Data
def get_mpqa_data():

    locations = lc.get_locations()

    ref_data_path = locations['REF_DATA_PATH']

    mpqa_file_path = os.path.join(ref_data_path,'subjectivity_clues_hltemnlp05/subjclueslen1-HLTEMNLP05.tff')

    sentiment_map = lc.get_sentiment_map()

    with open(mpqa_file_path,'r') as f:

        data = f.readlines()

    mpqa_dict = {}

    for line in data:

        elem_bag = line.strip('\n').split(' ')

        for elem in elem_bag:

            item = elem.split('=')

            if item[0] == 'word1':

                word = item[1]

            elif item[0] == 'priorpolarity':
                binary_output = [0]*len(sentiment_map)
                if sentiment_map.get(item[1]) is not None:
                    binary_output[sentiment_map.get(item[1])] = 1
                    mpqa_dict[word] = binary_output

    return mpqa_dict

############### GET ABBREVIATIONS DATA
def get_twitter_abbreviations_data():

    locations = lc.get_locations()

    ref_data_path = locations['REF_DATA_PATH']

    abbr_file_path = os.path.join(ref_data_path, 'twitter_slang/twitter_slang_data.txt')

    data = pd.read_csv(abbr_file_path,sep='|',names=['abbr','meaning'])
    data_len = len(data['abbr'])

    abbr_list = data['abbr'].values
    meaning_list = data['meaning'].values
    slang_dict = {}

    for ind in range(data_len):
        slang_dict[abbr_list[ind]] = meaning_list[ind]

    return slang_dict


################ GET SENTIMENT DATA
def get_sentiment_data():

    locations = lc.get_locations()

    ref_data_path = locations['REF_DATA_PATH']

    sentiment_file_path = os.path.join(ref_data_path, 'sentiment_data/sentiment_lexicon.txt')

    sentiment_dict = defaultdict(dict)

    ## Reading and storing Emoticon words
    with open(sentiment_file_path, 'r') as f:

        emoticon_data = f.readlines()

    ## Creating Sentiment Dictionary
    for line in emoticon_data:

        line_split = line.split('\t');

        emoticon = line_split[0]

        mean_variance = [float(line_split[1]), float(line_split[2])]

        sentiment_dict[emoticon] = mean_variance

    return sentiment_dict

################# GET NEWS DATA
def get_news_data(folder_name,file_name):

    locations = lc.get_locations()

    NEWS_DATA_PATH = os.path.join(locations['REF_DATA_PATH'], folder_name+'/'+file_name)

    news_data = pd.read_csv(NEWS_DATA_PATH)

    news_data = news_data.drop(["Unnamed: 0"], axis=1).set_index('Unnamed: 0.1')

    news_data['category'] = prepare_data(news_data['category'],'y_data',list_type=int)

    news_data['text'] = prepare_data(news_data['text'],'x_data')

    news_data = news_data[news_data.text != 'None']

    news_data.to_dict('index')

    return news_data

#################################################
def get_tweet_data(file_type,file_name):

    locations = lc.get_locations()
    if file_type=='json':

        TWEETS_DATA_PATH = os.path.join(locations['INTERMED_DATA_PATH'], file_name)

        tweet_data = pd.read_json(TWEETS_DATA_PATH, orient='records',convert_axes=False)

        tweet_data['tweet_cmplt'] = prepare_data(tweet_data['tweet_cmplt'],'x_data')

        return  tweet_data

    elif file_type == 'txt':

        TWEETS_DATA_PATH = os.path.join(locations['INTERMED_DATA_PATH'], file_name)

        if file_name == 'tweets_classified.txt':

            tweet_data = pd.read_csv(TWEETS_DATA_PATH,sep='|',names=['tweet_id','class_pos','class_both','tweet_word_list','tweet_cmplt']).drop_duplicates().set_index('tweet_id')

        elif file_name == 'tweet_truth.txt':

            tweet_data = pd.read_csv(TWEETS_DATA_PATH, sep='|').drop_duplicates(subset='tweet_id').set_index('tweet_id')

            tweet_data['class_annotated'] = prepare_data(tweet_data['class_annotated'], 'y_data',list_type=str)

        tweet_data['tweet_cmplt'] = prepare_data(tweet_data['tweet_cmplt'], 'x_data')

        tweet_data['tweet_word_list'] = prepare_data(tweet_data['tweet_word_list'], 'word_bag')

        tweet_data['class_pos'] = prepare_data(tweet_data['class_pos'], 'y_data')

        tweet_data['class_both'] = prepare_data(tweet_data['class_both'], 'y_data')

        tweet_data.to_dict('index')

        return tweet_data


################## Prepare Data
def prepare_data(data,data_purpose,list_type=str):

    if data_purpose == 'x_data':

        X_data = splitAndProcess(data, text_prepare, 8, 8)

        new_X_data = []

        for elem in X_data:

            new_X_data.append(str(elem))


        return new_X_data

    elif data_purpose == 'word_bag':

        X_data = splitAndProcess(data,list_prepare_star, 8,8,list_type)

        return X_data

    elif data_purpose == 'y_data':

        Y_data = splitAndProcess(data,list_prepare_star,8,8,list_type)

        Y_data_list = []

        for elem in Y_data:

            if (type(elem) != list):

                elem = list(elem)

                Y_data_list.append(elem)
            else:
                Y_data_list.append(elem)

        return Y_data_list


##############################################
def text_prepare(texts):

    processed_texts = []

    for i, text in enumerate(texts):

        try:

            _text = text.lower()

            _text = re.compile('[/(){}\[\]\|@,;]').sub(r' ', _text)

            _text = re.compile('[^0-9a-z #+_]').sub(r'', _text)

            _text = ' '.join([word for word in _text.split() if word not in set(stopwords.words('english'))])

            processed_texts.append(_text)

        except Exception as e:

            processed_texts.append('None')


    return processed_texts

##################################################
def list_prepare_star(zipped_obj):

    return list_prepare(zipped_obj[0],zipped_obj[1])

##################################################
def list_prepare(lists,lst_type):

    processed_lists = []

    if lst_type == str:

        for one_list in lists:

            try:

                one_list = eval(one_list)

                processed_lists.append(one_list)

            except Exception as e:

                print(one_list)

                print(e)

                processed_lists.append('None')

    elif lst_type == int:

        class_map = lc.get_class_map()
        inv_class_map = {v:k for k,v in class_map.items()}

        for one_list in lists:

            try:

                one_list = eval(one_list)
                new_list = [inv_class_map.get(int(item)) for item in one_list]

                processed_lists.append(new_list)

            except Exception as e:

                print(e)

                processed_lists.append('None')

    return processed_lists

################################################
def splitAndProcess(X, func, num_partitions, num_cores=4, list_type=None):

    df_split = np.array_split(np.array(X), num_partitions)

    pool = Pool(num_cores)

    if list_type is None:

        X = np.concatenate(pool.map(func, df_split))

    else:

        X = np.concatenate(pool.map(func, zip(df_split, itertools.repeat(list_type))))

    pool.close()

    pool.join()

    return X

##############################################
if __name__=='__main__':
    # get_twitter_abbreviations_data()
    # data = get_news_data(folder_name='keyword_data',file_name='annotator_data_dump_with_text')
    res = get_sentiment_data()



    print(res)
    # get_tweet_data(file_type='json',file_name='intermed_dict.json')
    # get_tweet_data(file_type='txt', file_name='tweet_truth.txt')
