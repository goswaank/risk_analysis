from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from stop_words import get_stop_words
import pymongo
from w3lib.html import remove_tags
from collections import defaultdict
import time
from tweet import Tweet
from user import User
import numpy as np
import os
import json
from sklearn.metrics.pairwise import cosine_similarity
from lrf import unsupervised
from lrf import lrf_config
##################### Get Word To Category Mapping - Modular


#################### Get Word to Category Mapping
def getWordToCategMap(risk_keywords,glove_risk_dict):
    pos_risk_dict = {}
    neg_risk_dict = {}

    posKeyGloveLst = []
    negKeyGloveLst = []
    inv_pos_key_index = {}
    inv_neg_key_index = {}

    pos_key_index = 0
    neg_key_index = 0

    for category in risk_keywords:
        ## Initializing the key index to store the inverse index and word dicitonary - Pos Words
        for word in risk_keywords[category]['pos']:
            if word in pos_risk_dict:
                pos_risk_dict[word] = pos_risk_dict[word]+[category]
            else:
                pos_risk_dict[word] = []
                pos_risk_dict[word] = pos_risk_dict[word]+[category]
                if word in glove_risk_dict:
                    posKeyGloveLst.append(glove_risk_dict[word][0])
                    inv_pos_key_index[pos_key_index] = word
                    pos_key_index = pos_key_index + 1

        for word in risk_keywords[category]['neg']:
            if word in neg_risk_dict:
                neg_risk_dict[word] = neg_risk_dict[word] + [category]
            else:

                neg_risk_dict[word] = []
                neg_risk_dict[word] = neg_risk_dict[word] + [category]
                if word in glove_risk_dict:
                    negKeyGloveLst.append(glove_risk_dict[word][0])
                    inv_neg_key_index[neg_key_index] = word
                    neg_key_index = neg_key_index + 1

    posKeyGloveArr = np.asarray(posKeyGloveLst)
    negKeyGloveArr = np.asarray(negKeyGloveLst)
    glove_crux = defaultdict(dict)

    glove_crux['posKeyGloveArr'] = posKeyGloveArr
    glove_crux['negKeyGloveArr'] = negKeyGloveArr
    glove_crux['inv_pos_key_index'] = inv_pos_key_index
    glove_crux['inv_neg_key_index'] = inv_neg_key_index
    glove_crux['pos_risk_dict'] = pos_risk_dict
    glove_crux['neg_risk_dict'] = neg_risk_dict

    return glove_crux
########################## Get Membership Count

def getMembershipCount(tweet_neighbors,inv_key_index,risk_dict,membership_count):
    for key_index in tweet_neighbors:
        keyword = inv_key_index[key_index]
        for category in risk_dict[keyword]:
            if category not in membership_count:
                membership_count[category] = 1
            else:
                membership_count[category] = membership_count[category] + 1
    return membership_count

############### Process Tweets Dictionary
def processTweets_old(glove_tweet_dict,glove_risk_dict,risk_keywords,tweet_dict,tweet_cmplt,tweets_classified_path):

    ## Universal Glove Dictionary
    glove_tweet_dict.update(glove_risk_dict)

    glove_crux = getWordToCategMap(risk_keywords,glove_risk_dict)

    posKeyGloveArr = glove_crux['posKeyGloveArr']
    negKeyGloveArr = glove_crux['negKeyGloveArr']
    inv_pos_key_index = glove_crux['inv_pos_key_index']
    inv_neg_key_index = glove_crux['inv_neg_key_index']
    pos_risk_dict = glove_crux['pos_risk_dict']
    neg_risk_dict = glove_crux['neg_risk_dict']

    ## Processing the tweets: Based on Similarity measure -Categorization
    print(len(tweet_dict))
    with open(tweets_classified_path,'w') as f:
        for i,tweet_id in enumerate(tweet_dict):
            tweetLst = []
            for word in tweet_dict[tweet_id]:
                if word in glove_tweet_dict:
                    tweetLst.append(glove_tweet_dict[word][0])

            ## Preparing Tweet Array
            tweetArr = np.asarray(tweetLst)

            # print('np.shape(tweetArr) : ',np.shape(tweetArr))
            # print('np.shape(posKeyGloveArr) : ',np.shape(posKeyGloveArr))
            # print('np.shape(negKeyGloveArr) : ',np.shape(negKeyGloveArr))
            if len(tweetArr)!=0:
                ## Calculating cosine similarity
                pos_cos_similarity = cosine_similarity(tweetArr,posKeyGloveArr)
                neg_cos_similarity = cosine_similarity(tweetArr,negKeyGloveArr)

                pos_nearest_neighbors = np.argsort(pos_cos_similarity, axis=1)[:,-10:]
                neg_nearest_neighbors = np.argsort(neg_cos_similarity, axis=1)[:,:10]

                pos_tweet_neighbors = [item for sublist in pos_nearest_neighbors for item in sublist]
                neg_tweet_neighbors = [item for sublist in neg_nearest_neighbors for item in sublist]

                membership_count = {}

                membership_count_pos = getMembershipCount(pos_tweet_neighbors, inv_pos_key_index, pos_risk_dict, membership_count)
                membership_count_both = getMembershipCount(neg_tweet_neighbors, inv_neg_key_index, neg_risk_dict, membership_count_pos.copy())


                ###### Getting the Categorywith maximum membership count -- POS ALONE
                v_pos = list(membership_count_pos.values())
                k_pos = list(membership_count_pos.keys())
                output_pos=k_pos[v_pos.index(max(v_pos))]

                ###### Getting the Category with maximum membership count -- BOTH POS and NEG
                v_both = list(membership_count_both.values())
                k_both = list(membership_count_both.keys())
                output_both = k_both[v_both.index(max(v_both))]

                ###### Creating the Tweet Str
                tweet_str = tweet_id + '|' + str(output_pos)  +'|' + str(output_both) + '|' + str(tweet_dict[tweet_id]) + '|' + str(tweet_cmplt[tweet_id])+'\n'

                ###### Writing the output to the tweet file

                f.write(tweet_str)


############### Process Tweets Dictionary
def processTweets(tweet_dict,risk_keywords,tweet_cmplt,tweets_classified_path,vector_type):

    result_pos,result_both = unsupervised.UnsupervisedClassifiers.CosineSimilarity().classify(tweet_dict,risk_keywords,vector_type)
    print('DONE CLASSIFYING')

    with open(tweets_classified_path,'w') as f:
        for id in result_pos:
            ###### Creating the Tweet Str
            tweet_str = id + '|' + result_pos[id]  +'|' + result_both[id] + '|' + str(tweet_dict[id]) + '|' + str(tweet_cmplt[id])+'\n'

            ###### Writing the output to the tweet file
            f.write(tweet_str)
    print('DONE WRITING')
#######################################################################
def main():
    locations = lrf_config.get_locations()
    INTERMED_DATA_PATH = locations['INTERMED_DATA_PATH']

    intermedJsonPath = os.path.join(INTERMED_DATA_PATH, 'intermed_dict.json')
    tweets_classified_path = os.path.join(INTERMED_DATA_PATH, 'tweets_classified.txt')

    refRiskCatPath = os.path.join(INTERMED_DATA_PATH, 'risk_category_file.json')

    with open(intermedJsonPath,'r') as f:
        intermed_data = json.load(f)

    with open(refRiskCatPath,'r') as f:
        risk_data = json.load(f)

    ## reading data into the dictionaries again
    tweet_dict = dict(intermed_data['tweet_dict'])
    tweet_cmplt = dict(intermed_data['tweet_cmplt'])
    # hash_tag_bag = dict(intermed_data['hash_tag_bag'])
    # user_ref_bag = dict(intermed_data['user_ref_bag'])
    # emoticon_bag = dict(intermed_data['emoticon_bag'])
    # glove_tweet_dict = intermed_data['glove_tweet_dict']

    processTweets(tweet_dict,risk_data,tweet_cmplt,tweets_classified_path,vector_type='word_embeddings')
    print('DONE')

if __name__=='__main__':
    main()