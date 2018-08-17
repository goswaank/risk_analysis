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

## Reading the files written so far
lrf_path = os.getcwd()
proj_path = os.path.abspath(os.path.join(lrf_path, os.pardir))
intermedDataPath = os.path.join(proj_path, 'intermed_data')
refDataPath = os.path.join(proj_path, 'reference_data')

intermedJsonPath = os.path.join(intermedDataPath, 'intermed_dict.json')

refRiskCatPath = os.path.join(refDataPath, 'risk_category_file.json')
with open(intermedJsonPath, 'r') as f:
    intermedData = json.load(f)

with open(refRiskCatPath, 'r') as f:
    riskData = json.load(f)

## reading data into the dictionaries again

tweet_dict = dict(intermedData['tweet_dict'])
hash_tag_bag = dict(intermedData['hash_tag_bag'])
user_ref_bag = dict(intermedData['user_ref_bag'])
emoticon_dict = dict(intermedData['emoticon_dict'])
glove_tweet_dict = dict(intermedData['glove_tweet_dict'])

## Reading the keywords of various categories
risk_keywords = dict(riskData['category_keywords'])
glove_risk_dict = dict(riskData['glove_risk_dict'])

glove_tweet_dict.update(glove_risk_dict)

for key_index, keyword in enumerate(glove_risk_dict):
    if keyword in glove_risk_dict:
        keywordGloveLst.append(glove_risk_dict[keyword][0])
        key_index_dict[keyword] = key_counter
        inv_key_index_dict[key_counter] = keyword;
        key_counter = key_counter + 1;

keywordGloveArr = np.array(keywordGloveLst)

## Defining the categories:
pos_risk_dict = {}
neg_risk_dict = {}

posKeyGloveLst = []
negKeyGloveLst = []
inv_pos_key_index = {}
inv_neg_key_index = {}
key_counter = 0

for category in risk_keywords:

    ## Initializing the key index to store the inverse index and word dicitonary - Pos Words
    key_index = 0
    for word in risk_keywords[category]['pos']:
        if word in pos_risk_dict:
            pos_risk_dict[word] = pos_risk_dict[word] + [category]
        else:

            pos_risk_dict[word] = []
            pos_risk_dict[word] = pos_risk_dict[word] + [category]
            posKeyGloveLst.append(glove_risk_dict[word][0])
            inv_pos_key_index[key_index] = word

    ## Re-Initializing the key index to store the inverse index and word dicitonary - Neg Words
    for word in risk_keywords[category]['neg']:
        if word in neg_risk_dict:
            neg_risk_dict[word] = neg_risk_dict[word] + [category]
        else:

            neg_risk_dict[word] = []
            neg_risk_dict[word] = neg_risk_dict[word] + [category]
            neg_risk_dict[word] = neg_risk_dict[word] + [category]
            negKeyGloveLst.append(glove_risk_dict[word][0])
            inv_neg_key_index[key_index] = word

## Processing the tweets: Based on Similarity measure -Categorization
output = {}

for tweet_id in tweet_dict:
    tweetLst = []
    pos_membership_count = {}
    neg_membership_count = {}
    for word in tweet_dict[tweet_id]:
        if word in glove_tweet_dict:
            tweetLst.append(glove_tweet_dict[word][0])

    ## Preparing Tweet Array
    tweetArr = np.asarray(tweetLst)

    ## Calculating cosine similarity
    cos_similarity = cosine_similarity(tweetArr, keywordGloveArr)

    nearest_neighbors = np.argsort(cos_similarity, axis=1)[:, -10:]

    tweet_neighbors = [item for sublist in nearest_neighbors for item in sublist]

    for key_index in tweet_neighbors:
        keyword = inv_key_index_dict[key_index]
        for category_lst in word_risk_dict[keyword]:
            if category_lst[1] == 'pos':
                if category_lst[0] not in pos_membership_count:
                    pos_membership_count[category_lst[0]] = 1
                else:
                    pos_membership_count[category_lst[0] + '__' + category_lst[1]] = membership_count[
                                                                                         category_lst[0] + '__' +
                                                                                         category_lst[1]] + 1

    print(membership_count)
    exit(0)

    label = 0  # find label
    output[tweet_id] = label

    exit(0)

# print(cosine_similarity(glove_tweet_dict['elephant'],glove_tweet_dict['browser']))