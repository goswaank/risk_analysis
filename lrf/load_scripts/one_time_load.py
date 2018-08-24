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
import os
import json
from lrf.configs import connection_config as connectConf
from lrf.utilities import utility


##############################################################################
############## Parse Raw Tweet
##############################################################################
def parseTweetMetaData(collection,NUM_OF_TWEETS):
    ## Getting Tweets from
    myTweets = collection.find({}).limit(NUM_OF_TWEETS)

    user_dict = defaultdict(dict)
    raw_tweet_dict = defaultdict(dict)

    s_tweet = time.clock()

    ## Creating objects of the User and Tweet types
    for doc in myTweets:
        tweet = doc['tweet']
        user_details = tweet['user']

        if(tweet['lang'] == 'en'):
            ## Fields in Twitter Class
            tweet_id = tweet['id']
            tweet_text = tweet['text']
            tweet_lang = tweet['lang']
            tweet_timestamp = tweet['timestamp_ms']
            tweet_src = remove_tags(tweet['source'])
            tweet_is_retweet = True if 'retweeted_status' in tweet else False;
            tweet_user_id = tweet['user']['id']



            ## Extracting fields for User Class
            user_is_geo_enabled = user_details['geo_enabled']
            user_id = user_details['id']
            user_name = user_details['name']
            user_desc = user_details['description']
            user_screen_name = user_details['screen_name']
            user_follower_count = user_details['followers_count']
            user_lang = user_details['lang']
            user_location = user_details['location']
            user_time_zone = user_details['time_zone']
            user_friends_count = user_details['friends_count']

            ## Create Tweet Object
            curr_tweet = Tweet(tweet_id, tweet_text, tweet_lang, tweet_timestamp, tweet_src, tweet_is_retweet,
                                          tweet_user_id);

            ## Create User Object
            curr_user = User(user_is_geo_enabled, user_id, user_name, user_desc, user_screen_name, user_follower_count,
                             user_lang, user_location, user_time_zone, user_friends_count)

            ## Store the Tweet Object
            if tweet_id not in raw_tweet_dict:
                raw_tweet_dict[tweet_id] = curr_tweet;

            ## Store the User Object
            if user_id not in user_dict:
                user_dict[user_id] = curr_user;


    e_tweet = time.clock()
    print('Tweet_Parsing Time: ',e_tweet-s_tweet)
    return raw_tweet_dict,user_dict


##############################################################################
###################### Processing Tweet Text Content
###################### Parsing the Tweet text content to form
##############################################################################
def processTweetText(raw_tweet_dict,sentiment_dict):

    ## Defining the dictionaries to be used
    emoticon_bag = defaultdict(dict)
    hash_tag_bag = defaultdict(dict)
    user_ref_bag = defaultdict(dict)
    tweet_dict = defaultdict(dict)
    tweet_cmplt = defaultdict(dict)
    sentiment_score_dict = defaultdict(dict)


    tweetSplitter = TweetTokenizer()
    sid = SentimentIntensityAnalyzer()

    ## Downloading Stop words from NLTK
    nltkStopWords = list(stopwords.words('english'))
    stopWords = list(get_stop_words('en'))
    stopWords.extend(nltkStopWords)

    wordDict = defaultdict()

    s_token = time.clock()

    for tweet_id in raw_tweet_dict:
        tweet = raw_tweet_dict[tweet_id];
        sentence = tweet.tweet_text
        sentiment_score = sid.polarity_scores(sentence)
        wordBag = tweetSplitter.tokenize(sentence.replace('RT ',''));
        newWordBag = []
        emoticonList = []
        hashTagList = []
        userRefList = []

        for word_case in wordBag:
            word = word_case.lower()
            if word not in stopWords:
                if(word in sentiment_dict):
                    emoticonList.append(word)
                else:
                    if(word.startswith('@')):
                        userRefList.append(word.replace('@',''))
                    else:
                        if(word.startswith('#')):
                            print(word)
                            hashTagList.append(word.replace('#',''))
                        else:
                            if word.isalpha():
                                if word not in wordDict:
                                    wordDict[word] = 1;
                                newWordBag.append(word)
        if len(newWordBag) > 3:
            tweet_dict[tweet_id] = newWordBag
            hash_tag_bag[tweet_id] = hashTagList
            emoticon_bag[tweet_id] = emoticonList
            user_ref_bag[tweet_id] = userRefList
            sentiment_score_dict[tweet_id] = sentiment_score
            tweet_cmplt[tweet_id] = sentence.replace('\n',' ')

    final_dict = defaultdict(dict)

    final_dict['tweet_dict'] = tweet_dict
    final_dict['hash_tag_bag'] = hash_tag_bag
    final_dict['emoticon_bag'] = emoticon_bag
    final_dict['user_ref_bag'] = user_ref_bag
    final_dict['sentiment_score_dict'] = sentiment_score_dict
    final_dict['tweet_cmplt'] = tweet_cmplt
    final_dict['wordDict'] = wordDict

    e_token = time.clock()
    print('processTweetText() Time : ',e_token-s_token)

    return final_dict

##############################################################################
################## MAIN Function
##############################################################################
def getTweetData():
    start_time = time.clock()

    ## Getting Paths for various files
    LRF_PATH = os.getcwd()
    PROJ_PATH = os.path.abspath(os.path.join(LRF_PATH, os.pardir))
    REF_DATA_PATH = os.path.join(PROJ_PATH, 'reference_data')
    INTERMED_DATA_PATH = os.path.join(PROJ_PATH, 'intermed_data')

    SENTIMENT_FILE = os.path.join(REF_DATA_PATH, 'sentiment_data/sentiment_lexicon.txt')
    GLOVE_FILE = os.path.join(REF_DATA_PATH, 'glove_data/glove.twitter.27B.200d.txt')

    INTERMED_FILE = os.path.join(INTERMED_DATA_PATH, 'intermed_dict.json')
    GLOVE_SUBSET_FILE = os.path.join(INTERMED_DATA_PATH, 'glove_subset.json')
    sentiment_dict = utility.getSentimentData(SENTIMENT_FILE)

    ## Creating Connection to MONGO DB
    client = pymongo.MongoClient(connectConf.MONGODB_URL, connectConf.MONGODB_PORT)
    db = client[connectConf.MONGODB_DB]
    collection = db[connectConf.MONGO_COLLECTION]
    NUM_OF_TWEETS = connectConf.NUM_OF_TWEETS

    ## Get the meta data of the user from
    raw_tweet_dict,user_dict = parseTweetMetaData(collection,NUM_OF_TWEETS)
    processedTweet = processTweetText(raw_tweet_dict,sentiment_dict)

    tweet_dict = processedTweet['tweet_dict']
    hash_tag_bag  = processedTweet['hash_tag_bag']
    emoticon_bag = processedTweet['emoticon_bag']
    user_ref_bag = processedTweet['user_ref_bag']
    sentiment_score_dict = processedTweet['sentiment_score_dict']
    tweet_cmplt = processedTweet['tweet_cmplt']
    wordDict = processedTweet['wordDict']


    ## Reading Glove Dictionary and collecting only those words which are to be used for the given tweets

    ## Glove File Name

    uniqueHashTags = set([item for sublist in hash_tag_bag.values() for item in sublist])
    uniqueUserRef = set([item for sublist in user_ref_bag.values() for item in sublist])


    uniqueWords = set(wordDict).union(uniqueHashTags).union(uniqueUserRef)
    print('len(uniqueWords) : ',len(uniqueWords))

    s_glove = time.clock()
    glove_tweet_dict = utility.getGloveVec(uniqueWords)
    e_glove = time.clock()

    print('len(glove_tweet_dict) : ',len(glove_tweet_dict))
    print('GloveProcess Time : ',e_glove-s_glove)

    ## Writing the important information from current program to files

    with open(GLOVE_SUBSET_FILE, 'w') as f:
        json.dump(glove_tweet_dict, f)

    ########### Tweet Data
    final_tweet_data = defaultdict(dict)
    final_tweet_data['tweet_dict'] = tweet_dict
    final_tweet_data['tweet_cmplt'] = tweet_cmplt
    final_tweet_data['hash_tag_bag'] = hash_tag_bag
    final_tweet_data['user_ref_bag'] = user_ref_bag
    final_tweet_data['emoticon_bag'] = emoticon_bag

    #final_data['glove_tweet_dict']= glove_tweet_dict
    final_tweet_data['sentiment_score_dict'] = sentiment_score_dict

    ## Writing TWEET Data
    with open(INTERMED_FILE, 'w') as f:
        json.dump(final_tweet_data, f)

    end_time = time.clock()

    print('OVERALL TIME : ' ,end_time-start_time )
    return final_tweet_data

if __name__=='__main__':
    getTweetData()

