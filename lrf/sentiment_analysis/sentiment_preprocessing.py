from lrf.configs import lrf_config
from lrf.utilities import datasets, removers, replacers
import os
import json
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
import numpy as np


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

def get_microblogging_features(tweet):

    tt = TweetTokenizer()
    tweet_tokens = tt.tokenize(tweet)

    expanded_tweet = []

    sentiment_score = []        ## Feature 1
    intensifier_presence = 0        ## Feature 2
    diminisher_presence = 0     ## Feature 3

    for word in tweet_tokens:

        ## Check if any sentiment token present in the
        if word.lower() in sentiment_data:
            mean, variance = sentiment_data.get(word.lower())
            sentiment_score.append(mean / (0.01 + variance))

        ## Check for Diminishers and Intensifiers and Correct the Spelling if wrong
        correct_word, is_intensifier, is_diminisher = replacers.RepeatReplacer().intensifier_and_spell_chck(word)

        ## Replacing with Abbreviations if abbreviation found
        if correct_word in abbreviation_data:
            correct_word = abbreviation_data.get(correct_word)

        ## Append the correct word
        expanded_tweet.append(correct_word)


    sentiment_mean = np.mean(sentiment_score) if sentiment_score != [] else 0

    return expanded_tweet,[intensifier_presence, diminisher_presence, sentiment_mean]

###########################################################################
def get_lexicon_features(sentence):

    lexicon_features = [0] * len(lrf_config.get_sentiment_map())
    word_list = []

    for word_case in sentence:

        word = word_case.lower()

        if word not in en_stopwords:

            if word in mpqa_dataset:

                mpqa_sentiment_score = mpqa_dataset.get(word)

                lexicon_features[0] = lexicon_features[0] + mpqa_sentiment_score[0]
                lexicon_features[1] = lexicon_features[1] + mpqa_sentiment_score[1]
                lexicon_features[2] = lexicon_features[2] + mpqa_sentiment_score[2]

            word_list.append(word)

    summed_val =sum(lexicon_features)

    if summed_val != 0:

        lexicon_features = [elem/summed_val for elem in lexicon_features]

    return ' '.join(word_list), lexicon_features

###########################################################################
def negate(word_list):

    sentence = replacers.AntonymReplacer().replace_negations(word_list)

    return sentence

##########################################################################
def pre_process(str):

    ## Extra Features - Initializations

    str = removers.rm_hashtag_symbol(str)
    str = removers.rm_at_user(str)
    str = removers.rm_url(str)
    str = removers.rm_time(str)

    sentence, microblogging_features = get_microblogging_features(str)
    sentence, lexicon_features = get_lexicon_features(sentence)

    return sentence, microblogging_features, lexicon_features

########################## MAIN FUNCTION
def main():


    locations = lrf_config.get_locations()

    ref_data_dir = locations['REF_DATA_PATH'] + 'sentiment_data'

    intermed_data_dir = locations['INTERMED_DATA_PATH']


    x_filename = 'tweets.txt'

    ##load and process samples
    print('start loading and process samples...')

    tweets = []

    more_features = []

    with open(os.path.join(ref_data_dir,x_filename)) as f:

        for i, line in enumerate(f):

            tweet_meta_features = {}

            tweet_obj = json.loads(line.strip(), encoding='utf-8')

            # Twitter Text contents
            content = tweet_obj['text'].replace("\n", " ")

            postprocessed_tweet,microblogging_features, lexicon_features = pre_process(content)

            tweets.append(postprocessed_tweet)

            tweet_meta_features['microblogging_features'] = microblogging_features

            tweet_meta_features['lexicon_features'] = lexicon_features

            more_features.append(tweet_meta_features)


    # Write process tweet text to file
    with open(os.path.join(ref_data_dir, 'tweets_processed.txt'), 'w') as f:
        for tweet in tweets:
            f.write('%s\n' %tweet)


    # write additional tweet features to file
    with open(os.path.join(ref_data_dir, 'more_tweet_features.txt'), 'w', encoding='utf-8') as f:
        f.write(json.dumps(more_features, ensure_ascii=False))

    print("Preprocessing is completed")


if __name__=='__main__':
    mpqa_dataset = datasets.get_mpqa_data()
    sentiment_data = datasets.get_sentiment_data()
    abbreviation_data = datasets.get_twitter_abbreviations_data()
    en_stopwords = set(stopwords.words('english'))
    main()