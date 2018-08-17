## Basic Imports will come here

import sys
import os
sys.path.append('./')

class Tweet:    
    def __init__(self,tweet_id,tweet_text,tweet_lang,tweet_timestamp,tweet_src,tweet_is_retweet,tweet_user_id):
        self.tweet_id = tweet_id;
        self.tweet_text = tweet_text
        self.tweet_lang = tweet_lang
        self.tweet_timestamp = tweet_timestamp
        self.tweet_src = tweet_src
        self.tweet_is_retweet = tweet_is_retweet
        self.tweet_user_id = tweet_user_id