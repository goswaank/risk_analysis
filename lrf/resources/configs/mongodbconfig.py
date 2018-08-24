#!/user/bin/env python

mongodb_config = {
    'onespace_host' : '172.29.33.45',
    'port' : 27017,
    'db' : 'singhose',
    'tweet_collection' : 'tweets',
    'num_of_tweets' : 100000,
    'dbs': {
        'rss' : {
            "db" : 'singhose',
            "collection" : "rss"
        },
        'news_tweets': {
            "db" : "micromort",
            "collection" : "news_tweets"
        }
    }
}