if __name__=='__main__':

    tweets = ['h3llo how are you','rab ne bana di jodi','wow, spaghetti cheese balls']

    from lrf.configs import lrf_config
    import numpy as np

    locations = lrf_config.get_locations()
    import json

    with open(locations['INTERMED_DATA_PATH']+'more_tweet_features.txt','r') as json_data:
        more_tweet_feat = json.load(json_data)

    microblogging_features = []
    lexicon_features = []

    for elem in more_tweet_feat:

        lexicon_features.append(elem['lexicon_features'])
        microblogging_features.append(elem['microblogging_features'])
        print(np.hstack((lexicon_features,microblogging_features)))



