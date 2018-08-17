import os
import json
from lrf import unsupervised
from lrf import lrf_config
from lrf import classifier

############### Process Tweets Dictionary
def processTweets(tweet_dict,risk_keywords,tweet_cmplt,tweets_classified_path,vector_type):

    result_pos, result_both = classifier.classify('cosine_sim',)
    #result_pos,result_both = unsupervised.UnsupervisedClassifiers.CosineSimilarity().classify(tweet_dict,risk_keywords,vector_type)

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



    processTweets(tweet_dict,risk_data,tweet_cmplt,tweets_classified_path,vector_type='word_embeddings')

    print('DONE')

if __name__=='__main__':
    main()