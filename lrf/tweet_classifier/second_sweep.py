import os
from lrf.keyword_generator import keyword_generation_micromort as keyGen
import pandas as pd
import json
from lrf.utilities import risk_categories as rc
from lrf.tweet_preprocessing import tweet_processing_old as tp

## Getting the Data path for the first iteration tweets classified
lrf_path = os.getcwd()
proj_path = os.path.abspath(os.path.join(lrf_path, os.pardir))
intermedDataPath = os.path.join(proj_path, 'intermed_data')
refDataPath = os.path.join(proj_path, 'reference_data')

intermedJsonPath = os.path.join(intermedDataPath,'intermed_dict.json')
tweets_classified = os.path.join(intermedDataPath,'tweets_classified.txt')
annotated_tweets = os.path.join(intermedDataPath,'tweet_truth.txt')
second_sweep_res = os.path.join(intermedDataPath,'second_sweep.txt')
refRiskCatPath = os.path.join(refDataPath,'risk_category_file.json')

with open(intermedJsonPath,'r') as f:
    intermedData = json.load(f)

with open(refRiskCatPath,'r') as f:
    riskData = json.load(f)

class_mapping = rc.getClassMap()



data = pd.read_csv(tweets_classified, sep='|', index_col=False,names=['tweet_id','output_pos','output_both','tweet_bag','tweet_cmplt'])


with open(annotated_tweets,'r') as f:
    annotated_data = f.readlines()

parsedData = []
for line in annotated_data:
    try:
        record = eval(line)
        parsedData.append(record)
        tweet_id = record[0]
    except Exception as e:
        print(record)
        print(len(record))

annotated_new = pd.DataFrame.from_records(parsedData,columns=['tweet_id','output_pos','output_both','tweet_cmplt','truth'])

keywords = keyGen.getNewsKeywords('lr',50,class_mapping,test=annotated_new)
exit(0)
keywords = keyGen.getCategKeywords('svc',20,data,class_mapping)


## reading data into the dictionaries again
tweet_dict = dict(intermedData['tweet_dict'])
tweet_cmplt = dict(intermedData['tweet_cmplt'])
hash_tag_bag = dict(intermedData['hash_tag_bag'])
user_ref_bag = dict(intermedData['user_ref_bag'])
emoticon_dict = dict(intermedData['emoticon_dict'])
glove_tweet_dict= dict(intermedData['glove_tweet_dict'])

unique_keywords = []
## To Calculate Unique Keywords:
for categ in keywords:
    unique_keywords = unique_keywords+list(keywords[categ]['pos'].keys())
    unique_keywords = unique_keywords+list(keywords[categ]['neg'].keys())

unique_keywords = set(unique_keywords)

glove_risk_dict = rc.getGloveVec(unique_keywords)


tp.processTweets(glove_tweet_dict,glove_risk_dict,keywords,tweet_dict,tweet_cmplt,second_sweep_res)
print('done !!')