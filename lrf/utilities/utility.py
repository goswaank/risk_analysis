from collections import defaultdict
import os
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from lrf.configs import lrf_config
import numpy as np
import json
import sklearn
import matplotlib.pyplot as plt
from wordcloud import WordCloud
###########################################################
def getGloveVec(uniqueKeywords):

    ## Finding Glove words for all of the above words

    lrf_path = os.getcwd()

    proj_path = os.path.abspath(os.path.join(lrf_path, os.pardir))

    ref_data_path = os.path.join(proj_path, 'reference_data')

    glove_file = os.path.join(ref_data_path, 'glove_data/glove.twitter.27B.200d.txt')

    with open(glove_file,'r') as f:

        raw_glove_data = f.readlines();

    glove_risk_dict = defaultdict(dict)


    for line in raw_glove_data:

        tokenized_glove = line.split(' ');

        if tokenized_glove[0] in uniqueKeywords:

            glove_risk_dict[tokenized_glove[0]] = [tokenized_glove[1:]]

    return glove_risk_dict


####################################################
def get_multilabel_binarizer(class_mapping):

    mlb = MultiLabelBinarizer(classes=sorted(class_mapping.keys()))

    return mlb

#####################################################
def binarize_data(data,class_mapping=None,binarizer=None):

    if class_mapping is None:
        class_mapping = lrf_config.get_class_map()

    if binarizer is None:
        binarizer = MultiLabelBinarizer(classes=sorted(class_mapping.keys()))
    try:
        fitted_binarizer = binarizer.fit(data)
        data_transformed = fitted_binarizer.transform(data)
    except Exception as e:
        print('See EXCEPTION BELOW... ')
        print(e)

    #return fitted_binarizer, data_transformed
    return fitted_binarizer,data_transformed
###################################################
def get_tf_idf_vectorizer(data,ngram_range=5):

    if all(isinstance(n, list) for n in data):

        data =get_str_from_list(data)

    tfidf_vectorizer = TfidfVectorizer(min_df=.001, ngram_range=(1, ngram_range), token_pattern='(\S+)').fit(data)

    return tfidf_vectorizer

#################################################
def get_str_from_list(list_data):

    if all(isinstance(n, list) for n in list_data):

        str_data = []

        for record in list_data:

            new_rec = ' '.join(record)

            str_data.append(new_rec)

        return str_data
    else:

        return list_data

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

###########################################
def get_glove_dict(file_path):

    with open(file_path,'r') as f:

        glove_data = dict(json.load(f))

    return glove_data
###########################################
def getWordToCategMap(risk_keywords,glove_risk_dict,key_nature):
    risk_dict = {}

    key_glove_lst = []

    inv_key_index = {}

    key_index = 0

    for category in risk_keywords:

        for word in risk_keywords[category][key_nature]:

            if word in risk_dict:

                risk_dict[word] = risk_dict[word] + [category]

            else:

                risk_dict[word] = []

                risk_dict[word] = risk_dict[word] + [category]

                if word in glove_risk_dict:

                    key_glove_lst.append(glove_risk_dict[word][0])

                    inv_key_index[key_index] = word

                    key_index = key_index + 1

    key_glove_arr = np.asarray(key_glove_lst)

    glove_crux = defaultdict(dict)

    glove_crux['key_glove_arr'] = key_glove_arr

    glove_crux['inv_key_index'] = inv_key_index

    glove_crux['risk_dict'] = risk_dict

    return glove_crux

###########################################
def getMembershipCount(data_neighbors,inv_key_index,keywords_dict,membership_count):

    for key_index in data_neighbors:

        keyword = inv_key_index[key_index]

        for category in keywords_dict[keyword]:

            if category not in membership_count:

                membership_count[category] = 1

            else:

                membership_count[category] = membership_count[category] + 1

    return membership_count
##############################################
def split_data(data,test_size = 0.25):

    train_data,test_data = sklearn.model_selection.train_test_split(data,test_size=test_size,random_state=0)

    return train_data,test_data

############################################
def show_wordcloud(source, max_words=50):
    wordcloud = WordCloud(scale=4, max_words=1000)
    if type(source).__name__ == 'str' or type(source).__name__ == 'unicode':
        wordcloud.generate_from_text(source)
    else:
        wordcloud.generate_from_frequencies(source)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()

###########################################
def main():
    locations = lrf_config.get_locations()
    glove_data_dict = get_glove_dict(locations['INTERMED_DATA_PATH'] + 'glove_key_subset.json')


if __name__=='__main__':
    main()