#################### BLOCK 1

import pandas as pd
import nltk
import sklearn
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from multiprocessing import Pool
import re
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score

from lrf.utilities import risk_categories as rc

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

import os
from nltk.corpus import stopwords

############ BLOCK 2
############ Getting the Data
def getNewsData(onlyTrain=False):
    lrf_path = os.getcwd()
    proj_path = os.path.abspath(os.path.join(lrf_path, os.pardir))
    refDataPath = os.path.join(proj_path, '../reference_data')
    keywordDataPath = os.path.join(refDataPath, 'keyword_data/')

    annotated_data = pd.read_csv(keywordDataPath + "annotator_data_dump_with_text")
    annotated_data = annotated_data.drop(["Unnamed: 0", "Unnamed: 0.1"], axis=1)

    if onlyTrain==False:
        train, test = sklearn.model_selection.train_test_split(pd.DataFrame.from_records(annotated_data))
        print('################# ',type(train))
        return train,test
    else:
        return pd.DataFrame.from_records(annotated_data)

############# BLOCK 4
############# TFIDF Vectorizer
def getTfIdfVectorizer(data):
    tfidf_vectorizer = TfidfVectorizer(min_df=.001, ngram_range=(1, 5), token_pattern='(\S+)').fit(data)
    return tfidf_vectorizer

############# BLOCK 3
############# Data Preparation
def prepareData(data,categ_type=int,inv_class_mapping=None):
    if categ_type==int:
        X_data, y_data = data['text'].values, data['category'].values
        X_data = splitAndProcess(X_data, text_prepare, 8, 8)
    if categ_type==str:
        X_data, y_data = data['tweet_cmplt'].values, data['truth'].values
        X_data = splitAndProcess(X_data, text_prepare, 8, 8)
    nltk.download('stopwords')


    new_X = []
    new_y = []
    for elem_index, elem in enumerate(X_data):
        if elem != 'None':
            new_X.append(elem)
            new_y.append(y_data[elem_index])

    X_data = np.asarray(new_X)
    y_data = np.asarray(new_y)


    y_new = []
    if categ_type==int:
        for y in y_data:
            new_ind_bag = []
            ind_bag = y.replace('\"', '').replace("\'", '').strip('[]').split(',')
            ind_bag = [int(item) for item in ind_bag]
            for item in ind_bag:
                item = int(item)
                if item == -1:
                    item = 0
                new_ind_bag.append(item)
            y_new.append(new_ind_bag)
        y_data = y_new

    elif categ_type==str:
        for y in y_data:
            new_ind_bag = []
            ind_bag = y.replace('\"', '').replace("\'", '').strip('[]').split(',')
            ind_bag = [inv_class_mapping.get(item) for item in ind_bag]
            for item in ind_bag:
                item = int(item)
                if item == -1:
                    item = 0
                new_ind_bag.append(item)
            y_new.append(new_ind_bag)
        y_data = y_new

    return[X_data,y_data]


##############################################
def prepareTweetData(data, inv_class_map):

    nltk.download('stopwords')
    X_data, y_data = data['tweet_bag'].values, data['output_pos'].values
    X_data_new = []
    y_data_new = []
    for i,row in enumerate(X_data):
        print('############### ',row)
        if type(row) == str:
            ind_bag = row.replace('\"', '').replace("\'", '').strip('[]').split(',')
            new_str = ''
            for elem in ind_bag:
                new_str = new_str + elem + ' '

            X_data_new.append(new_str)
            y_data_new.append([inv_class_map.get(y_data[i])])

    X_data = X_data_new
    y_data = y_data_new

    X_data = np.asarray(X_data)
    y_data = np.asarray(y_data)

    return[X_data,y_data]
##############################################
def text_prepare(texts):
    processed_texts = []
    for i, text in enumerate(texts):
        try:
            _text = text.lower()
            _text = re.compile('[/(){}\[\]\|@,;]').sub(r' ', _text)
            _text = re.compile('[^0-9a-z #+_]').sub(r'', _text)
            _text = ' '.join([word for word in _text.split() if word not in set(stopwords.words('english'))])
            processed_texts.append(_text)
        except Exception as e:
            processed_texts.append('None')
    return processed_texts


def splitAndProcess(X, func, num_partitions, num_cores=4,text_type='news'):
    if text_type=='news':
        df_split = np.array_split(np.array(X), num_partitions)
    else:
        if text_type=='tweets':
            df_split = X
    pool = Pool(num_cores)
    X = np.concatenate(pool.map(func, df_split))
    pool.close()
    pool.join()
    return X


######################### BLOCK 14
def getTagCounts(label_data,class_mapping):
    tags_counts = {}
    for y_ind, y in enumerate(label_data):
        for tag_ind, tag in enumerate(y):
            if class_mapping[tag] in tags_counts:
                tags_counts[class_mapping[tag]] = tags_counts[class_mapping[tag]] + 1
            else:
                tags_counts[class_mapping[tag]] = 1

######################## BLOCK 16
def show_wordcloud(source, max_words=50):
    wordcloud = WordCloud(scale=4, max_words=1000)
    if type(source).__name__ == 'str' or type(source).__name__ == 'unicode':
        wordcloud.generate_from_text(source)
    else:
        wordcloud.generate_from_frequencies(source)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()



########################## BLOCK 19

def multiLabelBinarizer(data,class_mapping):
    mlb = MultiLabelBinarizer(classes=sorted(class_mapping.keys()))
    data_transformed = mlb.fit_transform(data)
    return [mlb,data_transformed]


########################## BLOCK 21
def lrClassifier(X_train_data,y_train_data,X_test_data,y_test_data):
    LR_classifier = OneVsRestClassifier(LogisticRegression(), n_jobs=-1)
    lr_model = LR_classifier.fit(X_train_data, y_train_data)
    train_y_pred_lr = lr_model.predict(X_train_data)
    test_y_pred_lr = lr_model.predict(X_test_data)

    train_accuracy =accuracy_score(y_train_data, train_y_pred_lr)
    test_accuracy = accuracy_score(y_test_data, test_y_pred_lr)

    return(LR_classifier,lr_model,train_accuracy,test_accuracy)

########################### BLOCK 22
def linearSvc(X_train_data,y_train_data,X_test_data,y_test_data):
    Svc_classifier = OneVsRestClassifier(LinearSVC(random_state=0))
    svc_model = Svc_classifier.fit(X_train_data, y_train_data)
    test_y_pred_svm = svc_model.predict(X_test_data)
    train_y_pred_svm = svc_model.predict(X_train_data)
    train_accuracy = accuracy_score(y_train_data, train_y_pred_svm)
    test_accuracy = accuracy_score(y_test_data, test_y_pred_svm)
    return (Svc_classifier,test_y_pred_svm, train_accuracy, test_accuracy)



########################### BLOCK 22
def chiSquare(X_train_data,y_train_data,num_of_keywords,X_test_data=1,y_test_data=1):
    chi2_selector = SelectKBest(chi2, k=num_of_keywords).fit(X_train_data,y_train_data)
    X_new = chi2_selector.fit_transform(X_train_data, y_train_data)
    return chi2_selector
######################### BLOCK 23
def print_words_for_tag(binarizer,classifier, tag, tfidf_reversed_vocab,class_mapping,number_of_words=50):
    cof = classifier.estimators_[binarizer.classes.index(tag)].coef_

    top_negative_words = {}
    top_positive_words = {}
    for k in np.argsort(-cof[0])[:number_of_words]:
        top_positive_words[tfidf_reversed_vocab[k]] = cof[0][k]

    for k in np.argsort(cof[0])[:number_of_words]:
        top_negative_words[tfidf_reversed_vocab[k]] = cof[0][k]


    keywords = {'pos' : top_positive_words,'neg' : top_negative_words}
    return keywords

######################## BLOCK 24
def displayKeywords(vocab,classifier,binarizer,class_mapping,num_of_keywords):
    tfidf_reversed_vocab = {i: word for word, i in vocab.items()}
    top_keywords = {}
    for key in class_mapping.keys():
        keywords = print_words_for_tag(binarizer,classifier, key, tfidf_reversed_vocab,class_mapping,num_of_keywords)
        top_keywords[class_mapping[key]] = keywords
    return top_keywords

#################################### Definition of getNewsKeywords()
def getNewsKeywords(feature_extraction,num_of_keywords,class_mapping,test=None,train=None):

    inv_class_mapping = {v: k for k, v in class_mapping.items()}
    if train is None and test is None:
        train,test = getNewsData()
        preparedTrainData = prepareData(train)
        preparedTestData = prepareData(test)

    elif train is   None and test is not None:
        train = getNewsData(onlyTrain=True)
        preparedTrainData = prepareData(train)
        preparedTestData = prepareData(test, categ_type=str, inv_class_mapping=inv_class_mapping)



    X_train,y_train = preparedTrainData[0],preparedTrainData[1]
    X_test, y_test = preparedTestData[0], preparedTestData[1]
    tf_idf_vectorizer = getTfIdfVectorizer(X_train)
    X_train_tfidf = tf_idf_vectorizer.transform(X_train)
    X_test_tfidf = tf_idf_vectorizer.transform(X_test)

    ## Classifiers Corner
    binarizer,y_train_binary = multiLabelBinarizer(y_train,class_mapping)
    binarizer, y_test_binary = multiLabelBinarizer(y_test, class_mapping)

    ## Vocabulary
    tfidf_vocab = getTfIdfVectorizer(X_train).vocabulary_

    if(feature_extraction=='lr'):
         lr_result = lrClassifier(X_train_tfidf,y_train_binary,X_test_tfidf,y_test_binary)
         classifier, y_pred,train_accuracy,test_accuracy = lr_result[0],lr_result[1],lr_result[2],lr_result[3]
         top_keywords = displayKeywords(tfidf_vocab, classifier, binarizer, class_mapping, num_of_keywords)
         print('Train Accuracy LR : ', train_accuracy)
         print('Test Accuracy LR : ', test_accuracy)
    else:
        if(feature_extraction=='svc'):
            classifier,y_pred,train_accuracy,test_accuracy = linearSvc(X_train_tfidf,y_train_binary,X_test_tfidf,y_test_binary)
            top_keywords = displayKeywords(tfidf_vocab, classifier, binarizer, class_mapping, num_of_keywords)
            print('Train Accuracy SVC : ',train_accuracy)
            print('Test Accuracy SVC : ', test_accuracy)

    #show_wordcloud(tfidf_vocab, 1500)
    sorted(tfidf_vocab.items(), key=lambda x: x[1], reverse=True)[:50]


    # train_tags_counts = getTagCounts(y_train,class_mapping)
    # test_tags_counts = getTagCounts(y_test, class_mapping)

    if(feature_extraction=='chi2'):
        classifier = chiSquare(X_train_tfidf,y_train_binary,num_of_keywords)
        top_keywords = displayKeywords(tfidf_vocab, classifier, binarizer, class_mapping, num_of_keywords)


    return top_keywords

############################### Definition of getCategKeywords()

def getCategKeywords(feature_extraction=0,num_of_keywords=20,data=0,class_mapping=''):

    inv_map = {v: k for k, v in class_mapping.items()}


    train, test = sklearn.model_selection.train_test_split(pd.DataFrame.from_records(data))

    preparedTrainData = prepareTweetData(train,inv_map)
    preparedTestData = prepareTweetData(test,inv_map)


    X_train,y_train = preparedTrainData[0],preparedTrainData[1]
    X_test, y_test = preparedTestData[0], preparedTestData[1]
    tf_idf_vectorizer = getTfIdfVectorizer(X_train)
    X_train_tfidf = tf_idf_vectorizer.transform(X_train)
    X_test_tfidf = tf_idf_vectorizer.transform(X_test)

    ## Classifiers Corner
    binarizer,y_train_binary = multiLabelBinarizer(y_train,class_mapping)
    binarizer, y_test_binary = multiLabelBinarizer(y_test, class_mapping)

    ## Vocabulary
    tfidf_vocab = getTfIdfVectorizer(X_train).vocabulary_

    if(feature_extraction=='lr'):
         lr_result = lrClassifier(X_train_tfidf,y_train_binary,X_test_tfidf,y_test_binary)
         classifier, y_pred,pred_accuracy = lr_result[0],lr_result[1],lr_result[2]
         top_keywords = displayKeywords(tfidf_vocab, classifier, binarizer, class_mapping, num_of_keywords)
    else:
        if(feature_extraction=='svc'):
            classifier,y_pred,pred_accuracy = linearSvc(X_train_tfidf,y_train_binary,X_test_tfidf,y_test_binary)
            top_keywords = displayKeywords(tfidf_vocab, classifier, binarizer, class_mapping, num_of_keywords)

    #show_wordcloud(tfidf_vocab, 1500)
    sorted(tfidf_vocab.items(), key=lambda x: x[1], reverse=True)[:50]



    if(feature_extraction=='chi2'):
        classifier = chiSquare(X_train_tfidf,y_train_binary,num_of_keywords)
        top_keywords = displayKeywords(tfidf_vocab, classifier, binarizer, class_mapping, num_of_keywords)

    return top_keywords


########################### Definition of Main function
def main():
    class_mapping = rc.getClassMap()

    inv_map = {v: k for k, v in class_mapping.items()}

    news_dict = getNewsKeywords('svc', 20, class_mapping)
    print(news_dict)
    # getNewsKeywords('lr', 50)
    #
    # lrf_path = os.getcwd()
    # proj_path = os.path.abspath(os.path.join(lrf_path, os.pardir))
    # refDataPath = os.path.join(proj_path, 'reference_data')
    # keywordDataPath = os.path.join(refDataPath, 'keyword_data/')
    # data = pd.read_csv('/home/onespace/ankita/risk_analysis/intermed_data/tweets_classified.txt', sep='|',
    #                    index_col=False, names=['tweet_id', 'output_pos', 'output_both', 'tweet_bag', 'tweet_cmplt'])
    #
    # tweet_dict = getCategKeywords('svc', 20, data, class_mapping)
    #
    # for c in inv_map:
    #     print('################################### ', c)
    #     print(tweet_dict[c]['pos'])
    #     print(news_dict[c]['pos'])


###################### Main function called if this code run
if __name__=='__main__':
    main()


