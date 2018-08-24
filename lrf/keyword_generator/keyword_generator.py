import numpy as np
from lrf.configs import lrf_config
from lrf.utilities import datasets, utility
from lrf.tweet_classifier import classifier


#####################################33
def get_keywords_for_tag(binarizer,classifier, tag, tfidf_reversed_vocab,number_of_words=50):

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
def get_keywords(train_data, classifier, binarizer, class_mapping=lrf_config.get_class_map(), num_of_keywords=20):

    tfidf_vocab = utility.get_tf_idf_vectorizer(train_data).vocabulary_

    tfidf_reversed_vocab = {i: word for word, i in tfidf_vocab.items()}

    top_keywords = {}

    for key in class_mapping.keys():
        if key!='skip':

            keywords = get_keywords_for_tag(binarizer,classifier, key, tfidf_reversed_vocab,num_of_keywords)

            top_keywords[key] = keywords

    return top_keywords


######################## Driver
def keyword_driver(classifier_type,X_train,Y_train,num_of_keywords=50):

    if all(isinstance(n, list) for n in X_train):
        X_train = utility.get_str_from_list(X_train)

    binarizer = utility.get_multilabel_binarizer(lrf_config.get_class_map())

    Y_train_binary = utility.binarize_data(Y_train)

    tfidf_vectorizer = utility.get_tf_idf_vectorizer(X_train)
    X_tfidf = tfidf_vectorizer.transform(X_train)

    model = classifier.get_classification_model(classifier_type, X_tfidf, Y_train_binary[1])

    keywords = get_keywords(X_train, model, binarizer,num_of_keywords=num_of_keywords)
    return keywords

######################### Main
def main():
    # news_data = datasets.get_news_data('keyword_data','annotator_data_dump_with_text')
    # train_data,test_data = utility.split_data(news_data)
    # field_names = ['text','category']

    tweet_data = datasets.get_tweet_data('txt', 'tweet_truth.txt')
    train_data, test_data = utility.split_data(tweet_data)
    field_names = ['tweet_word_list', 'class_annotated']

    keywords = keyword_driver('svc',train_data,field_names)

    print(keywords)


#############################
if __name__=='__main__':
    main()