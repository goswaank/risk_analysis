import numpy as np
<<<<<<< HEAD

=======
from lrf import lrf_config
from lrf import utility
from lrf import classifier
>>>>>>> 9af9d4b1e982bbca90d4f5989640d5baa4cc6377
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
<<<<<<< HEAD
def get_keywords(train_data, classifier, binarizer, class_mapping=lrf_config.get_class_map(), num_of_keywords=20):
=======
def get_keywords(train_data,classifier,binarizer,class_mapping=lrf_config.get_class_map(),num_of_keywords=20):
>>>>>>> 9af9d4b1e982bbca90d4f5989640d5baa4cc6377

    tfidf_vocab = utility.get_tf_idf_vectorizer(train_data).vocabulary_

    tfidf_reversed_vocab = {i: word for word, i in tfidf_vocab.items()}

    top_keywords = {}

    for key in class_mapping.keys():

        keywords = get_keywords_for_tag(binarizer, classifier, key, tfidf_reversed_vocab, num_of_keywords)

        top_keywords[key] = keywords

    return top_keywords

######################### Main
def main():
    print("main_code goes here")

    classifier_type = 'svc'

<<<<<<< HEAD
    news_data = datasets.get_news_data('keyword_data', 'annotator_data_dump_with_text')
=======
    news_data = datasets.get_news_data('keyword_data','annotator_data_dump_with_text')
>>>>>>> 9af9d4b1e982bbca90d4f5989640d5baa4cc6377
    train_data,test_data = utility.split_data(news_data)

    X_train = train_data['text']
    Y_train = train_data['category']

    binarizer = utility.get_multilabel_binarizer(lrf_config.get_class_map())
    Y_train_binary = utility.binarize_data(Y_train)

    tfidf_vectorizer = utility.get_tf_idf_vectorizer(X_train)
    X_tfidf = tfidf_vectorizer.transform(X_train)

<<<<<<< HEAD
    model = classifier.get_classification_model(classifier_type, X_tfidf, Y_train_binary)
=======
    model = classifier.get_classification_model(classifier_type,X_tfidf,Y_train_binary)
>>>>>>> 9af9d4b1e982bbca90d4f5989640d5baa4cc6377

    h = get_keywords(X_train,model,binarizer)
    print(h)

#############################
if __name__=='__main__':
    main()