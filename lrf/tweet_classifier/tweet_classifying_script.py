from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, f1_score, recall_score,accuracy_score
import numpy as np
import re
import os
import time
from nltk.corpus import stopwords

class_mapping = {
        'health': 91, 'safety_security' : 92, 'environment' : 93,
        'social_relations' : 94, 'meaning_in_life' : 95, 'achievement': 96,
        'economics' :97 , 'politics': 98, 'not_applicable': 99, 'skip': 0}

######################## Get Tf Idf Vecorizer Function
def get_tf_idf_vectorizer(data,ngram_range=5):

    from sklearn.feature_extraction.text import TfidfVectorizer

    tfidf_vectorizer = TfidfVectorizer(min_df=.001, ngram_range=(1, ngram_range), token_pattern='(\S+)').fit(data)

    return tfidf_vectorizer

######################## get feature selector
def get_feature_selector(data,labels,n_components=1700):

    from sklearn.feature_selection import SelectKBest,chi2

    fitted_feature_selector = SelectKBest(chi2, k=n_components).fit(data,labels)

    return fitted_feature_selector


######################## Prepare Data
def prepare_data(texts):

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


####################### Get the best model
def get_best_model(x_tweet_data,y_tweet_data, x_news_data, y_news_data):

    x_tweet_data = prepare_data(x_tweet_data)
    x_news_data = prepare_data(x_news_data)


    tf_idf_vectorizer = get_tf_idf_vectorizer(x_tweet_data)

    x_tweet_data = tf_idf_vectorizer.transform(x_tweet_data)
    x_news_data = tf_idf_vectorizer.transform(x_news_data)

    y_tweet_data = [class_mapping.get(elem[0]) for elem in y_tweet_data]
    y_news_data = [class_mapping.get(elem[0]) for elem in y_news_data]

    f1_coll = []
    model_coll = []

    ########################### FOR SVC

    svc_model = SVC(C=1, kernel='linear', degree=4, gamma='auto', coef0=0.0, shrinking=True,
                probability=False, tol=0.001, cache_size=200, class_weight='balanced', verbose=False,
                max_iter=-1, decision_function_shape='ovr', random_state=None)

    fitted_model, p,r,f,acc= best_model_selection(x_tweet_data, y_tweet_data, x_news_data,
                                                                        y_news_data, model_name='SVC',
                                                                        curr_model=svc_model, lower_lim=50,
                                                                        upper_lim=5000, step=50)

    f1_coll.append(f)
    model_coll.append(fitted_model)

    print('#################### SVC : ')
    print('precision : ',p)
    print('recall : ',r)
    print('max_f1: ', f)
    print('acc: ', acc)
    f1_coll.append(f)
    model_coll.append(fitted_model)


    ########################### FOR SGD

    sgd_model = SGDClassifier(loss='hinge', penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True,
                              max_iter=None, tol=0.001, shuffle=True, verbose=0, epsilon=0.1, n_jobs=1,
                              random_state=None,
                              learning_rate='optimal', eta0=0.0, power_t=0.5, class_weight=None, warm_start=False,
                              average=False, n_iter=None)

    fitted_model, p,r,f,acc= best_model_selection(x_tweet_data, y_tweet_data, x_news_data,
                                                                        y_news_data, model_name='SGD',
                                                                        curr_model=sgd_model, lower_lim=50,
                                                                        upper_lim=5000, step=50)
    f1_coll.append(f)
    model_coll.append(fitted_model)

    print('#################### SGD : ')
    print('precision : ',p)
    print('recall : ',r)
    print('max_f1: ', f)
    print('acc: ', acc)


    ########################### FOR PA

    pa_model = PassiveAggressiveClassifier(C=1.0, fit_intercept=True, max_iter=None, tol=0.001,
                                            shuffle=True, verbose=0, loss='hinge', n_jobs=1, random_state=None,
                                            warm_start=False,
                                            class_weight='balanced', average=True, n_iter=None)

    fitted_model,p,r,f,acc= best_model_selection(x_tweet_data, y_tweet_data, x_news_data,
                                                                        y_news_data, model_name='PA',
                                                                        curr_model=pa_model, lower_lim=50,
                                                                        upper_lim=5000, step=50)

    f1_coll.append(f)
    model_coll.append(fitted_model)

    print('#################### PA : ')
    print('precision : ', p)
    print('recall : ', r)
    print('max_f1: ', f)
    print('acc: ', acc)

    best_model_ind = np.argmax(f1_coll)


    return tf_idf_vectorizer, model_coll[best_model_ind]



################################################################################################################

def predict_and_classify(model, x_tweet_data, y_tweet_data, x_news_data, y_news_data, n_splits=5, print_stats=False,
                      print_report=False):

    n_splits = n_splits;

    avg_p = 0;
    avg_r = 0;
    avg_f1 = 0
    avg_acc = 0

    kf = KFold(n_splits=n_splits)

    x_tweet_data = x_tweet_data.toarray()
    x_news_data = x_news_data.toarray()

    y_tweet_data = np.asarray(y_tweet_data)
    y_news_data = np.asarray(y_news_data)

    t1 = time.time()

    for train, test in kf.split(x_tweet_data):

        train_data = np.append(x_tweet_data[train],x_news_data,axis=0)

        train_labels = np.append(y_tweet_data[train], y_news_data)

        test_data = x_tweet_data[test]
        test_labels = y_tweet_data[test]


        model.fit(train_data, train_labels)
        predicts = model.predict(test_data)

        if (print_report == 'True' or print_report == 'true'):
            print(classification_report(test_labels, predicts))
        avg_p += precision_score(test_labels, predicts, average='weighted')
        avg_r += recall_score(test_labels, predicts, average='weighted')
        avg_f1 += f1_score(test_labels, predicts, average='weighted')
        avg_acc += accuracy_score(test_labels,predicts)

    if (print_stats):
        print('Average Compute Time is %f.' % (time.time() - t1))
        print('Average Precision is %f.' % (avg_p / n_splits))
        print('Average Recall is %f.' % (avg_r / n_splits))
        print('Average F1 Score is %f.' % (avg_f1 / n_splits))
        print('Average Acc Score is %f.' % (avg_acc / n_splits))


    return (model, avg_p / n_splits, avg_r / n_splits, avg_f1 / n_splits, avg_acc / n_splits)


#################### Plotting the Graph
def plot_graph(xaxis,metrics, plot_title):

    import matplotlib
    matplotlib.use('TkAgg')
    from matplotlib import pyplot as plt

    plt.figure(figsize=(10, 8))
    plt.plot(xaxis, metrics["precision"], label='Precision curve')
    plt.plot(xaxis, metrics["recall"], label='Recall curve')
    plt.plot(xaxis, metrics["f1"], label='F1-Score curve')
    plt.legend()
    plt.grid(True)
    plt.xlabel(plot_title + " performance with Top-k features using Chi2")
    outpath = os.path.join(os.getcwd(), 'plots/feature_selection_curve_' + plot_title + '.png')
    plt.savefig(outpath)


#################### Select Best Model
def best_model_selection(x_tweet_data, y_tweet_data, x_news_data,y_news_data ,model_name,curr_model, lower_lim, upper_lim, step):

    metrics = {}
    metrics['precision'] = []
    metrics['recall'] = []
    metrics['f1'] = []

    classifier_model, p, r, f , acc= predict_and_classify(model=curr_model, x_tweet_data=x_tweet_data,
                                                                       y_tweet_data=y_tweet_data,
                                                                       x_news_data=x_news_data, y_news_data=y_news_data)

    return classifier_model, p, r, f, acc


###################### Train Model
def train_model(raw_tweets, raw_news, model_dump_path):

    x_news_data = raw_news['text'].values
    y_news_data = raw_news['category'].values

    x_tweet_data = raw_tweets['tweet_cmplt'].values
    y_tweet_data = raw_tweets['class_annotated'].values

    tf_idf_vectorizer, model = get_best_model(x_tweet_data=x_tweet_data,y_tweet_data=y_tweet_data, x_news_data=x_news_data, y_news_data=y_news_data)

    from sklearn.externals import joblib

    joblib.dump(model,os.path.join(model_dump_path,'tweet_classifying_model.pkl'))
    joblib.dump(tf_idf_vectorizer, os.path.join(model_dump_path, 'tf_idf_vectorizer.pkl'))

    print('########### Training Complete ###########')


###################### Classifying Path ###############
def classify_tweets(tweets_list,model_path, output_path):

    tweets_list = prepare_data(tweets_list)

    from sklearn.externals import joblib

    classifier = joblib.load(os.path.join(model_path, 'tweet_classifying_model.pkl'))
    tf_idf_vectorizer = joblib.load(os.path.join(model_path, 'tf_idf_vectorizer.pkl'))

    x_data = tf_idf_vectorizer.transform(tweets_list)
    classified_data = classifier.predict(x_data)

    inv_class_map = {v:k for k,v in class_mapping.items()}

    classified_data = [inv_class_map[elem] for elem in classified_data]
    # classified_data = zip(tweets_list,classified_data)

    joblib.dump(classified_data,filename=os.path.join(output_path,'classified_tweets.pkl'))

    print('############ CLASSIFICATION COMPLETE ##################')

    return classified_data




#################### Main ############################
if __name__=='__main__':
    from lrf.utilities import datasets



    file_type = 'txt'
    file_name = 'tweet'

    raw_tweets = datasets.get_tweet_data(file_type='txt', file_name='tweet_truth.txt')
    raw_news = datasets.get_news_data(folder_name='keyword_data',file_name='annotator_data_dump_with_text')
    model_dump_path = '../../classifier_data_n_model/'
    output_path = '../../classifier_data_n_model/'

    ## Training the Model

    # train_model(raw_tweets,raw_news,model_dump_path)

    ## Classify new tweets data

    classify_tweets(raw_tweets['tweet_cmplt'],model_dump_path, output_path)

    ## Dumping the results
    from sklearn.externals import joblib
    result = joblib.load(os.path.join(output_path,'classified_tweets.pkl'))
    print('############ CLASSIFICATION COMPLETE ##################')