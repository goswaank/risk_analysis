import argparse
import os
import numpy as np
from lrf import utility, sentiment_ml_classifier
from sklearn.preprocessing import scale
import json
from scipy.sparse import coo_matrix
from sklearn.feature_selection import chi2,SelectKBest
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from lrf import lrf_config
from scipy.sparse import coo_matrix,hstack
############################ GLOBAL Variables
########################## data paths

locations = lrf_config.get_locations()
data_dir = locations['REF_DATA_PATH']+'sentiment_data'
tweets_data = 'tweets_processed.txt'
labels_data = 'labels.txt'
more_tweet_data = 'more_tweet_features.txt'
MORE_FEAT_FLAG = True

######################## select_top_k_features(data,labels,n_components=1700)
def select_top_k_features(data,labels,n_components=1700):
    data = SelectKBest(chi2, k=n_components).fit_transform(data,labels)
    return data

############################### load_and_process
def load_and_process(data_file, label_file):

    with open(os.path.join(data_dir, data_file), 'r') as f:
        x = f.readlines()
    with open(os.path.join(data_dir, label_file), 'r') as f:
        y = np.array(f.readlines())

    tfidf_vectorizer = utility.get_tf_idf_vectorizer(data=x,ngram_range=2)

    x_feats = tfidf_vectorizer.transform(x)

    x_feats = scale(x_feats, with_mean=False)

    return(x_feats, y)

################################ execute_model_selection()
def execute_model_selection():
    # Best ML Model Selection
    sentiment_ml_classifier.best_model_selection(datafile=tweets_data,labelfile=labels_data,include_more_feat=MORE_FEAT_FLAG, more_feat_file=more_tweet_data,print_report=False)

############################### execute_best_feat_selection_models()
def execute_best_feat_selection_models():

    ## Call Passive Aggressive Classifier without additional features
    p,r,f1 = sentiment_ml_classifier.execute(datafile=tweets_data, labelfile=labels_data, include_more_feat=MORE_FEAT_FLAG, more_feat_file=more_tweet_data, model_name='PA',n_components=400)
    print('Performance of PA Classifier without additional features: ')
    print('Average Precision is %f.'%p)
    print('Average Recall is %f.'%r)
    print('Average F1 is %f.'%f1)

    ## Call SGD Classifier without additional features
    p, r, f1 = sentiment_ml_classifier.execute(datafile=tweets_data, labelfile=labels_data, include_more_feat=MORE_FEAT_FLAG, more_feat_file=more_tweet_data, model_name='SGD',n_components=1610)
    print('Performance of SGD Classifier without additional features: ')
    print('Average Precision is %f.' % p)
    print('Average Recall is %f.' % r)
    print('Average F1 is %f.' % f1)

    ## Call RF without additional features
    p, r, f1 = sentiment_ml_classifier.execute(datafile=tweets_data, labelfile=labels_data, include_more_feat=MORE_FEAT_FLAG, more_feat_file=more_tweet_data, model_name='RF',n_components=2350)
    print('Performance of Random Forest Classifier without additional features: ')
    print('Average Precision is %f.' % p)
    print('Average Recall is %f.' % r)
    print('Average F1 is %f.' % f1)

    ## Call SVC Classifier without additional features
    p, r, f1 = sentiment_ml_classifier.execute(datafile=tweets_data, labelfile=labels_data, include_more_feat=MORE_FEAT_FLAG, more_feat_file=more_tweet_data, model_name='SVC',n_components=470)
    print('Performance of SVC Classifier without additional features: ')
    print('Average Precision is %f.' % p)
    print('Average Recall is %f.' % r)
    print('Average F1 is %f.' % f1)

#################################### execute_fusion()
def execute_fusion():
    ## Selection of top 3 classifiers for the matter based on F1 score from above classifiers
    ## We also create an 80-20 split

    data, labels = load_and_process(tweets_data,labels_data)


    with open(os.path.join(data_dir,more_tweet_data)) as json_data:
        more_tweet_feat = json.load(json_data)

    microblogging_features = []
    lexicon_features = []
    for elem in more_tweet_feat:
        lexicon_features.append(elem['lexicon_features'])
        microblogging_features.append(elem['microblogging_features'])

    lexicon_features = np.asarray(lexicon_features)
    microblogging_features = np.asarray(microblogging_features)

    more_data = np.hstack((lexicon_features,microblogging_features))

    # TODO: RF (2250) > SGD (1650) > PA (1600) > SVC (1400)

    data_rf = select_top_k_features(data, labels, n_components=2250)
    data_rf = hstack((data_rf, coo_matrix(more_data)))
    data_sgd = select_top_k_features(data, labels, n_components=1650)
    data_sgd = hstack((data_sgd, coo_matrix(more_data)))
    data_pa = select_top_k_features(data,labels,n_components=1600)
    data_pa = hstack((data_pa, coo_matrix(more_data)))
    data_svc = select_top_k_features(data, labels, n_components=1400)
    data_svc = hstack((data_svc, coo_matrix(more_data)))


    ## Splitting in random 80-20 splits
    indices = np.random.permutation(data.shape[0])
    train_ids, test_ids = indices[:int(0.8*data.shape[0])],indices[int(0.8*data.shape[0]):]

    train_data_rf,test_data_rf = data_rf.toarray()[train_ids,:], data_rf.toarray()[test_ids,:]
    train_data_sgd, test_data_sgd = data_sgd.toarray()[train_ids, :], data_sgd.toarray()[test_ids, :]
    train_data_pa, test_data_pa = data_pa.toarray()[train_ids, :], data_pa.toarray()[test_ids, :]
    train_data_svc, test_data_svc = data_svc.toarray()[train_ids, :], data_svc.toarray()[test_ids, :]



    train_labels,test_labels = labels[train_ids], labels[test_ids]

    # TODO: RF (2250) > SGD (1650) > PA (1600) > SVC (1400)

    rf_score = sentiment_ml_classifier.get_raw_scores(train_data_rf, train_labels, test_data_rf, model_name='RF')
    sgd_score = sentiment_ml_classifier.get_raw_scores(train_data_sgd, train_labels, test_data_sgd, model_name='SGD')
    pa_score = sentiment_ml_classifier.get_raw_scores(train_data_pa, train_labels, test_data_pa, model_name='PA')
    svc_score = sentiment_ml_classifier.get_raw_scores(train_data_svc, train_labels, test_data_svc, model_name='SVC')


    # Label Fusion
    # Order of Fusion
    label_fusion = []

    for i in range(test_labels.shape[0]):
        score_lst = [pa_score[i],sgd_score[i],svc_score[i],rf_score[i]]
        for score in set(score_lst):
            if score_lst.count(score) >=3:
                label_fusion.append(score)
                break
            elif (score_lst.count(score) == 2 and len(set(score_lst))!=2) or (score_lst.count(score)==2 and score in [score_lst[0],score_lst[1]]) :
                label_fusion.append(score)
                break
            else:
                label_fusion.append(score_lst[0])
                break


    ## Calculating Accuracy Metrics
    fus_p = precision_score(test_labels,label_fusion,average='macro')
    fus_r = recall_score(test_labels,label_fusion,average='macro')
    fus_f1 = f1_score(test_labels,label_fusion,average='macro')

    print('\n Performance of Fusion of Classifiers with additional features')
    print('Fusion Prediction is %f.'%fus_p)
    print('Recall Prediction is %f.' % fus_r)
    print('F1 Score Prediction is %f.' % fus_f1)

################################ get_args()
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(" --more_features",type=str)
    args = parser.parse_args()
    return args


############################# main()
if __name__=='__main__':
    # args = get_args()
    more_features_ind = 'true'
    if more_features_ind.lower() in ['yes','true','y']:
        MORE_FEAT_FLAG = True
    else:
        MORE_FEAT_FLAG = False


    ## Performance of different models based on features selected from Chi2 test
    # execute_model_selection()

    ## Performance of different models based on top features
    # execute_best_feat_selection_models()

    ## Performance of fusion model based on PA,SGD,NB
    execute_fusion()