# encoding=utf8

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from sklearn.decomposition import TruncatedSVD
from sklearn import preprocessing
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from time import time
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
from lrf.configs import lrf_config
import os
from scipy.sparse import coo_matrix,hstack

locations = lrf_config.get_locations()
# intermed_data_dir = locations['INTERMED_DATA_PATH']
ref_data_dir = os.path.join(locations['REF_DATA_PATH'],'sentiment_data')

########################################################################################################################
def select_top_k_features(data,labels,n_components=1700):
    data = SelectKBest(chi2, k=n_components).fit_transform(data,labels)
    return data

########################################################################################################################
def load_and_process(data_file, label_file):
    print("Loading data...")
    with open(os.path.join(ref_data_dir, data_file), 'r') as f:
        x = f.readlines()
    with open(os.path.join(ref_data_dir, label_file), 'r') as f:
        y = np.array(f.readlines())

    y = [elem.strip('\n') for elem in y]

    y = np.asarray(y)

    print("Extract features...")

    x_feats = TfidfVectorizer().fit_transform(x)
    x_feats = preprocessing.scale(x_feats, with_mean=False)
    return (x_feats, y)

########################################################################################################################
def transform_feat(data, n_components, n_iter=10):
    svd = TruncatedSVD(n_components=n_components, n_iter=n_iter, algorithm='randomized')
    svd.fit(data)
    data = svd.transform(data)
    return (data)

########################################################################################################################
def select_feat(data, labels, n_components):

    data = SelectKBest(chi2, k=n_components).fit_transform(data, labels)
    return data

########################################################################################################################
def train_and_predict(model, data, labels, transform, select, n_components, more_data=None, n_splits=10, print_stats=True,
                      print_report=False):
    n_splits = n_splits;
    avg_p = 0;
    avg_r = 0;
    avg_f1 = 0
    kf = KFold(n_splits=n_splits)

    if (transform):
        data = transform_feat(data, n_components=n_components)

    if (select):
        data = select_feat(data, labels, n_components=n_components)
    if more_data is not None:
        data = hstack((data,more_data))

    data = data.toarray()

    t1 = time()

    for train, test in kf.split(data):
        model.fit(data[train], labels[train])
        predicts = model.predict(data[test])
        if (print_report == 'True' or print_report == 'true'):
            print(classification_report(labels[test], predicts))
        avg_p += precision_score(labels[test], predicts, average='macro')
        avg_r += recall_score(labels[test], predicts, average='macro')
        avg_f1 += f1_score(labels[test], predicts, average='macro')

    if (print_stats):
        print('Average Compute Time is %f.' % (time() - t1))
        print('Average Precision is %f.' % (avg_p / n_splits))
        print('Average Recall is %f.' % (avg_r / n_splits))
        print('Average F1 Score is %f.' % (avg_f1 / n_splits))

    return (avg_p / n_splits, avg_r / n_splits, avg_f1 / n_splits)

########################################################################################################################
def best_model_feature_curve(model_name, data, labels, print_report, more_data=None, lower_lim=1,upper_lim=2,step=50):
    if (model_name == 'MultinomialNB'):
        model = MultinomialNB()
    elif (model_name == 'SGD'):
        model = SGDClassifier(loss='hinge', penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True,
                              max_iter=None, tol=0.001, shuffle=True, verbose=0, epsilon=0.1, n_jobs=1,
                              random_state=None,
                              learning_rate='optimal', eta0=0.0, power_t=0.5, class_weight=None, warm_start=False,
                              average=False, n_iter=None)
    elif (model_name == 'SVM' or model_name == 'SVC'):
        model = SVC(C=1, kernel='linear', degree=4, gamma='auto', coef0=0.0, shrinking=True,
                    probability=False, tol=0.001, cache_size=200, class_weight='balanced', verbose=False,
                    max_iter=-1, decision_function_shape='ovr', random_state=None)
    elif (model_name == 'RF' or model_name == 'random_forest'):
        model = RandomForestClassifier(n_estimators=10, criterion='entropy', max_depth=None, min_samples_split=2,
                                    min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto',
                                    max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None,
                                    bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False, class_weight=None)
    elif (model_name == 'PA'):
        model = PassiveAggressiveClassifier(C=1.0, fit_intercept=True, max_iter=None, tol=0.001,
                                            shuffle=True, verbose=0, loss='hinge', n_jobs=1, random_state=None,
                                            warm_start=False,
                                            class_weight='balanced', average=True, n_iter=None)

    metrics = {"precision": [], "recall": [], "f1": []}
    xaxis = []

    for i in tqdm(range(lower_lim, upper_lim+step,step)):
        p, r, f = train_and_predict(model=model, data=data, more_data=more_data, labels=labels, transform=False, select=True,
                                    n_components=i, print_stats=False, print_report=print_report)
        metrics['precision'].append(p)
        metrics['recall'].append(r)
        metrics['f1'].append(f)
        xaxis.append(i)


    # print("Optimal number of features for {} based on F1 (only) is {}").format(model_name, 50*metrics['f1'].index(max(metrics['f1'])))

    plt.figure(figsize=(10, 8))
    plt.plot(xaxis, metrics["precision"], label='Precision curve')
    plt.plot(xaxis, metrics["recall"], label='Recall curve')
    plt.plot(xaxis, metrics["f1"], label='F1-Score curve')
    plt.legend()
    plt.grid(True)
    plt.xlabel(model_name + " performance with Top-k features using Chi2")
    outpath = os.path.join(os.getcwd()+'/../../', 'plots/lrf_sentiment/feature_selection_curve_' + model_name + '.png')
    plt.savefig(outpath)

    return xaxis,metrics

############################################################################################################################
def execute(datafile, labelfile, model_name, include_more_feat, more_feat_file=None, print_report=False,n_components = 2100):
    data, labels = load_and_process(datafile, labelfile)
    # if additional data to be considered.
    if (include_more_feat):

        with open(os.path.join(ref_data_dir, more_feat_file)) as json_data:
            more_tweet_feat = json.load(json_data)

        more_data = []
        for elem in more_tweet_feat:
            tmp = elem['lexicon_features'] + elem['lexicon_features']
            more_data.append(tmp)

        # more_data = hstack((coo_matrix(more_tweet_feat['lexicon_features']),coo_matrix(more_tweet_feat['_features'])))
        more_data = np.asarray(more_data)

        more_data = preprocessing.scale(more_data, with_mean=False)

    print('model_name : ',model_name)

    if (model_name == 'SGD'):
        model = SGDClassifier(loss='hinge', penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True,
                              max_iter=None, tol=0.001, shuffle=True, verbose=0, epsilon=0.1, n_jobs=1,
                              random_state=None,
                              learning_rate='optimal', eta0=0.0, power_t=0.5, class_weight=None, warm_start=False,
                              average=False, n_iter=None)
    elif (model_name == 'SVM' or model_name == 'SVC'):
        model = SVC(C=1, kernel='linear', degree=4, gamma='auto', coef0=0.0, shrinking=True,
                    probability=False, tol=0.001, cache_size=200, class_weight='balanced', verbose=False,
                    max_iter=-1, decision_function_shape='ovr', random_state=None)
    elif (model_name == 'RF' or model_name == 'random_forest'):
        model = RandomForestClassifier(n_estimators=10, criterion='entropy', max_depth=None, min_samples_split=2,
                                    min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto',
                                    max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None,
                                    bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False, class_weight=None)
    elif (model_name == 'PA'):
        model = PassiveAggressiveClassifier(C=1.0, fit_intercept=True, max_iter=None, tol=0.001,
                                            shuffle=True, verbose=0, loss='hinge', n_jobs=1, random_state=None,
                                            warm_start=False,
                                            class_weight='balanced', average=True, n_iter=None)

    p, r, f1 = train_and_predict(model=model, data=data, more_data=coo_matrix(more_data), labels=labels, transform=False, select=True, n_components=n_components,
                                 print_stats=False, print_report=print_report)
    return ([p, r, f1])

###############################################################################################################
def best_model_selection(datafile, labelfile, include_more_feat=True, more_feat_file=None, print_report=False):
    data, labels = load_and_process(datafile, labelfile)

    # if additional data to be considered.
    if (include_more_feat):

        with open(os.path.join(ref_data_dir, more_feat_file)) as json_data:
            more_tweet_feat = json.load(json_data)

        microblogging_features = []
        lexicon_features = []
        for elem in more_tweet_feat:

            lexicon_features.append(elem['lexicon_features'])
            microblogging_features.append(elem['microblogging_features'])

        lexicon_features = np.asarray(lexicon_features)
        microblogging_features = np.asarray(microblogging_features)

        more_data = hstack((coo_matrix(lexicon_features), coo_matrix(microblogging_features)))

    model_name = 'SGD'
    xaxis, metrics = best_model_feature_curve(model_name=model_name, data=data, more_data=coo_matrix(more_data), labels=labels, print_report=print_report,lower_lim=1000,upper_lim=2000,step=10)
    print('####################### SGD :')
    max = 0
    for i in range(len(xaxis)):
        print(xaxis[i],' : ',metrics['f1'][i], ' : ',metrics['precision'][i],' : ',metrics['recall'][i])
        if metrics['f1'][i] > max:
            max = metrics['f1'][i]
            x_max = xaxis[i]

    print('%%%%%%%%%%%%%%%%%%%%%% SGD : ',max,x_max)

    model_name = 'PA'
    max = 0
    xaxis, metrics = best_model_feature_curve(model_name=model_name, data=data, more_data=coo_matrix(more_data), labels=labels, print_report=print_report,lower_lim=0,upper_lim=2000,step=50)
    print('####################### PA :')
    for i in range(len(xaxis)):
        print(xaxis[i], ' : ', metrics['f1'][i], ' : ', metrics['precision'][i], ' : ', metrics['recall'][i])
        if metrics['f1'][i] > max:
            max = metrics['f1'][i]
            x_max = xaxis[i]
    print('%%%%%%%%%%%%%%%%%%%%%% PA : ', max, x_max)

    model_name = 'SVC'
    xaxis, metrics = best_model_feature_curve(model_name=model_name, data=data, more_data=coo_matrix(more_data), labels=labels,
                             print_report=print_report,lower_lim=0,upper_lim=1000,step=10)
    max = 0
    print('####################### SVC :')
    for i in range(len(xaxis)):
        print(xaxis[i], ' : ', metrics['f1'][i], ' : ', metrics['precision'][i], ' : ', metrics['recall'][i])
        if metrics['f1'][i] > max:
            max = metrics['f1'][i]
            x_max = xaxis[i]
    print('%%%%%%%%%%%%%%%%%%%%%% SVC : ', max, x_max)

    model_name = 'RF'
    xaxis, metrics = best_model_feature_curve(model_name=model_name, data=data, more_data=coo_matrix(more_data), labels=labels,
                             print_report=print_report,lower_lim=2000,upper_lim=3000,step=50)
    max = 0
    print('####################### RF :')
    for i in range(len(xaxis)):
        print(xaxis[i], ' : ', metrics['f1'][i], ' : ', metrics['precision'][i], ' : ', metrics['recall'][i])
        if metrics['f1'][i] > max:
            max = metrics['f1'][i]
            x_max = xaxis[i]

    print('%%%%%%%%%%%%%%%%%%%%%% RF : ', max, x_max)

    plt.show()

##############################################################################################################
def get_raw_scores(train_data, train_labels, test_data, model_name):

    if (model_name == 'SGD'):
        model = SGDClassifier(loss='hinge', penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True,
                              max_iter=None, tol=0.001, shuffle=True, verbose=0, epsilon=0.1, n_jobs=1,
                              random_state=None,
                              learning_rate='optimal', eta0=0.0, power_t=0.5, class_weight=None, warm_start=False,
                              average=False, n_iter=None)
    elif (model_name == 'SVM' or model_name == 'SVC'):
        model = SVC(C=1, kernel='linear', degree=4, gamma='auto', coef0=0.0, shrinking=True,
                    probability=False, tol=0.001, cache_size=200, class_weight='balanced', verbose=False,
                    max_iter=-1, decision_function_shape='ovr', random_state=None)
    elif (model_name == 'RF' or model_name == 'random_forest'):
        model = RandomForestClassifier(n_estimators=10, criterion='entropy', max_depth=None, min_samples_split=2,
                                    min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto',
                                    max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None,
                                    bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False, class_weight=None)
    elif (model_name == 'PA'):
        model = PassiveAggressiveClassifier(C=1.0, fit_intercept=True, max_iter=None, tol=0.001,
                                            shuffle=True, verbose=0, loss='hinge', n_jobs=1, random_state=None,
                                            warm_start=False,
                                            class_weight='balanced', average=True, n_iter=None)
    model.fit(train_data, train_labels)
    predicts = model.predict(test_data)
    return (predicts)

    #TODO: RF (2250) > SGD (1650) > PA (1600) > SVC (1400)
