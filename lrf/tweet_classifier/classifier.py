from lrf.tweet_classifier import supervised
from lrf.tweet_classifier import unsupervised
from lrf.utilities import datasets
import pandas as pd
import sklearn
from sklearn.model_selection import KFold
from lrf.utilities import utility
from lrf.keyword_generator import keyword_generator
from lrf.metric import calculate_accuracy as ca, calculate_f_measure as cf
from statistics import mean


#########################################################################
#########################################################################
def classify_bkp(classifier_type, X_train, Y_train=None,X_test=None, Y_test=None ,keywords=None, class_map = None,is_X_text=True):
    _, Y_train = utility.binarize_data(Y_train, class_mapping=class_map)
    _, Y_test = utility.binarize_data(Y_test, class_mapping=class_map)
    svc_class = supervised.SupervisedClassifier.SvcClassifier()
    Y_pred, train_acc, test_acc = svc_class.classify(X_train, Y_train, X_test, Y_test)
    return Y_pred, train_acc, test_acc

def classify(classifier_type, X_train, Y_train=None,X_test=None, Y_test=None ,keywords=None, class_map = None,is_X_text=True):

    if (classifier_type in ['svc','lr','ada_boost']) :

        if Y_train is None:

            raise ValueError(classifier_type , ' is a Supervised Algorithm, pass training labels ...')

        elif X_test is None and Y_test is None:

            train_data = zip(X_train,Y_train)

            train_data, test_data = sklearn.model_selection.train_test_split(pd.DataFrame.from_records(train_data))

            X_train, Y_train = train_data[0], train_data[1]

            X_test, Y_test = test_data[0], test_data[1]

            print('Since no TEST Data provided, splitting given data into train and test')


        X_train = utility.get_str_from_list(X_train)

        X_test = utility.get_str_from_list(X_test)

        # if class_map is not None:
        #
        #     fitted_binarizer, Y_train_binary = utility.binarize_data(Y_train,class_mapping=class_map)
        # else:
        #     fitted_binarizer, Y_train_binary = utility.binarize_data(Y_train)

        print(Y_train)
        exit(0)

        if Y_test is not None:

            f, Y_test_binary = utility.binarize_data(Y_test,class_mapping=class_map)

        if is_X_text == True:
            tf_idf_vectorizer = utility.get_tf_idf_vectorizer(X_train)

            X_train_tf_idf = tf_idf_vectorizer.transform(X_train)

            X_test_tf_idf = tf_idf_vectorizer.transform(X_test)
        else:
            X_train_tf_idf = X_train
            X_test_tf_idf = X_test

        if classifier_type == 'svc':

            svc_class = supervised.SupervisedClassifier.SvcClassifier()

            if Y_test is not None:

                Y_pred, train_acc, test_acc = svc_class.classify(X_train_tf_idf,Y_train_binary,X_test_tf_idf,Y_test_binary)

                return Y_pred, train_acc, test_acc

            else:

                Y_pred, train_acc = svc_class.classify(X_train_tf_idf, Y_train_binary, X_test_tf_idf)

                return Y_pred, train_acc

            return fitted_binarizer.inverse_transform(Y_pred),train_acc, test_acc


        elif classifier_type == 'lr':

            lr_class = supervised.SupervisedClassifier.LogisticRClassifier()

            if Y_test is not None:
                Y_pred, train_acc, test_acc = lr_class.classify(X_train_tf_idf, Y_train_binary, X_test_tf_idf, Y_test_binary)
                return Y_pred, train_acc, test_acc

            else:
                Y_pred, train_acc = lr_class.classify(X_train_tf_idf, Y_train_binary, X_test_tf_idf)
                return Y_pred, train_acc


        elif classifier_type == 'ada_boost':

            ada_boost_class = supervised.SupervisedClassifier.AdaBoostClassifier()

            if Y_test is not None:
                Y_pred, train_acc, test_acc = ada_boost_class.classify(X_train_tf_idf, Y_train_binary, X_test_tf_idf,
                                                            Y_test_binary)

                return Y_pred, train_acc, test_acc

            else:
                Y_pred, train_acc = ada_boost_class.classify(X_train_tf_idf, Y_train_binary, X_test_tf_idf)

                return Y_pred, train_acc

    elif classifier_type == 'cosineSim':

        cosine_sim_class = unsupervised.UnsupervisedClassifiers.CosineSimilarity()

        Y_pred_pos,Y_pred_both = cosine_sim_class.classify(X_train,keywords,vector_type='word_embeddings')

        return Y_pred_pos,Y_pred_both

#######################################################
def get_classifier(classifier_type):

    if classifier_type == 'svc':

        svc_class = supervised.SupervisedClassifier.SvcClassifier()

        classifier = svc_class.get_classifier()

        return classifier

    elif classifier_type == 'lr':

        lr_class = supervised.SupervisedClassifier.LogisticRClassifier()

        classifier = lr_class.get_classifier()

        return classifier


#######################################################
def get_classification_model(classifier_type,X_train,Y_train):

    if classifier_type == 'svc':

        svc_class = supervised.SupervisedClassifier.SvcClassifier()

        classifier = svc_class.get_model(X_train,Y_train)

        return classifier

    elif classifier_type == 'lr':

        lr_class = supervised.SupervisedClassifier.LogisticRClassifier()

        classifier = lr_class.get_model(X_train,Y_train)

        return classifier
########################### Get Best Model
########################### Definition of Main function
def main():
    news_dict = datasets.get_news_data(folder_name='keyword_data',file_name='annotator_data_dump_with_text')

    category_names = ['tweet_cmplt','class_annotated']
    category_names_news = ['text', 'category']

    twitter_dict = datasets.get_tweet_data('txt', 'tweet_truth.txt')

    kf = KFold(n_splits=5)
    kf.get_n_splits(twitter_dict)

    some_dict = {}
    train_acc = []
    test_acc = []

    acc_both = []
    f_both = []
    acc_pos = []
    f_pos = []

    ada_test_list = []
    ada_train_list = []

    news_train = news_dict['text']
    news_class = news_dict['category']

    for train_index, test_index in kf.split(twitter_dict):
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

        X_train = twitter_dict['tweet_cmplt'].iloc[train_index]
        Y_train = twitter_dict['class_annotated'].iloc[train_index]
        X_test = twitter_dict['tweet_cmplt'].iloc[test_index]
        Y_test = twitter_dict['class_annotated'].iloc[test_index]

        some_dict['tweet_cmplt'] = X_train.append(news_train)
        some_dict['class_annotated'] = Y_train.append(news_class)

        ada_predicted, ada_train_acc, ada_test_acc = classify('ada_boost', some_dict['tweet_cmplt'],
                                                            some_dict['class_annotated'], X_test, Y_test)

        ada_train_list.append(ada_train_acc)
        ada_test_list.append(ada_test_acc)

        exit(0)

        print('ada_train_list : ',ada_train_list)
        print('ada_test_list : ',ada_test_list)

        keywords = keyword_generator.keyword_driver('svc',some_dict['tweet_cmplt'],some_dict['class_annotated'],num_of_keywords=50)

        for item in keywords:
            print(item , ' : ',keywords[item])

        predicted,curr_train_acc,curr_test_acc = classify('svc',some_dict['tweet_cmplt'],some_dict['class_annotated'],X_test,Y_test)

        train_acc.append(curr_train_acc)
        test_acc.append(curr_test_acc)

        print('train_acc SVC: ',train_acc)
        print('test_acc SVC: ', test_acc)

        # Y_pred_pos, Y_pred_both = classify('cosineSim', X_test ,keywords = keywords)

        Y_test_list = []
        Y_pred_both_list = []
        Y_pred_pos_list = []

        for i in Y_test.keys():
            Y_test_list.append(Y_test.get_value(i))
            Y_pred_pos_list.append(Y_pred_pos[i])
            Y_pred_both_list.append(Y_pred_both[i])

        Y_test_binary = utility.binarize_data(Y_test_list)
        Y_pred_pos_binary = utility.binarize_data(Y_pred_pos_list)
        Y_pred_both_binary = utility.binarize_data(Y_pred_both_list)


        acc_both.append(ca.calculate_accuracy(Y_pred_both,Y_test))
        f_both.append(cf.calculate_f_measure(Y_test_binary[1],Y_pred_both_binary[1]))
        acc_pos.append(ca.calculate_accuracy(Y_pred_pos, Y_test))
        f_pos.append(cf.calculate_f_measure(Y_test_binary[1], Y_pred_pos_binary[1]))

    print('################################ BOTH')
    print('acc_both : ', mean(acc_both))
    print('f_both : ', mean(f_both))
    print('################################ POS')
    print('acc_pos : ', mean(acc_pos))
    print('f_pos : ', mean(f_pos))
    print('############################### SVC')
    print('Train_Accuracy : ', mean(train_acc))
    print('Test_Accuracy : ', mean(test_acc))
    print('############################### ADA_Boost')
    print('Train_Accuracy : ', mean(ada_train_list))
    print('Test_Accuracy : ', mean(ada_test_list))
    exit(0)

    # TWEET DATA
    twitter_dict = datasets.get_tweet_data('txt', 'tweet_truth.txt')
    train_data,test_data = utility.split_data(twitter_dict)
    category_names = ['tweet_cmplt','class_annotated']
    #category_names_tweet = ['tweet_word_list', 'class_annotated']

    predicted_data, train_acc, test_acc = classify('lr',train_data[category_names[0]],train_data[category_names[1]],test_data[category_names[0]],test_data[category_names[1]])
    #predicted_data, train_acc, test_acc = classify('svc', news_dict[category_names_news[0]], news_dict[category_names_news[1]],
    #                                                   twitter_dict[category_names_tweet[0]], twitter_dict[category_names_tweet[1]])
    # print(predicted_data)

    print('train_acc : ', train_acc)

    print('test_acc : ', test_acc)


###################### Main function called if this code run
if __name__=='__main__':
    main()