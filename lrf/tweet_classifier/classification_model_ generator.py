from lrf.utilities import datasets, utility
from lrf import keyword_generator
from lrf.tweet_classifier import classifier
from lrf.metric import calculate_accuracy as ca, calculate_f_measure as cf
from sklearn.model_selection import KFold
from statistics import mean
################################################################################################
######### MAIN Function
################################################################################################

def validation_model():
    ## Classifier Name
    classifier_name = 'svc'
    keyword_classifier_name = 'svc'
    train_data_src = 'tweets_and_news'
    num_splits = 5

    ## News Data
    news_data = datasets.get_news_data(folder_name='keyword_data', file_name='annotator_data_dump_with_text')

    ## Tweet Data
    tweet_data = datasets.get_tweet_data(file_type='txt', file_name='tweet_truth.txt')

    if train_data_src == 'news':
        data = news_data
        field_names = ['text', 'category']

    elif train_data_src == 'tweets':
        data = tweet_data
        field_names = ['tweet_cmplt', 'class_annotated']

    elif train_data_src == 'tweets_and_news':
        data = tweet_data
        data_extra = news_data
        field_names = ['tweet_cmplt', 'class_annotated']
        field_names_extra = ['text', 'category']

    kf = KFold(n_splits=num_splits)
    kf.get_n_splits(data)
    train_acc = []
    test_acc = []

    pos_f_measure = []
    both_f_measure = []
    pos_acc_list = []
    both_acc_list = []

    for train_index, test_index in kf.split(data):
        X_train = data[field_names[0]].iloc[train_index]
        Y_train = data[field_names[1]].iloc[train_index]
        X_test = data[field_names[0]].iloc[test_index]
        Y_test = data[field_names[1]].iloc[test_index]

        if train_data_src == 'tweets_and_news':
            X_extra = data_extra[field_names_extra[0]]
            Y_extra = data_extra[field_names_extra[1]]

            X_train = X_train.append(X_extra)
            Y_train = Y_train.append(Y_extra)

        if classifier_name in ['svc', 'lr', 'ada_boost']:
            Y_predicted, curr_train_acc, curr_test_acc = classifier.classify(classifier_name, X_train, Y_train, X_test,
                                                                             Y_test)
            train_acc.append(curr_train_acc)
            test_acc.append(curr_test_acc)

        elif classifier_name == 'cosineSim':

            keywords = keyword_generator.keyword_driver(keyword_classifier_name, X_train, Y_train, num_of_keywords=50)
            Y_predicted_pos, Y_predicted_both = classifier.classify(classifier_name, X_test, keywords=keywords)

            Y_test_list = []
            Y_pred_both_list = []
            Y_pred_pos_list = []

            for i in Y_test.keys():
                Y_test_list.append(Y_test.get_value(i))
                Y_pred_pos_list.append(Y_predicted_pos[i])
                Y_pred_both_list.append(Y_predicted_both[i])

            Y_test_binary = utility.binarize_data(Y_test_list)
            Y_pred_pos_binary = utility.binarize_data(Y_pred_pos_list)
            Y_pred_both_binary = utility.binarize_data(Y_pred_both_list)

            both_acc_list.append(ca.calculate_accuracy(Y_predicted_both, Y_test))
            both_f_measure.append(cf.calculate_f_measure(Y_test_binary[1], Y_pred_both_binary[1]))
            pos_acc_list.append(ca.calculate_accuracy(Y_predicted_pos, Y_test))
            pos_f_measure.append(cf.calculate_f_measure(Y_test_binary[1], Y_pred_pos_binary[1]))

    if classifier_name == 'svc':
        print('SVC train Acc : ', mean(train_acc))
        print('SVC test Acc : ', mean(test_acc))

    elif classifier_name == 'cosineSim':

        print('cosineSim POS Acc : ', mean(pos_acc_list))
        print('cosineSim BOTH Acc : ', mean(both_acc_list))
        print('cosineSim POS F : ', mean(pos_f_measure))
        print('cosineSim BOTH F : ', mean(both_f_measure))

def process_tweets(classifier_name='svc',train_data_src='tweets_and_news',keyword_classifier_name='svc'):

    ## News Data
    news_data = datasets.get_news_data(folder_name='keyword_data', file_name='annotator_data_dump_with_text')
    news_fields = ['text','category']

    ## Tweets Data
    training_tweet_data = datasets.get_tweet_data(file_type='txt', file_name='tweet_truth.txt')
    unlabeled_tweet_data = datasets.get_tweet_data(file_type='json', file_name='intermed_dict.json')
    tweet_fields = ['tweet_cmplt','class_annotated']

    X_test = unlabeled_tweet_data[tweet_fields[0]]

    if classifier_name == 'news':
        train_data = news_data
        X_train,Y_train = train_data[news_fields[0]], train_data[news_fields[1]]

    if classifier_name == 'tweets':
        train_data = training_tweet_data
        X_train, Y_train = train_data[tweet_fields[0]],train_data[tweet_fields[1]]

    ## Classify Below
    if train_data_src == 'tweets_and_news':
        X_train_news, Y_train_news = news_data[news_fields[0]], news_data[news_fields[1]]
        X_train_tweets, Y_train_tweets = training_tweet_data[tweet_fields[0]], training_tweet_data[tweet_fields[1]]

        X_train = X_train_tweets.append(X_train_news)
        Y_train = Y_train_tweets.append(Y_train_news)


    train_acc = []

    if classifier_name in ['svc', 'lr', 'ada_boost']:
        Y_predicted, curr_train_acc = classifier.classify(classifier_name, X_train, Y_train, X_test)
        train_acc.append(curr_train_acc)

        return Y_predicted

    elif classifier_name == 'cosineSim':

        keywords = keyword_generator.keyword_driver(keyword_classifier_name, X_train, Y_train, num_of_keywords=50)
        Y_predicted_pos, Y_predicted_both = classifier.classify(classifier_name, X_test, keywords=keywords)

        return Y_predicted_both, Y_predicted_pos

    if classifier_name == 'svc':
        print('SVC train Acc : ', mean(train_acc))




def main():
    result = process_tweets()


if __name__=='__main__':
    main()