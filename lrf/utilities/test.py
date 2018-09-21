def show_wordcloud(source, max_words=50):
    from wordcloud import WordCloud
    from matplotlib import pyplot as plt
    wordcloud = WordCloud(scale=4, max_words=1000)
    if type(source).__name__ == 'str' or type(source).__name__ == 'unicode':
        wordcloud.generate_from_text(source)
    else:
        wordcloud.generate_from_frequencies(source)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()

if __name__=='__main__':
    from lrf.utilities import datasets,utility
    import numpy as np
    from lrf.keyword_generator import keyword_generator
    from lrf.configs import lrf_config
    import os
    import json

    locations = lrf_config.get_locations()
    risk_cat_file = os.path.join(locations['INTERMED_DATA_PATH'], 'risk_category_file.json')

    news_data = datasets.get_news_data('keyword_data', 'annotator_data_dump_with_text')
    train_data_news, test_data_news = utility.split_data(news_data)
    field_names_news = ['text', 'category']

    tweet_data = datasets.get_tweet_data('txt', 'tweet_truth.txt')
    train_data_tweets, test_data_tweets = utility.split_data(tweet_data)
    field_names_tweets = ['tweet_cmplt', 'class_annotated']

    X_train_data = np.append(train_data_news[field_names_news[0]].values,train_data_tweets[field_names_tweets[0]].values)
    Y_train_data = np.append(train_data_news[field_names_news[1]].values,
                             train_data_tweets[field_names_tweets[1]].values)

    category_keywords = keyword_generator.keyword_driver('svc',X_train_data,Y_train_data,num_of_keywords=50)

    for elem in category_keywords:
        show_wordcloud(category_keywords[elem]['pos'])
        show_wordcloud(category_keywords[elem]['neg'])

    with open(risk_cat_file, 'w') as f:
        json.dump(category_keywords, f)

