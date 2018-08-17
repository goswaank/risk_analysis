from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from lrf import datasets
from lrf import utility
from lrf import lrf_config
import numpy as np
class FeatureSelection:

    class ChiSquare:

        def chiSquare(X_data, Y_data, num_of_keywords):

            chi2_selector = SelectKBest(chi2, k=num_of_keywords).fit_transform(X_data, Y_data)

            return chi2_selector



def main():

    fs = FeatureSelection()

    data = datasets.get_news_data('keyword_data','annotator_data_dump_with_text')

    X_data, Y_data = data['text'].values,data['category'].values

    tf_idf_vectorizer = utility.get_tf_idf_vectorizer(X_data)

    X_data_tf = tf_idf_vectorizer.transform(X_data)

    Y_binary = utility.binarize_data(data=Y_data)

    res = fs.ChiSquare.chiSquare(X_data_tf,Y_binary,50)

    for i in range(20):
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        print(res[0])

if __name__=='__main__':
    main()