from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from lrf.utilities import utility
from lrf.configs import lrf_config


######################### Unsupervised Classifier
class UnsupervisedClassifiers:

    class CosineSimilarity:

        #################################################################
        def classify(self,data_dict,keywords,vector_type,keyword_type='both' ,glove_data_file='glove_subset.json',glove_key_file='glove_key_subset.json'):

            if vector_type == 'word_embeddings':

                print('ENTERED word_embeddings')

                pos_result,both_result = self.glove_classification(data_dict, keywords, keyword_type, glove_data_file, glove_key_file)

                print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
                print(set(data_dict.keys()).difference(set(pos_result.keys())))
                print(set(data_dict.keys()).difference(set(both_result.keys())))

            elif vector_type == 'tfidf':

                pos_result,both_result = self.tf_idf_classification(self, data_dict, keywords, 'both')

            return pos_result,both_result

        #################################################################
        def tf_idf_classification(self,data_dict,keywords,keyword_type='both'):

            data_arr = []

            data_index = {}

            for i, ind in enumerate(data_dict):

                record = data_dict[ind]

                data_index[i] = ind

                if type(record) == list:

                    new_rec = ' '.join(record)

                    data_arr.append(new_rec)

                elif type(record) == str:

                    data_arr.append(record)

            pos_risk = []

            neg_risk = []

            category_index = {}

            ind = 0

            for category in keywords:

                pos_rec = ' '.join(keywords[category]['pos'].keys())

                pos_risk.append(pos_rec)

                category_index[ind] = category

                if keyword_type == 'both':

                    neg_rec = ' '.join(keywords[category]['neg'].keys())

                    neg_risk.append(neg_rec)

            tf_idf_vectorizer = utility.get_tf_idf_vectorizer(data_arr)

            data_tfidf = tf_idf_vectorizer.transform(data_arr)

            pos_category_tfidf = tf_idf_vectorizer.transform(pos_risk)

            cos_sim_pos = cosine_similarity(data_tfidf, pos_category_tfidf)

            pos_res = np.argmax(cos_sim_pos, axis=1)

            if keyword_type == 'both':

                neg_category_tfidf = tf_idf_vectorizer.transform(neg_risk)

                cos_sim_neg = cosine_similarity(data_tfidf, neg_category_tfidf)

                cos_sim_both = cos_sim_pos - cos_sim_neg

                both_res = np.argmax(cos_sim_both, axis=1)

            pos_result = {}

            both_result = {}

            for i in data_index:

                pos_result[data_index[i]] = category_index[pos_res[i]]

                if keyword_type == 'both':

                    both_result[data_index[i]] = category_index[both_res[i]]

            return pos_result, both_result


        #################################################################
        def glove_classification(self,data_dict,keywords,keyword_type,glove_data_file,glove_key_file):

            print('ENTERED GLOVE_CLASSIFICATION')

            locations = lrf_config.get_locations()

            glove_data_dict = utility.get_glove_dict(locations['INTERMED_DATA_PATH'] + glove_data_file)

            glove_key_dict = utility.get_glove_dict(locations['INTERMED_DATA_PATH'] + glove_key_file)

            glove_crux_pos = utility.getWordToCategMap(keywords, glove_key_dict, 'pos')

            pos_key_glove_arr = glove_crux_pos['key_glove_arr']

            inv_pos_key_index = glove_crux_pos['inv_key_index']

            pos_risk_dict = glove_crux_pos['risk_dict']


            if keyword_type == 'both':

                glove_crux_neg = utility.getWordToCategMap(keywords, glove_key_dict, 'neg')

                neg_key_glove_arr = glove_crux_neg['key_glove_arr']

                inv_neg_key_index = glove_crux_neg['inv_key_index']

                neg_risk_dict = glove_crux_neg['risk_dict']

            pos_predictions = {}

            both_predictions = {}

            for id in data_dict.keys():

                data_lst = []

                for word in data_dict.get_value(id):

                    if word in glove_data_dict:

                        data_lst.append(glove_data_dict[word][0])


                ## Preparing Tweet Array
                data_arr = np.asarray(data_lst)

                if len(data_arr) != 0:

                    ## Calculating cosine similarity
                    pos_cos_similarity = cosine_similarity(data_arr, pos_key_glove_arr)

                    pos_nearest_neighbors = np.argsort(pos_cos_similarity, axis=1)[:, -10:]

                    pos_tweet_neighbors = [item for sublist in pos_nearest_neighbors for item in sublist]

                    membership_count = {}

                    membership_count_pos = utility.getMembershipCount(pos_tweet_neighbors, inv_pos_key_index,
                                                                      pos_risk_dict,
                                                                      membership_count)

                    v_pos = list(membership_count_pos.values())

                    k_pos = list(membership_count_pos.keys())

                    output_pos = k_pos[v_pos.index(max(v_pos))]

                    if keyword_type == 'both':

                        neg_cos_similarity = cosine_similarity(data_arr, neg_key_glove_arr)

                        neg_nearest_neighbors = np.argsort(neg_cos_similarity, axis=1)[:, :10]

                        neg_tweet_neighbors = [item for sublist in neg_nearest_neighbors for item in sublist]

                        membership_count_both = utility.getMembershipCount(neg_tweet_neighbors, inv_neg_key_index,
                                                                           neg_risk_dict,
                                                                           membership_count_pos.copy())
                        v_both = list(membership_count_both.values())

                        k_both = list(membership_count_both.keys())

                        output_both = k_both[v_both.index(max(v_both))]

                    pos_predictions[id] = [output_pos]

                    both_predictions[id] = [output_both] if keyword_type == 'both' else None

            return pos_predictions,both_predictions