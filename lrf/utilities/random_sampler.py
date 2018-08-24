import os
from random import sample
import pandas as pd

def main():
    print('hello')

    lrf_path = os.getcwd()
    proj_path = os.path.abspath(os.path.join(lrf_path, os.pardir))
    intermedDataPath = os.path.join(proj_path, 'intermed_data')
    tweets_classified_path = os.path.join(intermedDataPath, 'tweets_classified.txt')
    tweet_truth = os.path.join(intermedDataPath, 'tweet_truth.txt')

    classified_data = pd.read_csv(tweets_classified_path, sep='|', index_col=False,
                                  names=['tweet_id', 'output_pos', 'output_both', 'tweet_bag', 'tweet_cmplt'])

    tweet_id_list = classified_data['tweet_id'].values
    output_pos_list = classified_data['output_pos'].values
    output_both_list = classified_data['output_both'].values
    tweet_cmplt_list = classified_data['tweet_cmplt'].values

    soln = zip(tweet_id_list, output_pos_list, output_both_list, tweet_cmplt_list)

    print('################ SOLN ################')

    for elem in soln:
        print(elem)
        break
    original = list(soln).copy()

    random_sample = {}
    with open(tweet_truth, 'w') as f:
        for i in range(10):
            curr_random = sample(list(original), 200)
            random_sample[i] = curr_random
            for item in random_sample[i]:
                f.write(str(item))
                f.write('\n')

if __name__=='__main__':
    main()