import os
from lrf.metric import calculate_accuracy as ca, calculate_f_measure as cfm
from matplotlib import pyplot


def main():
    lrf_path = os.getcwd()
    proj_path = os.path.abspath(os.path.join(lrf_path, os.pardir))
    intermedDataPath = os.path.join(proj_path, 'intermed_data')
    tweetTruthPath = os.path.join(intermedDataPath, 'tweet_truth.txt')

    with open(tweetTruthPath,'r') as f:
        data = f.readlines()
    i = 0

    accuracy_pos_hist = []
    accuracy_both_hist = []
    f1_pos_hist = []
    f1_both_hist = []

    while i<2000:
        parsedData = []
        posPredictions = {}
        bothPredictions = {}
        goldenTruth = {}

        y_pos_predicted = []
        y_both_predicted = []
        y_true = []


        for line in data[i:i+200]:
            try:
                record = eval(line)
                parsedData.append(record)

                tweet_id = record[0]
                posPredictions[tweet_id] = record[1]
                bothPredictions[tweet_id] = record[2]
                goldenTruth[tweet_id] = record[4]

                y_pos_predicted.append(record[1])
                y_both_predicted.append(record[2])
                y_true.append(record[4])

            except Exception as e:
                print(record)
                print(len(record))
        i = i+200

        accuracy_pos = ca.calculate_accuracy(posPredictions,goldenTruth)
        accuracy_both = ca.calculate_accuracy(bothPredictions, goldenTruth)

        f1_pos = cfm.calculate_f_measure(y_true,y_pos_predicted)
        f1_both = cfm.calculate_f_measure(y_true, y_both_predicted)

        accuracy_pos_hist.append(accuracy_pos)
        accuracy_both_hist.append(accuracy_both)
        f1_pos_hist.append(f1_pos)
        f1_both_hist.append(f1_both)

    plotHist(accuracy_pos_hist)
    plotHist(accuracy_both_hist)
    plotHist(f1_pos_hist)
    plotHist(f1_both_hist)


def plotHist(data):
    pyplot.hist(data)
    pyplot.show()

if __name__=='__main__':
    main()