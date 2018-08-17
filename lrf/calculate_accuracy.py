def calculate_accuracy(predictedData,targetData):
    correct_count = 0
    total_count = 0
    for item in predictedData.keys():

        if predictedData[item]==targetData.get_value(item):
            correct_count = correct_count+1
        total_count = total_count + 1

    accuracy = correct_count/total_count
    return accuracy