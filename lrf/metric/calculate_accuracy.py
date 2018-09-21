def calculate_accuracy(predictedData,targetData):
    correct_count = 0
    total_count = 0
    for i,item in enumerate(predictedData):
        if predictedData[i]==targetData[i]:
            correct_count = correct_count+1
        total_count = total_count + 1

    accuracy = correct_count/total_count
    return accuracy