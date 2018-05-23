import pandas as pd
import numpy as np
import math
import operator

def split_dataset(dataset, percentage_train, percentage_validation):
    # -- Find total number of tuples before splitting -- #
    count_rows = dataset.shape[0]
    # -- Split into training_set, test_set and validation_set -- #
    training_set = dataset.iloc[:count_rows * percentage_train / 100]
    test_set = dataset.iloc[count_rows * percentage_train / 100:count_rows * (100 - percentage_validation) / 100]
    validation_set = dataset.iloc[count_rows * (100 - percentage_validation) / 100:]
    return training_set, test_set, validation_set

def get_delta(tuple_1, tuple_2, length):
    # -- Calculate distance between two tuples -- #
    delta = 0
    for i in range(1, length):  # range from 1, so it skips the first letter, which is the label
        if type(tuple_1[i]) is float:
            delta += (tuple_1[i] - tuple_2[i]) * (tuple_1[i] - tuple_2[i])
        else:
            if tuple_1[i] != tuple_2[i]:
                delta += 1
    return math.sqrt(delta)

def getKNN(training_set, testing_instance, k):
    deltas = [] 
    for row_training_set in training_set:
        delta = get_delta(row_training_set, testing_instance, len(testing_instance))
        # -- Add to deltas list the dist to training_set[i] tuple and the tuple itself -- #
        deltas.append((row_training_set, delta))
    # -- Sort by the added deltas -- #
    deltas.sort(key = operator.itemgetter(1))  # sort the distances by second col
    # -- Pick the first k deltas -- #
    k_nearest_neighbours_with_deltas_to_them = deltas[:k]
    return k_nearest_neighbours_with_deltas_to_them

def get_most_common_in_list(lst):
    return max(set(lst), key=lst.count)

def getKNN_prediction(kNN):
    votes = []
    # -- Put all votes from neighbours in an array/list -- #
    for neighbour_and_delta_to_it in kNN:
        votes.append(neighbour_and_delta_to_it[0][0])
    # -- Return the most common vote -- #
    return get_most_common_in_list(votes)

def predict_dataset(k, dataset_to_predict, training_set):
    prediction_and_actual_set = []

    for testing_instance in dataset_to_predict:
        k_nearest_neighbours_and_deltas_to_them = getKNN(training_set, testing_instance, k)
        prediction_from_k_nearest = getKNN_prediction(k_nearest_neighbours_and_deltas_to_them)
        prediction_and_actual_set.append((prediction_from_k_nearest, testing_instance[0]))
        # print('prediction_from_k_nearest => ' + prediction_from_k_nearest + ', testing_instance[0] => ' + testing_instance[0])
    return prediction_and_actual_set

def get_accuracy(prediction_and_actual_set): # accuracy = (TP + TN) / (P + N)
    correct_predictions = 0
    for tuple_predicted_actual in prediction_and_actual_set:
        if tuple_predicted_actual[0] == tuple_predicted_actual[1]:
            correct_predictions += 1
    return [100.0 * correct_predictions / len(prediction_and_actual_set), correct_predictions, len(prediction_and_actual_set)]

def main(dataset):
    # -- Shuffle data ( by indexing with a shuffled index ) -- #
    dataset = dataset.iloc[np.random.permutation(len(dataset))]

    # -- Split the dataset into training_set, test_set and validation_set -- #
    training_set, test_set, validation_set = split_dataset(dataset, 60, 10)  # inputs: %train, %validation

    # -- Validation phase: find the best k first -- #
    accuracies = []
    for k in range(5, 15):
        prediction_and_actual_set = predict_dataset(k, validation_set.values, training_set.values)
        accuracy_for_k = get_accuracy(prediction_and_actual_set)
        accuracies.append((k, accuracy_for_k))
    best_k = max(accuracies, key=lambda x: x[1])[0]
    print "VALIDATION phase: "
    print "[(k, [accuracy, correct predictions, total predictions])] => ", accuracies
    print
    print "Best k => ", best_k
    print

    # -- Use best found k -- #
    prediction_and_actual_set = predict_dataset(best_k, test_set.values, training_set.values)
    final_accuracy = get_accuracy(prediction_and_actual_set)
    print "Final [accuracy, correct predictions, total predictions] => ", final_accuracy
    print
    print "[(prediction, actual label)]", prediction_and_actual_set




# file_name = 'agaricus-lepiotadata.txt'
#
# # -- Read data -- #
# dataset = pd.read_csv(file_name, header = None)
# main(dataset)
