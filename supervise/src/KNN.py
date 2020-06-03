# KNN classifier
from math import sqrt
from operator import itemgetter


def find_k_neighbors(distances, k):
    return distances[0:k]


def find_class(neighbors):
    votes = [0, 0]
    for instance in neighbors:
        if instance[-2] >= 10:
            votes[1] += 1
        else:
            votes[0] += 1

    return votes


def determine_class(votes):
    if votes[1] > votes[0]:
        return 1
    else:
        return 0


def determine_origin(instance):
    if instance[-1] >= 10:
        return 1
    else:
        return 0


# main body of KNN
def function_knn(training_set, test_set, k):

    # Variables
    result = []
    distances = []
    dist = 0
    index = 0
    limit_train = len(training_set[0]) - 1
    limit_test = len(test_set[0]) - 1

    classes = [0, 1]
    # 1 means pass, when G3 >= 10, 0 otherwise
    try:
        # for each tuple of test data
        for test_instance in test_set:
            index += 1
            # for each tuple of training data
            for row in training_set:
                # form a tuple first, do not take the last onr into account
                # The result is the distance between the test tuple and the train tuple
                for x, y in zip(row[:limit_train], test_instance):
                    dist += (x-y) ** 2
                # distance got the training tuple and distance
                distances.append(row + [sqrt(dist)])
                dist = 0

            # key needs a function to specify which of the fields sort is relied on
            # itemgetter return a function of getting the specified field
            distances.sort(key=itemgetter(len(distances[0]) - 1))

            # find the nearest neighbors
            neighbors = find_k_neighbors(distances, k)

            # get the class with maximum votes
            votes = find_class(neighbors)

            # get voting result
            ispass = determine_class(votes)
            # original_judge = determine_origin(test_instance)

            # prediction tested
            # print('The predicted class for sample' + str(index) + ' is: ' + str(ispass))
            result.append(test_instance + [ispass])

            # empty the distance list
            distances.clear()
    except Exception as exp:
        print(exp)
    return result


# calculate measurement for the tests
def calculate_measurement(result):
    true_positive = 0
    false_positive = 0
    false_negative = 0
    true_negative = 0
    correct = 0
    mistake = 0
    for item in result:
        if item[-2] >= 10:
            if item[-1] == 1:
                true_positive += 1
                correct += 1
            else:
                false_negative += 1
                mistake += 1
        else:
            if item[-1] == 1:
                false_positive += 1
                mistake += 1
            else:
                true_negative += 1
                correct += 1
    P = true_positive / (true_positive + false_positive)
    R = true_positive / (true_positive + false_negative)
    final_score = 2 * P * R / (P + R)
    accuracy = correct / (correct + mistake)
    return final_score, accuracy
