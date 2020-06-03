import csv
import numpy as np
import pandas as pd
import KNN as KNN
import NaiveBayes as NABE
import SVM as SVM
import cvxopt
from sys import exit


def convert_to_number(data_set, mode):
    new_set = []
    try:
        if mode == 'training':
            for data in data_set:
                new_set.append([int(x) for x in data[:len(data) - 1]] + [data[len(data) - 1]])
        elif mode == 'test':
            for data in data_set:
                new_set.append([int(x) for x in data])
        return new_set
    except ValueError:
        print('Dataset value must be string that can be turned to integer')
        exit()


def load_data_set(filename):
    try:
        with open(filename, newline='') as csvfile:
            return list(csv.reader(csvfile, delimiter=';'))
    except FileNotFoundError as e:
        raise e


def dataframe_to_list(df):
    return np.array(df).tolist()


def partition_data(list, ratio_training, ratio_test):
    mod_num = ratio_test + ratio_training
    training_set = []
    testing_set = []
    for i in range(len(list)):
        if i % mod_num <= ratio_training - 1:
            training_set.append(list[i])
        else:
            testing_set.append(list[i])
    return training_set, testing_set


# checked
def extract_label(training_set):
    ret = []
    for item in training_set:
        if item[-1] >= 10:
            ret.append(1)
            item[-1] = 1
        else:
            ret.append(-1)
            item[-1] = -1
    return ret, training_set


def predicting_using_KNN(train_mat_with, train_mat_without, train_por_with,
                         train_por_without, test_mat_with, test_mat_without,
                         test_por_with, test_por_without, k):
    # KNN predicting

    # get result
    result1 = KNN.function_knn(train_mat_without, test_mat_without, k)
    result2 = KNN.function_knn(train_mat_with, test_mat_with, k)
    result3 = KNN.function_knn(train_por_without, test_por_without, k)
    result4 = KNN.function_knn(train_por_with, test_por_with, k)

    # get measurement
    f_score1, accuracy1 = KNN.calculate_measurement(result1)
    f_score2, accuracy2 = KNN.calculate_measurement(result2)
    f_score3, accuracy3 = KNN.calculate_measurement(result3)
    f_score4, accuracy4 = KNN.calculate_measurement(result4)

    # show result
    print('KNN Mat Without G1, G2: Accuracy: ' + str(accuracy1) + '  f_score: ' + str(f_score1))
    print('KNN Mat With G1, G2: Accuracy: ' + str(accuracy2) + '  f_score: ' + str(f_score2))
    print('KNN Por Without G1, G2: Accuracy: ' + str(accuracy3) + '  f_score: ' + str(f_score3))
    print('KNN Por With G1, G2: Accuracy: ' + str(accuracy4) + '  f_score: ' + str(f_score4))

    return 0


def predicting_using_naivebayes(train_mat_with, train_por_with, test_mat_with, test_por_with):

    # get temp vars
    totalsize_mat, probability_n_mat, probability_p_mat = NABE.calculate_probability_with(train_mat_with)
    totalsize_por, probability_n_por, probability_p_por = NABE.calculate_probability_with(train_por_with)

    # get final score
    f_score_mat, accuracy_mat = NABE.naive_bayes_with(totalsize_mat, probability_n_mat, probability_p_mat, test_mat_with)
    f_score_por, accuracy_por = NABE.naive_bayes_with(totalsize_por, probability_n_por, probability_p_por, test_por_with)

    # show results
    print('NABE Mat With G1, G2: Accuracy: ' + str(accuracy_mat) + '  f_score: ' + str(f_score_mat))
    print('NABE Por With G1, G2: Accuracy: ' + str(accuracy_por) + '  f_score: ' + str(f_score_por))

    return 0


def predicting_using_SVM(train_mat_with, train_mat_without, train_por_with,
                         train_por_without, test_mat_with, test_mat_without,
                         test_por_with, test_por_without, kernel='Linear', c=1.0):

    # Divide into labels and sets
    label_mat_with, train_mat_with_t = extract_label(train_mat_with)
    label_mat_without, train_mat_without_t = extract_label(train_mat_without)
    label_por_with, train_por_with_t = extract_label(train_por_with)
    label_por_without, train_por_without_t = extract_label(train_por_without)

    # To change kernel function, use 'Quadratic' or 'Gaussian' instead
    predictlabel_mat_with = SVM.svm_solver(train_mat_with_t, label_mat_with, test_mat_with, c, kernel=kernel)
    predictlabel_mat_without = SVM.svm_solver(train_mat_without_t, label_mat_without, test_mat_without, c, kernel=kernel)
    predictlabel_por_with = SVM.svm_solver(train_por_with_t, label_por_with, test_por_with, c, kernel=kernel)
    predictlabel_por_without = SVM.svm_solver(train_por_without_t, label_por_without, test_por_without, c, kernel=kernel)

    # get measurement
    f_score1, accuracy1 = SVM.calculate_measurements(predictlabel_mat_without, test_mat_without)
    f_score2, accuracy2 = SVM.calculate_measurements(predictlabel_mat_with, test_mat_with)
    f_score3, accuracy3 = SVM.calculate_measurements(predictlabel_por_without, test_por_without)
    f_score4, accuracy4 = SVM.calculate_measurements(predictlabel_por_with, test_por_with)

    # show result
    print('Kernel function type:' + kernel)
    print('SVM Mat Without G1, G2: Accuracy: ' + str(accuracy1) + '  f_score: ' + str(f_score1))
    print('SVM Mat With G1, G2: Accuracy: ' + str(accuracy2) + '  f_score: ' + str(f_score2))
    print('SVM Por Without G1, G2: Accuracy: ' + str(accuracy3) + '  f_score: ' + str(f_score3))
    print('SVM Por With G1, G2: Accuracy: ' + str(accuracy4) + '  f_score: ' + str(f_score4))

    return 0


def main():

    # Define user file address
    filename_mat = '..\data\student-mat.csv'
    filename_por = '..\data\student-por.csv'
    # filename_mat = 'C:\AILab2\student-mat.csv'
    # filename_por = 'C:\AILab2\student-por.csv'

    # Define cols we want to use
    usecols_knn_without = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'freetime', 'health', 'G3']
    usecols_knn_with = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'freetime', 'health', 'G1', 'G2', 'G3']

    # Must choose sep, why the fuck using ; as separator?
    df_mat_with = pd.read_csv(filename_mat, usecols=usecols_knn_with, sep=';')
    df_mat_without = pd.read_csv(filename_mat, usecols=usecols_knn_without, sep=';')
    df_por_with = pd.read_csv(filename_por, usecols=usecols_knn_with, sep=';')
    df_por_without = pd.read_csv(filename_por, usecols=usecols_knn_without, sep=';')

    # Turning into list and add header into it
    original_mat_with = dataframe_to_list(df_mat_with)
    original_mat_without = dataframe_to_list(df_mat_without)
    original_por_with = dataframe_to_list(df_por_with)
    original_por_without = dataframe_to_list(df_por_without)

    # Divide the set
    train_mat_with, test_mat_with = partition_data(original_mat_with, 3, 7)
    train_mat_without, test_mat_without = partition_data(original_mat_without, 3, 7)
    train_por_with, test_por_with = partition_data(original_por_with, 3, 7)
    train_por_without, test_por_without = partition_data(original_por_without, 3, 7)

    # KNN predicting

    # set k (KNN)
    k = 9

    # set C (SVM)
    c = 1.0

    # predicting

    # Using KNN
    predicting_using_KNN(train_mat_with, train_mat_without, train_por_with, train_por_without, test_mat_with,
                         test_mat_without, test_por_with, test_por_without, k)

    # Using Naïve Bayes
    predicting_using_naivebayes(train_mat_with, train_por_with, test_mat_with, test_por_with)

    # Using SVM
    # 注意！ 请不要同时运行以下三个例子，请分开运行，不然可能会产生意外错误
    # ATTENTION: PLEASE DO NOT RUN THE THREE TEST CASES BELOW SIMULTINAEOUSLY
    # predicting_using_SVM(train_mat_with, train_mat_without, train_por_with, train_por_without, test_mat_with,
    #                      test_mat_without, test_por_with, test_por_without, 'Linear')
    # predicting_using_SVM(train_mat_with, train_mat_without, train_por_with, train_por_without, test_mat_with,
    #                      test_mat_without, test_por_with, test_por_without, 'Quadratic')
    predicting_using_SVM(train_mat_with, train_mat_without, train_por_with, train_por_without, test_mat_with,
                         test_mat_without, test_por_with, test_por_without, 'Gaussian')

    return


if __name__ == '__main__':
    main()