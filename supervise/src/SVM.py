import numpy
from math import exp
from cvxopt import solvers
from cvxopt import matrix
# Attention: cvxopt is used to solve quadratic programming problem
# It is simply a 'math' problem, making repeatable and unoptimized wheel is nothing but a waste of time


# Three types of kernel functions
def kernel_func(type_of_kernel, Xi, Xj, sigma=1):
    result = 0
    if type_of_kernel == 'Linear':
        for x, y in zip(Xi[:-1], Xj[:-1]):
            result += x * y
    elif type_of_kernel == 'Quadratic':
        for x, y in zip(Xi[:-1], Xj[:-1]):
            result += x * y
        result += 1
        result = result ** 2
    elif type_of_kernel == 'Gaussian':
        for x, y in zip(Xi[:-1], Xj[:-1]):
            result += (x - y) ** 2
        result = (0 - result) / (sigma ** 2)
        result = exp(result)
    else:
        print('You did not specify a kernel function type!')

    return result


# Calculate alpha, using quadratic programming
# ATTENTION: We need to transform max problem into min first, simply making it minus
# GOAL: Minimize (1/2)X^T * P * X + q^T * X
#       subject to: G * X ≤ h (软间隔 + 大于零)
#                   A * x = b (sum(alpha * y) = 0)
def svm_soft_margin(points, labels, kernel, c=1.0):

    # number of features
    m = len(points[0]) - 1

    # number of points
    n = len(points)

    # Constructing P checked
    P = matrix(0.0, (n, n))
    for i in range(n):
        for j in range(n):
            P[i, j] = (labels[i] * labels[j] * kernel_func(kernel, points[i], points[j]))
    # print('P')
    # print(P)

    # Constructing q checked
    q = matrix(-1.0, (n, 1))
    # print('q')
    # print(q)

    # Constructing G checked
    G = matrix(0.0, (n + n, n))
    for i in range(0, n):
        G[i, i] = 1.0

    for j in range(n, n + n):
        G[j, j - n] = -1.0
    # print('G')
    # print(G)

    # Constructing h checked
    h = matrix(0.0, (n + n, 1))
    for i in range(0, n):
        h[i, 0] = c

    for j in range(n, n + n):
        h[j, 0] = 0
    # print('h')
    # print(h)

    # Constructing A
    A = matrix(0.0, (1, n))
    for i in range(n):
        A[0, i] = labels[i]
    # print('A')
    # print(A)

    # Constructing b checked
    b = matrix(0.0, (1, 1))
    # print('b')
    # print(b)

    # Solve Quadratic Programming
    x = solvers.qp(P, q, G, h, A, b)['x']
    # x is actually alpha in the original problem
    # print(len(x))
    return x


# Make predictions checked
def svm_solver(train_set, train_labels, test_set, c, kernel='Linear'):

    # Calculate alpha
    alpha = svm_soft_margin(train_set, train_labels, kernel, c)

    # Whether or not met the constraint
    vaild_or_not = 0
    for x, y in zip(alpha, train_labels):
        vaild_or_not += x * y
    if abs(vaild_or_not - 0) < 0.000001:
        print('Test success')
    else:
        print(vaild_or_not)
        print('Constraints not met')

    # Find out which of them are support vectors
    not_zero_index = []
    for index in range(len(alpha)):
        if abs(alpha[index] - 0) > 0.001:
            not_zero_index.append(index)

    # Calculate b
    first_index = not_zero_index[0]
    b = train_labels[first_index]
    for j in range(len(alpha)):
        b = b - train_labels[j] * alpha[j] * kernel_func(kernel, train_set[first_index], train_set[j])

    # Make predictions
    predictlabel = []
    for item in test_set:
        predict_result = 0
        for sv_index in not_zero_index:
            predict_result += alpha[sv_index] * train_labels[sv_index] * kernel_func(kernel, train_set[sv_index], item)
        predict_result += b
        if predict_result >= 0:
            predictlabel.append(1)
        else:
            predictlabel.append(-1)

    # return
    return predictlabel


# Calculate accuracy and f_score checked
def calculate_measurements(predictlabel, testset):

    # Measurement
    true_positive = 1
    false_positive = 1
    false_negative = 1
    true_negative = 1
    correct = 0
    mistake = 0

    for i in range(len(testset)):
        if predictlabel[i] == 1:
            if testset[i][-1] >= 10:
                correct += 1
                true_positive += 1
            else:
                mistake += 1
                false_positive += 1
        else:
            if testset[i][-1] >= 10:
                false_negative += 1
                mistake += 1
            else:
                true_negative += 1
                correct += 1

    accuracy = correct / (correct + mistake)
    P = true_positive / (true_positive + false_positive)
    R = true_positive / (true_positive + false_negative)
    final_score = 2 * P * R / (P + R)

    return final_score, accuracy
