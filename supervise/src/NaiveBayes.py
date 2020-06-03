# Naïve Bayes Classifier
# Using some of the features to predict G3
# Calculate distribution: P(G3, age, Medu, Fedu, traveltime, studytime, freetime, (G1), (G2))
# Need to calculate distribution P(G3) and P(age|G3) First


# When using G1, G2, calculate pre-probability for usage
def calculate_probability_with(training_set):
    probability_p = []
    probability_n = []
    tuplesize = len(training_set[0])

    # Total number
    totalsize = len(training_set)
    lowerbound = {0: 15, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 0, 7: 0}

    # G3 positive situation
    age_G3_p = [0] * 8        # 1
    medu_G3_p = [0] * 5       # 2
    fedu_G3_p = [0] * 5       # 3
    tt_G3_p = [0] * 4         # 4
    st_G3_p = [0] * 4         # 5
    ft_G3_p = [0] * 5         # 6
    g1_G3_p = [0] * 21        # 7
    g2_G3_p = [0] * 21        # 8
    G3 = [0] * 2              # 9

    # G3 negative situation
    age_G3_n = [0] * 8        # 1
    medu_G3_n = [0] * 5       # 2
    fedu_G3_n = [0] * 5       # 3
    tt_G3_n = [0] * 4         # 4
    st_G3_n = [0] * 4         # 5
    ft_G3_n = [0] * 5         # 6
    g1_G3_n = [0] * 21        # 7
    g2_G3_n = [0] * 21        # 8

    # for every tuple
    for item in training_set:
        # for every attritube except the G3
        if item[-1] >= 10:
            G3[1] += 1
            for i in range(tuplesize - 1):
                if i == 0:
                    age_G3_p[item[i] - lowerbound[i]] += 1
                    continue
                if i == 1:
                    medu_G3_p[item[i] - lowerbound[i]] += 1
                    continue
                if i == 2:
                    fedu_G3_p[item[i] - lowerbound[i]] += 1
                    continue
                if i == 3:
                    tt_G3_p[item[i] - lowerbound[i]] += 1
                    continue
                if i == 4:
                    st_G3_p[item[i] - lowerbound[i]] += 1
                    continue
                if i == 5:
                    ft_G3_p[item[i] - lowerbound[i]] += 1
                    continue
                if i == 6:
                    g1_G3_p[item[i] - lowerbound[i]] += 1
                    continue
                if i == 7:
                    g2_G3_p[item[i] - lowerbound[i]] += 1
        else:
            G3[0] += 1
            for i in range(tuplesize - 1):
                if i == 0:
                    age_G3_n[item[i] - lowerbound[i]] += 1
                    continue
                if i == 1:
                    medu_G3_n[item[i] - lowerbound[i]] += 1
                    continue
                if i == 2:
                    fedu_G3_n[item[i] - lowerbound[i]] += 1
                    continue
                if i == 3:
                    tt_G3_n[item[i] - lowerbound[i]] += 1
                    continue
                if i == 4:
                    st_G3_n[item[i] - lowerbound[i]] += 1
                    continue
                if i == 5:
                    ft_G3_n[item[i] - lowerbound[i]] += 1
                    continue
                if i == 6:
                    g1_G3_n[item[i] - lowerbound[i]] += 1
                    continue
                if i == 7:
                    g2_G3_n[item[i] - lowerbound[i]] += 1

    # positive events
    probability_p.append(age_G3_p)
    probability_p.append(medu_G3_p)
    probability_p.append(fedu_G3_p)
    probability_p.append(tt_G3_p)
    probability_p.append(st_G3_p)
    probability_p.append(ft_G3_p)
    probability_p.append(g1_G3_p)
    probability_p.append(g2_G3_p)
    probability_p.append(G3)

    # negative events
    probability_n.append(age_G3_n)
    probability_n.append(medu_G3_n)
    probability_n.append(fedu_G3_n)
    probability_n.append(tt_G3_n)
    probability_n.append(st_G3_n)
    probability_n.append(ft_G3_n)
    probability_n.append(g1_G3_n)
    probability_n.append(g2_G3_n)
    probability_n.append(G3)

    return totalsize, probability_n, probability_p


# predicting and calculating measurements
def naive_bayes_with(totalsize, probability_n, probability_p, test_set):

    # Measurement
    true_positive = 0
    false_positive = 0
    false_negative = 0
    true_negative = 0
    correct = 0
    mistake = 0

    # some configuartions needed
    prob_size = len(probability_p)
    lowerbound = {0: 15, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 0, 7: 0}
    attr_size = len(test_set[0]) - 1
    num_neg = probability_n[prob_size - 1][0]
    num_pos = probability_p[prob_size - 1][0]

    # original probability
    prob_neg_org = num_neg / totalsize
    prob_pos_org = num_pos / totalsize

    # final result for each attr
    attr_prob = [[0, 0]] * attr_size

    # loop for test set
    for item in test_set:
        prob_neg = prob_neg_org
        prob_pos = prob_pos_org
        for i in range(attr_size - 1):              # indexed i in probabiloty
            attr_prob[i][0] = probability_n[i][item[i] - lowerbound[i]] / num_neg
            attr_prob[i][1] = probability_p[i][item[i] - lowerbound[i]] / num_pos
            prob_neg *= attr_prob[i][0]
            prob_pos *= attr_prob[i][1]
            if prob_pos >= prob_neg:            # 1 in prediction
                if item[-1] >= 10:
                    true_positive += 1
                    correct += 1
                else:
                    false_positive += 1
                    mistake += 1
            else:
                if item[-1] >= 10:
                    false_negative += 1
                    mistake += 1
                else:
                    true_negative += 1
                    correct += 1

    P = true_positive / (true_positive + false_positive)
    R = true_positive / (true_positive + false_negative)
    final_score = 2 * P * R / (P + R)
    accuracy = correct / (correct + mistake)
    return final_score, accuracy