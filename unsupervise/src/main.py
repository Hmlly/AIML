import pandas as pd
import numpy as np
import random
import math
# for visualization
# import matplotlib.pyplot as plt


def dataframe_to_list(df):
    return np.array(df).tolist()


def separate_label_with_attr(data_set):
    labels = []
    for item in data_set:
        labels.append(item[0])
        item.remove(item[0])
    return labels, data_set


# standardrization
def standardrized_data_set(data_set):
    total_num = len(data_set)
    sum_of_feature = [0] * len(data_set[0])
    for item in data_set:
        for i in range(len(data_set[0])):
            sum_of_feature[i] += item[i]
    means_of_feature = []
    for num in sum_of_feature:
        means_of_feature.append(num / total_num)
    std_deviation_sq = [0] * len(data_set[0])
    for item in data_set:
        for i in range(len(data_set[0])):
            std_deviation_sq[i] += (item[i] - means_of_feature[i]) ** 2
    for i in range(len(std_deviation_sq)):
        std_deviation_sq[i] = math.sqrt(std_deviation_sq[i] / total_num)
    for item in data_set:
        for i in range(len(data_set[0])):
            item[i] = (item[i] - means_of_feature[i]) / std_deviation_sq[i]
    return data_set


# get corvariance matrix
def get_corvariance_mat(data_set):
    mat_data = np.array(data_set)
    mat_data = mat_data.T
    cor_mat_data = (1 / len(data_set)) * np.dot(mat_data, mat_data.T)
    return cor_mat_data


# Goal: First compute the linear transformation P
def PCA(data, threshold):

    # Data matrix
    data_mat = np.array(data)
    data_mat = data_mat.T
    # print(data_mat.shape)

    cor_mat = get_corvariance_mat(data)
    # print(cor_mat.shape)
    # Get eigvals and vecs
    eigen_val, eigen_vec = np.linalg.eig(cor_mat)

    # Turn into lists
    eigen_sum = 0
    eigen_val_list = eigen_val.tolist()
    eigen_vec_list = eigen_vec.tolist()
    for item in eigen_vec_list:
        for i in range(len(item)):
            item[i] = abs(item[i])
    eig_info = []
    for i in range(len(eigen_val_list)):
        eigen_sum += abs(eigen_val_list[i])
        temp = [abs(eigen_val_list[i]), eigen_vec_list[i]]
        eig_info.append(temp)

    # already sorted

    first_m_vec = eig_info[0][0]
    first_m_minus_vec = 0
    # m is the last index we want to take
    m = 0
    while True:
        # print(m)
        left_val = first_m_minus_vec / eigen_sum
        right_val = first_m_vec / eigen_sum
        if left_val < threshold and threshold <= right_val:
            break
        else:
            first_m_minus_vec += eig_info[m][0]
            m += 1
            first_m_vec += eig_info[m][0]
            if m == len(eigen_vec):
                m = m - 1
                break

    # count m to threshold
    print('After PCA dimensons: ' + str(m + 1))

    # get project matrix
    project_mat_list = []
    for i in range(m + 1):
        project_mat_list.append(eig_info[m][1])
    project_mat = np.array(project_mat_list)
    project_mat = project_mat.T
    # print(project_mat.shape)

    # dimenson reduction
    PCA_data_mat = np.dot(project_mat.T, data_mat)
    PCA_data_mat = PCA_data_mat.T
    PCA_data = PCA_data_mat.tolist()

    return PCA_data


def calculate_distance(pt1, pt2):
    dist = 0
    for x, y in zip(pt1, pt2):
        dist += (x - y) ** 2
    distance = math.sqrt(dist)
    return distance


def calculate_sihouette(data, class_info, pivot_pt):
    sihouette = 0
    for i in range(len(data)):
        # calculate distance within the cluster
        total_dist = 0
        total_num = 0
        for j in range(len(data)):
            if class_info[i] == class_info[j]:
                total_dist += calculate_distance(data[i], data[j])
                total_num += 1
        temp_a = total_dist / total_num
        # calculate the nearest cluster
        least_dist = 100
        least_index = -1
        for index in range(len(pivot_pt)):
            cur_dist = calculate_distance(data[i], pivot_pt[index])
            if cur_dist < least_dist:
                least_dist = cur_dist
                least_index = index
        # calculate the distance to the points of the nearest cluster
        total_near_dist = 0
        total_near_num = 0
        for j in range(len(data)):
            if class_info[j] == least_index:
                total_near_dist += calculate_distance(data[i], data[j])
                total_near_num += 1
        temp_b = total_near_dist / total_near_num
        # print(i, temp_a, temp_b)
        sihouette += (temp_b - temp_a) / max(temp_b, temp_a)
    sihouette = sihouette / len(data)
    return sihouette


def KMeans(k ,data):

    number_of_data = len(data)
    class_info = [0] * number_of_data
    pivot_pt = []

    # k original random picked points
    k_index = random.sample(range(0, number_of_data), k)
    for index in k_index:
        pivot_pt.append(data[index])

    # number of iterations:
    # when iteration times > 1, bug exists
    iter_num = 1
    while iter_num > 0:

        # assign class for each point
        for i in range(number_of_data):
            least_distance = 10000
            for j in range(k):
                cur_dist = calculate_distance(data[i], pivot_pt[j])
                if cur_dist < least_distance:
                    least_distance = cur_dist
                    class_info[i] = j

        # prepare for pivot transforming
        pivot_pt = []
        for i in range(k):
            pivot_pt.append([0] * len(data[0]))
        num_of_class = [0] * k

        # pivot transforming
        for i in range(number_of_data):
            num_of_class[class_info[i]] += 1
            class_idx = class_info[i]
            for x in range(len(data[0])):
                (pivot_pt[class_idx])[x] += data[i][x]

        class_index = 0
        for item in pivot_pt:
            for x in range(len(data[0])):
                if num_of_class[class_index] != 0:
                    item[x] = item[x] / num_of_class[class_index]
            class_index += 1

        # iteration goes up
        iter_num = iter_num - 1

    sihouette_coe = calculate_sihouette(data, class_info, pivot_pt)

    for item_index in range(len(data)):
        data[item_index].append(class_info[item_index])

    return data, sihouette_coe


def calculate_rand(data, org_class):
    a = 0
    b = 0
    c = 0
    d = 0
    for i in range(len(data)):
        for j in range(len(data)):
            if data[i][-1] == data[j][-1]:
                if org_class[i] == org_class[j]:
                    a += 1
                    # correct
                else:
                    # mistake
                    c += 1
            else:
                if org_class[i] == org_class[j]:
                    # mistake
                    b += 1
                else:
                    # correct
                    d += 1
    return (a + d) / (a + b + c + d)

"""
def plot_radar(data, save):

    # get label
    label = []
    for item in data:
        label.append(item[-1])

    # num of attr
    N = len(data[0])

    # configure radar
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))

    # size
    fig = plt.figure(figsize=(12, 12))

    # polar coordinates
    ax = fig.add_subplot(111, polar=True)

    # samples
    sam = ['r-', 'm-', 'g-', 'b-', 'y-', 'k-', 'w-', 'c-'] * 30

    # markers
    lab = []

    for i in range(len(data)):
        values = data[i]
        feature = ['attr1', 'attr2', 'attr3', 'attr4',
                   'attr5', 'attr6', 'attr7', 'attr8',
                   'attr9', 'attrA', 'attrB', 'attrC']
        # set properties
        values = np.concatenate((values, [values[0]]))
        ax.plot(angles, values, sam[i], linewidth=2)
        ax.fill(angles, values, alpha=0.5)
        ax.set_thetagrids(angles * 180 / np.pi, feature)
        ax.set_ylim(auto=True)
        plt.title('GLCM Graph')
        ax.grid(True)
        lab.append('Neighboring cluster: ' + str(i + 1))

    plt.savefig(save)
    plt.show()
    return
"""



def main():

    # Define user file address
    filename_wine = '../input/wine_data.csv'

    # read into dataframe
    df_wine_data = pd.read_csv(filename_wine, sep=',')

    # turning into list
    original_wine_data = dataframe_to_list(df_wine_data)

    # show and test
    # print(original_wine_data)

    # get labels and dataset
    wine_class, wine_data_un = separate_label_with_attr(original_wine_data)

    # standardrized
    wine_data = standardrized_data_set(wine_data_un)

    # dimenson reduction
    dr_wine_data = PCA(wine_data, 0.93)

    # set k
    k = 4

    # test showing
    # print(dr_wine_data)
    # print(len(dr_wine_data))

    # Now org: wine_data, after PCA: dr_wine_data, do K-Means
    classified_wine_data, sihouette_org = KMeans(k, wine_data)
    classified_dr_wine_data, sihouette_pca = KMeans(k, dr_wine_data)

    # turn into data frames
    df_classidfied_wine_data = pd.DataFrame(classified_wine_data)
    df_classidfied_dr_wine_data = pd.DataFrame(classified_dr_wine_data)

    # write into csv
    df_classidfied_dr_wine_data.to_csv('../output/PCA_classified_wine_data.csv')
    df_classidfied_wine_data.to_csv('../output/classified_wine_data.csv')

    # calculate rand
    rand_coef_org = calculate_rand(classified_wine_data, wine_class)
    rand_coef_pca = calculate_rand(classified_dr_wine_data, wine_class)

    # show rand and sihouette
    print('Original:     Sihouette: ' + str(sihouette_org) + '   Rand: ' + str(rand_coef_org))
    print('PCA:    :     Sihouette: ' + str(sihouette_pca) + '   Rand: ' + str(rand_coef_pca))

    # visualization - GLCM Graph
    """
    save = '../output/graph_org.pdf'
    plot_radar(classified_wine_data, save)
    save = '../output/graph_pca.pdf'
    plot_radar(classified_dr_wine_data, save)
    """

    return


if __name__ == '__main__':
    main()