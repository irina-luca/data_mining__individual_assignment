import numpy as np
import pandas as pd
import math
import sys

def normalize_data(data):
    # -- Normalize data per col/feature of the given tuples -- #
    data_normalized = []
    for tuple in data.transpose():
        nan_values_tuple = tuple
        min_val_in_tuple = min(nan_values_tuple)
        max_val_in_tuple = max(nan_values_tuple)
        data_normalized.append([(value - min_val_in_tuple) / (max_val_in_tuple - min_val_in_tuple) for value in tuple])
    return np.array(data_normalized).transpose()

def get_delta(tuple_1, tuple_2, length):
    # -- Calculate Euclidean distance -- #
    delta = 0
    for i in range(0, length):
        if tuple_1[i] != tuple_2[i]:
            delta += pow((tuple_1[i] - tuple_2[i]), 2)
    return math.sqrt(delta)

def append_tuple_to_best_cluster(clusters, clusters_centroids, tuple_obj):
    best_delta = sys.maxint # find the dist to the closest cluster
    best_cluster_index = -1
    for i, cluster in enumerate(clusters):
        dist = get_delta(clusters_centroids[i], tuple_obj[:-1], len(tuple_obj[:-1]))
        if dist < best_delta: # check if distance to the actual cluster's centroid is closer and if so, store the cluster id and the distance to its centroid
            best_cluster_index = i
            best_delta = dist
    clusters[best_cluster_index].append(tuple_obj)

def update_centroid(cluster, cluster_centroid):
    new_cluster_centroid = np.zeros(len(cluster_centroid)) # size here means the number of features
    # -- Calculate new cluster centroid by averaging the features -- #
    for tuple_obj in cluster:
        for tuple_obj_col_index, tuple_obj_col_val in enumerate(tuple_obj[:-1]):
            new_cluster_centroid[tuple_obj_col_index] += float(tuple_obj_col_val)
    for i, col in enumerate(new_cluster_centroid):
        new_cluster_centroid[i] = col / len(cluster) * 1.0
    return new_cluster_centroid, cluster_centroid

def k_means(k, random_k, dataset): # random_k are the initially random means
    clusters = {c: [] for c in range(k)}
    clusters_centroids = {} # my centroids do not have the last column (in this case, the class)

    for i in range(0, k):
        clusters[i].append(random_k[i])
        clusters_centroids[i] = random_k[i][:-1] # append the centroid without label/class

    converged = False

    while not converged:
        # Assignment Step
        for tuple_obj in dataset:
            append_tuple_to_best_cluster(clusters, clusters_centroids, tuple_obj)
        # print clusters

        # Update Step
        converged = True
        for i, cluster in enumerate(clusters):
            # print clusters[i]
            new_cluster_centroid, cluster_centroid = update_centroid(clusters[i], clusters_centroids[i])
            if not (new_cluster_centroid - cluster_centroid < 0.000000000000000000001).all():
                converged = False
            clusters_centroids[i] = new_cluster_centroid
            # print "____________________________________________________________"
        if not converged:
            clusters = {c: [] for c in range(k)}

    return clusters


def main(data, k):
    data_label_col = data[:, data.shape[1]-1:data.shape[1]]
    data_unlabeled = data[:,:-1]

    # -- Normalize dataset --#
    data_unlabeled_normalized = normalize_data(data_unlabeled)
    data_labeled_normalized = np.hstack((data_unlabeled_normalized, data_label_col))

    # print 'data_labeled_normalized', data_labeled_normalized

    # -- Pick k && random initial centroids -- #
    random_k = data_labeled_normalized[np.random.randint(0, data_labeled_normalized.shape[0], k)]

    # -- Run k_means -- #
    clusters = k_means(k, random_k, data_labeled_normalized)

    # -- Print results -- #
    c1 = [] # only for the case I chose
    c2 = [] # only for the case I chose
    g = open('kMeans-clusters.txt', 'w')
    for c in clusters:
        # print "___________________________________________________________________"
        g.write("___________________________________________________________________\n")
        for tuple in clusters[c]:
            if c == 0:
                c1.append(tuple[len(tuple)-1])
            else:
                c2.append(tuple[len(tuple)-1])

            # print tuple[len(tuple)-1]
            g.write(tuple[len(tuple)-1] + "\n")
    g.close()

    print "Cluster_1: "
    print c1
    print
    print "Cluster_2: "
    print c2

    return clusters

# data_file_name = "iris.csv"
#
# # -- Seed --#
# np.random.seed(0)  # seed, for testing
#
# # -- Read dataset --#
# data = pd.read_csv(data_file_name, sep=',').as_matrix()
# print data
# main(data, 3)  # Chose k = 3
