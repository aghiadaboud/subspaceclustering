import numpy as np
from itertools import chain
import csv
from subspaceclustering.cluster.fires import *
from subspaceclustering.utils.sample_generator import generate_samples
from subspaceclustering.cluster.clusterquality import quality_measures
import urllib.request
import time


def cluster_vowel():
    # 528 x 10
    samples = read_csv("real_world_data/vowel")
    labels = samples[:, -1]
    true_clusters = {tuple(range(10)): []}
    for label in np.setdiff1d(labels, -1):
        true_clusters.get(tuple(range(10))).append(np.where(labels == label)[0])
    clustering_method = Clustering_By_dbscan(0.05, 7)
    fires_instance = fires(samples[:, :-1], 1, 2, 1, clustering_method)
    fires_instance.process()
    found_clusters = fires_instance.get_clusters()
    # print_clustering(fires_instance)
    # for subspace in list(true_clusters):
    # print(subspace, found_clusters.get(subspace, []))
    evaluate_clustering(528, 10, true_clusters, found_clusters)


def cluster_diabetes():
    # 768 x 8
    samples = read_csv("real_world_data/diabetes")
    labels = samples[:, -1]
    true_clusters = {(0, 1, 2, 3, 4, 5, 6, 7): []}
    for label in np.setdiff1d(labels, -1):
        true_clusters.get((0, 1, 2, 3, 4, 5, 6, 7)).append(np.where(labels == label)[0])
    clustering_method = Clustering_By_dbscan(0.09, 12)
    fires_instance = fires(samples[:, :-1], 2, 3, 1, clustering_method)
    fires_instance.process()
    found_clusters = fires_instance.get_clusters()
    # print_clustering(fires_instance)
    # for subspace in list(true_clusters):
    # print(subspace, found_clusters.get(subspace, []))
    evaluate_clustering(768, 8, true_clusters, found_clusters)


def cluster_glass():
    # 214 x 9
    samples = read_csv("real_world_data/glass")
    labels = samples[:, -1]
    true_clusters = {tuple(range(9)): []}
    for label in np.setdiff1d(labels, -1):
        true_clusters.get(tuple(range(9))).append(np.where(labels == label)[0])
    clustering_method = Clustering_By_dbscan(0.09, 9)
    fires_instance = fires(samples[:, :-1], 1, 2, 1, clustering_method)
    fires_instance.process()
    found_clusters = fires_instance.get_clusters()
    # print_clustering(fires_instance)
    # for subspace in list(true_clusters):
    # print(subspace, found_clusters.get(subspace, []))
    evaluate_clustering(214, 9, true_clusters, found_clusters)


def cluster_D05(print_clusters = False, print_evaluation = True):
    """Reproduces Fires-related values in table 1 in the thesis."""
    true_clusters = {
        (0, 1, 4): [range(0, 302)],
        (0, 1, 3, 4): [range(0, 156), range(1008, 1158), range(1159, 1309)],
        (1, 2, 3): [range(303, 604)],
        (1, 2, 3, 4): [range(303, 459)],
        (0, 2, 3): [range(605, 755), range(1310, 1460)],
        (1, 2, 4): [tuple(chain(range(605, 655), range(756, 854)))],
        (0, 2, 3, 4): [range(855, 1007)],
    }
    samples = read_csv("db_dimensionality_scale/D05")
    samples = samples[:, :-1]
    clustering_method = Clustering_By_dbscan(0.8, 8)
    fires_instance = fires(samples, 2, 3, 2, clustering_method)
    fires_instance.process()
    found_clusters = fires_instance.get_clusters()
    # uncomment the below line to print all clustering information
    # print_clustering(fires_instance)
    if print_clusters:
        for subspace in list(true_clusters):
          print(subspace, found_clusters.get(subspace, []))
    if print_evaluation:
        evaluate_clustering(1595, 5, true_clusters, found_clusters)


def cluster_D10(print_clusters = False, print_evaluation = True):
    """Reproduces Fires-related values in table 2 in the thesis."""
    true_clusters = {
        (1, 2, 3, 5, 9): [range(0, 300)],
        (0, 1, 2, 3, 5, 6, 7, 9): [range(0, 150)],
        (0, 2, 4, 6, 9): [range(301, 602)],
        (0, 1, 2, 3, 4, 5, 6, 9): [range(301, 454)],
        (1, 2, 4, 5, 7, 8): [range(603, 753)],
        (0, 2, 3, 4, 5, 9): [tuple(chain(range(603, 653), range(754, 853)))],
        (1, 2, 3, 4, 5, 6, 7, 9): [range(854, 1003)],
        (0, 2, 3, 4, 5, 6, 8, 9): [range(1004, 1154)],
        (0, 1, 2, 3, 6, 7, 8): [range(1155, 1305)],
        (0, 3, 4, 6, 7, 8): [range(1306, 1457)],
    }
    samples = read_csv("db_dimensionality_scale/D10")
    samples = samples[:, :-1]
    clustering_method = Clustering_By_dbscan(0.82, 8)
    fires_instance = fires(samples, 2, 3, 2, clustering_method)
    fires_instance.process()
    found_clusters = fires_instance.get_clusters()
    # uncomment the below line to print all clustering information
    # print_clustering(fires_instance)
    if print_clusters:
        for subspace in list(true_clusters):
          print(subspace, found_clusters.get(subspace, []))
    if print_evaluation:
      evaluate_clustering(1595, 10, true_clusters, found_clusters)


def cluster_D15():
    true_clusters = {
        (1, 3, 6, 8, 10, 12): [range(0, 302)],
        (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13): [range(0, 152)],
        (1, 2, 4, 5, 8, 9, 11, 12, 14): [range(301, 603)],
        (1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 14): [range(302, 451)],
        (0, 1, 2, 4, 6, 10, 12, 13, 14): [range(610, 760)],
        (1, 3, 4, 8, 9, 10, 11, 12, 13): [
            tuple(chain(range(610, 660), range(760, 861)))
        ],
        (0, 1, 2, 4, 5, 6, 9, 10, 11, 12, 13, 14): [range(861, 1011)],
        (0, 1, 3, 4, 5, 7, 9, 10, 11, 12, 13, 14): [range(1011, 1163)],
        (0, 1, 2, 3, 4, 5, 7, 8, 10, 12, 13): [range(1163, 1316)],
        (1, 3, 4, 5, 6, 9, 10, 13, 14): [range(1316, 1468)],
    }
    samples = read_csv("db_dimensionality_scale/D15")
    samples = samples[:, :-1]
    clustering_method = Clustering_By_dbscan(0.8, 8)
    fires_instance = fires(samples, 2, 3, 2, clustering_method)
    fires_instance.process()
    found_clusters = fires_instance.get_clusters()
    # print_clustering(fires_instance)
    #for subspace in list(true_clusters):
     #   print(subspace, found_clusters.get(subspace, []))
    evaluate_clustering(1599, 15, true_clusters, found_clusters)


def cluster_D20():
    true_clusters = {
        (1, 2, 3, 4, 6, 7, 8, 9, 14, 17): [range(0, 300)],
        (0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 14, 15, 16, 17, 18): [range(0, 150)],
        (1, 2, 3, 4, 5, 6, 12, 17, 18, 19): [range(300, 600)],
        (0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 17, 18, 19): [range(300, 452)],
        (1, 2, 3, 4, 5, 7, 9, 11, 13, 14, 16, 18): [range(600, 750)],
        (1, 2, 3, 4, 5, 6, 7, 9, 13, 14, 17, 19): [
            tuple(chain(range(610, 654), range(750, 847)))
        ],
        (2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19): [range(847, 998)],
        (1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 16, 17, 18, 19): [range(998, 1148)],
        (0, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 17, 18, 19): [range(1148, 1299)],
        (1, 4, 7, 9, 10, 11, 12, 13, 14, 15, 18, 19): [range(1299, 1453)],
    }
    samples = read_csv("db_dimensionality_scale/D20")
    samples = samples[:, :-1]
    clustering_method = Clustering_By_dbscan(1, 8)
    fires_instance = fires(samples, 2, 3, 2, clustering_method)
    fires_instance.process()
    found_clusters = fires_instance.get_clusters()
    # print_clustering(fires_instance)
    # for subspace in list(true_clusters):
    #  print(subspace, found_clusters.get(subspace, []))
    evaluate_clustering(1595, 20, true_clusters, found_clusters)


def cluster_D25():
    true_clusters = {
        (0, 3, 4, 8, 11, 13, 14, 15, 17, 19, 20, 22, 24): [range(0, 302)],
        (0, 2, 3, 4, 5, 7, 8, 9, 11, 12, 13, 14, 15, 17, 19, 20, 21, 22, 23, 24): [
            range(0, 151)
        ],
        (0, 2, 4, 7, 9, 10, 11, 14, 15, 16, 17, 18, 20): [range(302, 602)],
        (0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 14, 15, 16, 17, 18, 20, 21, 22): [
            range(302, 454)
        ],
        (0, 1, 4, 5, 6, 9, 11, 14, 15, 17, 18, 19, 20, 23, 24): [range(602, 753)],
        (0, 1, 2, 3, 9, 10, 11, 13, 14, 15, 17, 19, 20, 21, 24): [
            tuple(chain(range(602, 654), range(753, 851)))
        ],
        (0, 1, 2, 4, 5, 6, 7, 8, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 23, 24): [
            range(851, 1001)
        ],
        (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 15, 17, 18, 19, 20, 23, 24): [
            range(1001, 1154)
        ],
        (0, 1, 2, 3, 5, 6, 7, 9, 10, 13, 14, 16, 18, 20, 21, 22, 23, 24): [
            range(1154, 1303)
        ],
        (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17): [
            range(1303, 1454)
        ],
    }
    samples = read_csv("db_dimensionality_scale/D25")
    samples = samples[:, :-1]
    clustering_method = Clustering_By_dbscan(1, 10)
    fires_instance = fires(samples, 2, 3, 2, clustering_method)
    fires_instance.process()
    found_clusters = fires_instance.get_clusters()
    # print_clustering(fires_instance)
    # for subspace in list(true_clusters):
    #  print(subspace, found_clusters.get(subspace, []))
    evaluate_clustering(1595, 25, true_clusters, found_clusters)


def cluster_D50():
    true_clusters = {
        (
            2,
            5,
            7,
            12,
            14,
            15,
            16,
            17,
            18,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            31,
            32,
            33,
            37,
            38,
            43,
        ): [range(0, 300)],
        (
            0,
            1,
            2,
            3,
            5,
            7,
            8,
            9,
            10,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
            33,
            34,
            35,
            37,
            38,
            41,
            43,
            45,
            46,
            47,
            49,
        ): [range(0, 151)],
        (
            1,
            2,
            4,
            7,
            10,
            12,
            13,
            14,
            23,
            24,
            26,
            28,
            29,
            30,
            32,
            34,
            35,
            37,
            38,
            39,
            40,
            44,
            45,
            47,
            48,
        ): [range(301, 601)],
        (
            1,
            2,
            3,
            4,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            16,
            18,
            20,
            22,
            23,
            24,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
            34,
            35,
            36,
            37,
            38,
            39,
            40,
            41,
            44,
            45,
            46,
            47,
            48,
            49,
        ): [range(300, 448)],
        (
            2,
            3,
            4,
            5,
            6,
            8,
            9,
            10,
            11,
            16,
            18,
            22,
            24,
            26,
            27,
            29,
            30,
            32,
            34,
            35,
            36,
            38,
            39,
            40,
            43,
            44,
            45,
            46,
            47,
            49,
        ): [range(606, 759)],
        (
            0,
            1,
            2,
            3,
            4,
            5,
            7,
            9,
            11,
            12,
            15,
            17,
            19,
            20,
            21,
            22,
            24,
            26,
            28,
            29,
            33,
            34,
            36,
            38,
            39,
            41,
            43,
            47,
            48,
            49,
        ): [tuple(chain(range(606, 657), range(759, 861)))],
        (
            0,
            1,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            30,
            31,
            34,
            35,
            36,
            37,
            38,
            40,
            42,
            43,
            44,
            46,
            47,
            48,
            49,
        ): [range(861, 1011)],
        (
            0,
            2,
            4,
            5,
            7,
            8,
            10,
            11,
            12,
            13,
            14,
            15,
            18,
            20,
            21,
            22,
            23,
            24,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
            33,
            34,
            35,
            37,
            38,
            39,
            40,
            42,
            43,
            44,
            45,
            46,
            47,
            48,
            49,
        ): [range(1011, 1161)],
        (
            0,
            1,
            2,
            4,
            5,
            6,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            17,
            18,
            19,
            21,
            22,
            24,
            26,
            27,
            28,
            29,
            30,
            31,
            36,
            37,
            38,
            39,
            42,
            43,
            44,
            46,
            48,
            49,
        ): [range(1161, 1314)],
        (
            0,
            1,
            2,
            6,
            7,
            8,
            9,
            10,
            11,
            13,
            16,
            23,
            24,
            25,
            27,
            29,
            31,
            32,
            33,
            37,
            38,
            39,
            40,
            41,
            42,
            43,
            44,
            46,
            48,
            49,
        ): [range(1314, 1466)],
    }
    samples = read_csv("db_dimensionality_scale/D50")
    samples = samples[:, :-1]
    clustering_method = Clustering_By_dbscan(1, 10)
    fires_instance = fires(samples, 2, 3, 2, clustering_method)
    fires_instance.process()
    found_clusters = fires_instance.get_clusters()
    # print_clustering(fires_instance)
    # for subspace in list(true_clusters):
    # print(subspace, found_clusters.get(subspace, []))
    evaluate_clustering(1596, 50, true_clusters, found_clusters)


def cluster_S1500():
    # We will consider only the first 10 dimensions.
    true_clusters = {
        (2, 3, 9): [range(0, 300)],
        (1, 2, 3, 5, 6, 7, 8, 9): [range(0, 152)],
        (1, 3, 4, 5, 7, 9): [range(300, 452)],
        (0, 1, 3, 5, 7, 9): [tuple(chain(range(300, 349), range(452, 554)))],
        (0, 1, 2, 3, 4, 5, 6, 7, 8): [range(554, 706)],
        (0, 2, 3, 4, 5, 6, 7, 8): [range(706, 859)],
        (0, 1, 2, 3, 4, 8, 9): [range(859, 1010)],
        (0, 1, 3, 5, 6, 7, 8): [range(1010, 1159)],
        (0, 1, 2, 4, 5, 6, 7, 8, 9): [range(1159, 1312)],
        (0, 1, 2, 3, 4, 5, 7, 9): [range(1312, 1462)],
    }
    samples = read_csv("db_size_scale/S1500")
    samples = samples[:, :10]
    clustering_method = Clustering_By_dbscan(0.8, 9)
    fires_instance = fires(samples, 2, 3, 2, clustering_method)
    start = time.perf_counter()
    fires_instance.process()
    stop = time.perf_counter()
    # found_clusters = fires_instance.get_clusters()
    # uncomment the below line to print all clustering information
    # print_clustering(fires_instance)
    # uncomment the below line to print found clusters in the wanted subspaces
    # for subspace in list(true_clusters):
    # print(subspace, found_clusters.get(subspace, []))
    # uncomment the below line to evaluate the clustering
    # evaluate_clustering(1595, 10, true_clusters, found_clusters)
    return stop - start


def cluster_S2500():
    # We will consider only the first 10 dimensions.
    true_clusters = {
        (0, 2, 7, 9): [range(0, 502)],
        (0, 1, 2, 4, 6, 7, 8, 9): [range(0, 253)],
        (0, 3, 7, 8, 9): [range(502, 752)],
        (0, 1, 2, 3, 6, 7, 9): [tuple(chain(range(502, 587), range(752, 917)))],
        (0, 1, 2, 3, 4, 5, 6, 8): [range(917, 1169)],
        (0, 1, 3, 4, 6, 7, 9): [range(1169, 1417)],
        (0, 2, 3, 4, 6, 7, 8, 9): [range(1417, 1667)],
        (2, 3, 4, 5, 6, 7, 8, 9): [range(1667, 1920)],
        (1, 4, 6, 7, 8, 9): [range(1920, 2173)],
        (0, 1, 4, 5, 6): [range(2173, 2427)],
    }
    samples = read_csv("db_size_scale/S2500")
    samples = samples[:, :10]
    clustering_method = Clustering_By_dbscan(0.7, 11)
    fires_instance = fires(samples, 2, 3, 2, clustering_method)
    start = time.perf_counter()
    fires_instance.process()
    stop = time.perf_counter()
    # found_clusters = fires_instance.get_clusters()
    # uncomment the below line to print all clustering information
    # print_clustering(fires_instance)
    # uncomment the below line to print found clusters in the wanted subspaces
    # for subspace in list(true_clusters):
    # print(subspace, found_clusters.get(subspace, []))
    # uncomment the below line to evaluate the clustering
    # evaluate_clustering(2658, 10, true_clusters, found_clusters)
    return stop - start


def cluster_S3500():
    # We will consider only the first 10 dimensions.
    true_clusters = {
        (0, 1, 2, 8, 9): [range(0, 704)],
        (0, 1, 2, 4, 5, 6, 8, 9): [range(0, 348)],
        (0, 1, 2, 3, 4, 7, 8): [range(710, 1062)],
        (2, 4, 5, 6, 7): [tuple(chain(range(710, 827), range(1062, 1296)))],
        (0, 1, 2, 3, 4, 6, 7, 8, 9): [range(1296, 1645)],
        (0, 1, 2, 3, 4, 5, 6, 8): [range(1645, 1996)],
        (0, 1, 3, 4, 5, 6, 7, 8, 9): [range(1996, 2344)],
        (0, 2, 3, 4, 5, 6, 7, 8): [range(2344, 2693)],
        (0, 1, 2, 4, 5, 7): [range(2693, 3043)],
        (0, 2, 4, 5, 6, 7): [range(3043, 3393)],
    }
    samples = read_csv("db_size_scale/S3500")
    samples = samples[:, :10]
    clustering_method = Clustering_By_dbscan(0.72, 12)
    fires_instance = fires(samples, 2, 3, 2, clustering_method)
    start = time.perf_counter()
    fires_instance.process()
    stop = time.perf_counter()
    # found_clusters = fires_instance.get_clusters()
    # uncomment the below line to print all clustering information
    # print_clustering(fires_instance)
    # uncomment the below line to print found clusters in the wanted subspaces
    # for subspace in list(true_clusters):
    # print(subspace, found_clusters.get(subspace, []))
    # uncomment the below line to evaluate the clustering
    # evaluate_clustering(3722, 10, true_clusters, found_clusters)
    return stop - start


def cluster_S4500():
    # We will consider only the first 10 dimensions.
    true_clusters = {
        (5, 6, 8): [range(0, 902)],
        (0, 2, 3, 4, 5, 6, 8, 9): [range(0, 450)],
        (0, 1, 3, 5, 9): [range(902, 1352)],
        (0, 3, 4, 6, 7, 8, 9): [tuple(chain(range(902, 1051), range(1352, 1651)))],
        (0, 1, 2, 3, 4, 5, 8, 9): [range(1651, 2102)],
        (1, 3, 4, 6, 7, 8, 9): [range(2102, 2553)],
        (0, 1, 2, 3, 4, 6, 7, 8): [range(2553, 3006)],
        (0, 1, 2, 3, 4, 5, 6, 7, 8, 9): [range(3006, 3458)],
        (1, 2, 3, 5, 6, 7, 9): [range(3458, 3910)],
        (0, 1, 2, 3, 5, 7, 8): [range(3910, 4363)],
    }
    samples = read_csv("db_size_scale/S4500")
    samples = samples[:, :10]
    clustering_method = Clustering_By_dbscan(0.8, 18)
    fires_instance = fires(samples, 3, 4, 1, clustering_method)
    start = time.perf_counter()
    fires_instance.process()
    stop = time.perf_counter()
    # found_clusters = fires_instance.get_clusters()
    # uncomment the below line to print all clustering information
    # print_clustering(fires_instance)
    # uncomment the below line to print found clusters in the wanted subspaces
    # for subspace in list(true_clusters):
    # print(subspace, found_clusters.get(subspace, []))
    # uncomment the below line to evaluate the clustering
    # evaluate_clustering(4785, 10, true_clusters, found_clusters)
    return stop - start


def cluster_S5500():
    # We will consider only the first 10 dimensions.
    true_clusters = {
        (1, 2, 3, 4, 9): [range(0, 1101)],
        (1, 2, 3, 4, 5, 6, 8, 9): [range(0, 572)],
        (1, 3, 5, 6, 7, 8, 9): [range(1101, 1652)],
        (0, 1, 4, 7, 8, 9): [tuple(chain(range(1101, 1289), range(1652, 2016)))],
        (0, 1, 2, 3, 5, 6, 7, 9): [range(2016, 2566)],
        (0, 1, 2, 3, 4, 5, 7, 8): [range(2566, 3118)],
        (0, 2, 3, 4, 5, 6, 8, 9): [range(3118, 3667)],
        (0, 1, 2, 3, 6, 7, 8): [range(3667, 4220)],
        (0, 2, 3, 4, 7, 8, 9): [range(4220, 4772)],
        (1, 2, 4, 6, 7): [range(4772, 5325)],
    }
    samples = read_csv("db_size_scale/S5500")
    samples = samples[:, :10]
    clustering_method = Clustering_By_dbscan(0.7, 25)
    fires_instance = fires(samples, 2, 3, 2, clustering_method)
    start = time.perf_counter()
    fires_instance.process()
    stop = time.perf_counter()
    # found_clusters = fires_instance.get_clusters()
    # uncomment the below line to print all clustering information
    # print_clustering(fires_instance)
    # uncomment the below line to print found clusters in the wanted subspaces
    # for subspace in list(true_clusters):
    # print(subspace, found_clusters.get(subspace, []))
    # uncomment the below line to evaluate the clustering
    # evaluate_clustering(5848, 10, true_clusters, found_clusters)
    return stop - start


def cluster_2d_100n_2sc():
    samples, labels = generate_samples(
        100, 2, 1, 0, [[range(40), range(0, 1), 0.2], [range(50, 95), range(1, 2), 0.3]]
    )
    clustering_method = Clustering_By_dbscan(0.3, 5)
    fires_instance = fires(samples, 1, 1, 2, clustering_method)
    fires_instance.process()
    print_clustering(fires_instance)


def cluster_3d_250n_3sc():
    samples, labels = generate_samples(
        250,
        3,
        1.5,
        0,
        [
            [tuple(chain(range(50), range(70, 100))), (0, 2), 0.3],
            [range(120, 190), range(1, 2), 0.3],
            [range(200, 235), (0, 2), 0.3],
        ],
    )
    clustering_method = Clustering_By_dbscan(0.3, 6)
    fires_instance = fires(samples, 1, 1, 1, clustering_method)
    fires_instance.process()
    print_clustering(fires_instance)


def cluster_4d_500n_4sc():
    samples, labels = generate_samples(
        500,
        4,
        1,
        0,
        [
            [range(0, 160, 2), (0, 3), 0.3],
            [range(180, 300), (1, 2, 3), 0.3],
            [range(320, 400), (0, 2), 0.4],
            [range(425, 475), range(2, 3), 0.4],
        ],
    )
    clustering_method = Clustering_By_dbscan(0.2, 6)
    fires_instance = fires(samples, 1, 1, 1, clustering_method)
    fires_instance.process()
    print_clustering(fires_instance)


def cluster_8d_1000n_7sc(mu, k, minClu):
    """Can reproduce figure 1,2,3 in the thesis."""
    true_clusters = {
        (1, 2, 3): [range(250)],
        (5, 6, 7): [range(200, 340)],
        (1, 2): [range(290, 390)],
        (4, 5, 6, 7): [range(380, 510)],
        (2, 3, 4, 5): [range(670, 780)],
        (0, 1): [range(800, 900)],
        (0, 3, 4, 6, 7): [range(900, 950)],
    }
    samples, labels = generate_samples(
        1000,
        8,
        1.5,
        2,
        [
            [range(250), (1, 2, 3), 0.25],
            [range(200, 340), (5, 6, 7), 0.25],
            [range(290, 390), (1, 2), 0.3],
            [range(380, 510), range(4, 8), 0.25],
            [range(670, 780), range(2, 6), 0.3],
            [range(800, 890), (0, 1), 0.3],
            [range(890, 950), (0, 3, 4, 6, 7), 0.3],
        ],
    )
    # to shuffle the columns of the dataset
    # np.random.shuffle(samples.T)
    clustering_method = Clustering_By_dbscan(0.2, 6)
    fires_instance = fires(samples, mu, k, minClu, clustering_method)
    fires_instance.process()
    print('Base-clusters:')
    for (
        cluster_index,
        dimension,
    ) in fires_instance.get_baseCluster_to_dimension().items():
        cluster = fires_instance.get_pruned_C1()[cluster_index]
        print(
            "cluster_index ",
            cluster_index,
            "dim ",
            dimension,
            "cluster_size ",
            len(cluster),
        )
        print(sorted(cluster), "\n")
    print("k most similar clusters", "\n", fires_instance.get_k_most_similar_clusters())
    print("best merge candidates", "\n", fires_instance.get_best_merge_candidates())
    print("best merge clusters", "\n", fires_instance.get_best_merge_clusters())
    print(
        "pruned subspace cluster approximations",
        "\n",
        fires_instance.get_subspace_cluster_approximations(),
        "\n",
    )
    found_clusters = fires_instance.get_clusters()
    print('Found clusters(only in the desired subspaces):')
    for subspace in list(true_clusters):
        print(subspace, found_clusters.get(subspace, []))


def cluster_16d_2000n_6sc():
    samples, labels = generate_samples(
        2000,
        16,
        2,
        0,
        [
            [range(150), range(7), 0.2],
            [tuple(chain(range(50), range(200, 250))), range(11, 16), 0.2],
            [range(350, 450), range(7, 11), 0.2],
            [range(700, 780), range(9), 0.2],
            [range(760, 860), (13, 14, 15), 0.2],
            [range(1500, 1600), range(2, 16), 0.2],
        ],
    )
    clustering_method = Clustering_By_dbscan(0.07, 6)
    fires_instance = fires(samples, 3, 4, 3, clustering_method)
    fires_instance.process()
    found_clusters = fires_instance.get_clusters()
    print_clustering(fires_instance)


def read_csv(filename):
    samples = None
    with open("subspaceclustering/samples/" + filename + ".csv", newline="") as csvfile:
        reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
        samples = np.array(list(reader)).astype(float)
    return samples


def save_csv(samples, labels, filename):
    with open(
        "subspaceclustering/samples/" + filename + ".csv", "w", newline=""
    ) as csvfile:
        writer = csv.writer(csvfile)
        for row in samples:
            writer.writerow(row)

    with open(
        "subspaceclustering/samples/" + filename + "_labels.csv", "w", newline=""
    ) as csvfile:
        writer = csv.writer(csvfile)
        for row in labels:
            writer.writerow(row)


def download_file():
    # downloads the file over http in the directory of the file being excuted
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/david-c-hunn/edu.uwb.opensubspace/master/edu.uwb.opensubspace/Databases/synth_dimscale/D05.arff",
        "samples.csv",
    )


def print_clustering(fires_instance):
    for (
        cluster_index,
        dimension,
    ) in fires_instance.get_baseCluster_to_dimension().items():
        cluster = fires_instance.get_pruned_C1()[cluster_index]
        print(
            "cluster_index ",
            cluster_index,
            "dim ",
            dimension,
            "cluster_size ",
            len(cluster),
        )
        print(sorted(cluster), "\n")
    print("k most similar clusters", "\n", fires_instance.get_k_most_similar_clusters())
    print("best merge candidates", "\n", fires_instance.get_best_merge_candidates())
    print("best merge clusters", "\n", fires_instance.get_best_merge_clusters())
    print(
        "pruned subspace cluster approximations",
        "\n",
        fires_instance.get_subspace_cluster_approximations(),
        "\n",
    )
    clusters = fires_instance.get_clusters()
    for subspace, list_of_clusters in clusters.items():
        print(subspace, list(map(sorted, list_of_clusters)), "\n")


def evaluate_clustering(db_size, db_dimensionality, true_clusters, found_clusters):
    print(
        "F1 Recall %.2f"
        % quality_measures.overall_f1_recall(true_clusters, found_clusters)
    )
    print(
        "F1 Precision %.2f"
        % quality_measures.overall_f1_precision(true_clusters, found_clusters)
    )
    print(
        "F1 Merge %.2f"
        % quality_measures.overall_f1_merge(true_clusters, found_clusters)
    )
    print(
        "RNIA %.2f"
        % quality_measures.overall_rnia(
            db_size, db_dimensionality, true_clusters, found_clusters
        )
    )
    print(
        "CE %.2f"
        % quality_measures.overall_ce(
            db_size, db_dimensionality, true_clusters, found_clusters
        )
    )
    print("E4SC %.2f" % quality_measures.e4sc(true_clusters, found_clusters), "\n")


# cluster_vowel()
# cluster_diabetes()
# cluster_glass()
# cluster_D05()
# cluster_D10()
# cluster_2d_100n_2sc()
# cluster_3d_250n_3sc()
# cluster_4d_500n_4sc()
# cluster_8d_1000n_7sc(3,4,1)
# cluster_16d_2000n_6sc()
