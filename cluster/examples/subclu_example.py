import numpy as np
from itertools import chain
from subspaceclustering.cluster.subclu import subclu
from subspaceclustering.utils.sample_generator import generate_samples
from subspaceclustering.cluster.clusterquality import quality_measures
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import csv
import time


def cluster_vowel():
    # 528 x 10
    samples = read_csv("real_world_data/vowel")
    labels = samples[:, -1]
    true_clusters = {tuple(range(10)): []}
    for label in np.setdiff1d(labels, -1):
        true_clusters.get(tuple(range(10))).append(np.where(labels == label)[0])
    subclu_instance = subclu(samples[:, :-1], 0.8, 6)
    subclu_instance.process()
    found_clusters = subclu_instance.get_clusters()
    # print_clustering(found_clusters, subclu_instance.get_noise())
    evaluate_clustering(528, 10, true_clusters, found_clusters)


def cluster_diabetes():
    # 768 x 8
    samples = read_csv("real_world_data/diabetes")
    labels = samples[:, -1]
    true_clusters = {(0, 1, 2, 3, 4, 5, 6, 7): []}
    for label in np.setdiff1d(labels, -1):
        true_clusters.get((0, 1, 2, 3, 4, 5, 6, 7)).append(np.where(labels == label)[0])
    # print(true_clusters)
    subclu_instance = subclu(samples[:, :-1], 24, 15)
    subclu_instance.process()
    found_clusters = subclu_instance.get_clusters()
    # print_clustering(found_clusters, subclu_instance.get_noise())
    evaluate_clustering(768, 8, true_clusters, found_clusters)


def cluster_glass():
    # 214 x 9
    samples = read_csv("real_world_data/glass")
    labels = samples[:, -1]
    true_clusters = {tuple(range(9)): []}
    for label in np.setdiff1d(labels, -1):
        true_clusters.get(tuple(range(9))).append(np.where(labels == label)[0])
    subclu_instance = subclu(samples[:, :-1], 1, 3)
    subclu_instance.process()
    found_clusters = subclu_instance.get_clusters()
    # print_clustering(found_clusters, subclu_instance.get_noise())
    evaluate_clustering(214, 9, true_clusters, found_clusters)


def cluster_D05():
    """Reproduces Subclu-related values in table 1 in the thesis."""
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

    subclu_instance = subclu(samples, 25, 25)
    subclu_instance.process()
    found_clusters = subclu_instance.get_clusters()
    # uncomment the below line to print all found clusters
    # print_clustering(found_clusters, subclu_instance.get_noise())
    # uncomment the below line to print found clusters in the wanted subspaces
    # for subspace in list(true_clusters):
    # print(subspace, found_clusters.get(subspace, []))
    evaluate_clustering(1595, 5, true_clusters, found_clusters)


def cluster_D10():
    """Reproduces Subclu-related values in table 2 in the thesis."""
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
    subclu_instance = subclu(samples, 25, 20)
    subclu_instance.process()
    found_clusters = subclu_instance.get_clusters()
    # uncomment the below line to print all found clusters
    # print_clustering(found_clusters, subclu_instance.get_noise())
    # uncomment the below line to print found clusters in the wanted subspaces
    # for subspace in list(true_clusters):
    #    print(subspace, found_clusters.get(subspace, []))
    evaluate_clustering(1595, 10, true_clusters, found_clusters)


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
    subclu_instance = subclu(samples, 25, 10)
    start = time.perf_counter()
    subclu_instance.process()
    stop = time.perf_counter()
    # found_clusters = subclu_instance.get_clusters()
    # uncomment the below line to print all found clusters
    # print_clustering(found_clusters, subclu_instance.get_noise())
    # uncomment the below line to print found clusters in the wanted subspaces
    # for subspace in list(true_clusters):
    # print(subspace, found_clusters.get(subspace, []))
    # uncomment the below line to evaluate the clustering
    # evaluate_clustering(1595, 10, true_clusters, found_clusters)
    return stop - start


def cluster_S2500():
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
    subclu_instance = subclu(samples, 25, 15)
    start = time.perf_counter()
    subclu_instance.process()
    stop = time.perf_counter()
    # found_clusters = subclu_instance.get_clusters()
    # uncomment the below line to print all found clusters
    # print_clustering(found_clusters, subclu_instance.get_noise())
    # uncomment the below line to print found clusters in the wanted subspaces
    # for subspace in list(true_clusters):
    #  print(subspace, found_clusters.get(subspace, []))
    # uncomment the below line to evaluate the clustering
    # evaluate_clustering(2658, 10, true_clusters, found_clusters)
    return stop - start


def cluster_S3500():
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
    subclu_instance = subclu(samples, 25, 15)
    start = time.perf_counter()
    subclu_instance.process()
    stop = time.perf_counter()
    # found_clusters = subclu_instance.get_clusters()
    # uncomment the below line to print all found clusters
    # print_clustering(found_clusters, subclu_instance.get_noise())
    # uncomment the below line to print found clusters in the wanted subspaces
    # for subspace in list(true_clusters):
    # print(subspace, found_clusters.get(subspace, []))
    # uncomment the below line to evaluate the clustering
    # evaluate_clustering(3722, 10, true_clusters, found_clusters)
    return stop - start


def cluster_S4500():
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
    subclu_instance = subclu(samples, 27, 15)
    start = time.perf_counter()
    subclu_instance.process()
    stop = time.perf_counter()
    # found_clusters = subclu_instance.get_clusters()
    # uncomment the below line to print all found clusters
    # print_clustering(found_clusters, subclu_instance.get_noise())
    # uncomment the below line to print found clusters in the wanted subspaces
    # for subspace in list(true_clusters):
    #   print(subspace, found_clusters.get(subspace, []))
    # uncomment the below line to evaluate the clustering
    # evaluate_clustering(4785, 10, true_clusters, found_clusters)
    return stop - start


def cluster_S5500():
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
    subclu_instance = subclu(samples, 27, 15)
    start = time.perf_counter()
    subclu_instance.process()
    stop = time.perf_counter()
    # found_clusters = subclu_instance.get_clusters()
    # uncomment the below line to print all found clusters
    # print_clustering(found_clusters, subclu_instance.get_noise())
    # uncomment the below line to print found clusters in the wanted subspaces
    # for subspace in list(true_clusters):
    #  print(subspace, found_clusters.get(subspace, []))
    # uncomment the below line to evaluate the clustering
    # evaluate_clustering(5848, 10, true_clusters, found_clusters)
    return stop - start


def cluster_2d_100n_2sc():
    true_clusters = {}
    true_clusters[0] = [range(40)]
    true_clusters[1] = [range(50, 95)]
    samples, labels = generate_samples(
        100, 2, 1, 0, [[range(40), range(0, 1), 0.2], [range(50, 95), range(1, 2), 0.3]]
    )
    subclu_instance = subclu(samples, 0.3, 5)
    subclu_instance.process()
    found_clusters = subclu_instance.get_clusters()
    evaluate_clustering(100, 2, true_clusters, found_clusters)
    print_clustering(found_clusters, subclu_instance.get_noise())


def cluster_3d_250n_3sc():
    true_clusters = {
        (0, 2): [tuple(chain(range(50), range(70, 100))), range(200, 235)],
        1: [range(120, 190)],
    }
    samples, labels = generate_samples(
        250,
        3,
        1,
        1,
        [
            [tuple(chain(range(50), range(70, 100))), (0, 2), 0.3],
            [range(120, 190), range(1, 2), 0.3],
            [range(200, 235), (0, 2), 0.3],
        ],
    )
    subclu_instance = subclu(samples, 0.4, 6)
    subclu_instance.process()
    found_clusters = subclu_instance.get_clusters()
    evaluate_clustering(250, 3, true_clusters, found_clusters)
    print_clustering(found_clusters, subclu_instance.get_noise())


def cluster_4d_500n_4sc():
    true_clusters = {
        (0, 3): [range(0, 160, 2)],
        (1, 2, 3): [range(180, 300)],
        (0, 2): [range(320, 400)],
        2: [range(425, 475)],
    }
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
    subclu_instance = subclu(samples, 0.3, 6)
    subclu_instance.process()
    found_clusters = subclu_instance.get_clusters()
    evaluate_clustering(500, 4, true_clusters, found_clusters)
    print_clustering(found_clusters, subclu_instance.get_noise())


def cluster_8d_1000n_7sc():
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
            [range(800, 900), (0, 1), 0.3],
            [range(900, 950), (0, 3, 4, 6, 7), 0.3],
        ],
    )
    subclu_instance = subclu(samples, 0.6, 7)
    subclu_instance.process()
    found_clusters = subclu_instance.get_clusters()
    print_clustering(found_clusters, subclu_instance.get_noise())
    evaluate_clustering(1000, 8, true_clusters, found_clusters)
    for subspace in list(true_clusters):
        print(subspace, found_clusters.get(subspace, []))


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


def print_clustering(clusters, noise):
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


def plot_D05():
    """Reproduces figures 5,6 in the thesis."""
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
    pca = PCA(n_components=3)
    samples = pca.fit_transform(samples[:, :-1])
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(projection="3d")
    colours = [
        "yellow",
        "magenta",
        "yellowgreen",
        "c",
        "gray",
        "orange",
        "k",
        "salmon",
        "r",
        "b",
    ]
    for indicies in true_clusters.values():
        for cluster in indicies:
            ax.scatter(
                samples[cluster, 0],
                samples[cluster, 1],
                samples[cluster, 2],
                c=colours.pop(),
            )
    plt.show()


# cluster_vowel()
# cluster_diabetes()
# cluster_glass()
# cluster_D05()
# cluster_D10()
# plot_D05()
# cluster_2d_100n_2sc()
# cluster_3d_250n_3sc()
# cluster_4d_500n_4sc()
# cluster_8d_1000n_7sc()
