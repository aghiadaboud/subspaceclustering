import numpy as np
from subspaceclustering.cluster.subclu import subclu
from subspaceclustering.cluster.fires import *
from subspaceclustering.utils.sample_generator import generate_samples
from subspaceclustering.cluster.examples import fires_example
from subspaceclustering.cluster.examples import subclu_example
import matplotlib.pyplot as plt
import time
import csv


def compare_clustering_quality():
    """Reproduces table 3 in the thesis."""
    print("Clustering the Vowel dataset by SUBCLU:")
    subclu_example.cluster_vowel()
    print("Clustering the Diabetes dataset by SUBCLU:")
    subclu_example.cluster_diabetes()
    print("Clustering the Glass dataset by SUBCLU:")
    subclu_example.cluster_glass()
    print("Clustering D05 by SUBCLU:")
    subclu_example.cluster_D05()
    print("Clustering D10 by SUBCLU:")
    subclu_example.cluster_D10()
    print("Clustering the Vowel dataset by FIRES:")
    fires_example.cluster_vowel()
    print("Clustering the Diabetes dataset by FIRES:")
    fires_example.cluster_diabetes()
    print("Clustering the Glass dataset by FIRES:")
    fires_example.cluster_glass()
    print("Clustering D05 by FIRES:")
    fires_example.cluster_D05()
    print("Clustering D10 by FIRES:")
    fires_example.cluster_D10()


def cluster_10d_xn_2sc():
    """Reproduces figure 10 in the thesis."""
    size = list(range(1500, 5501, 1000))
    subclu_runtime = []
    fires_runtime = []
    fires_dbscan_params = [(0.1, 8), (0.1, 8), (0.1, 8), (0.07, 5), (0.06, 5)]
    for i, n in enumerate(size):
        samples, labels = generate_samples(
            n,
            10,
            3,
            1,
            [[range(400, 550), range(4, 8), 0.2], [range(700, 850), range(7), 0.15]],
        )
        subclu_instance = subclu(samples, 0.37, 10)
        start_subclu = time.perf_counter()
        subclu_instance.process()
        stop_subclu = time.perf_counter()
        subclu_runtime.append(stop_subclu - start_subclu)
        print("Clustering 10d, " + str(n) + " n, 2 sc by SUBCLU: done")
        clustering_method = Clustering_By_dbscan(
            fires_dbscan_params[i][0], fires_dbscan_params[i][1]
        )
        fires_instance = fires(samples, 3, 4, 1, clustering_method)
        start_fires = time.perf_counter()
        fires_instance.process()
        stop_fires = time.perf_counter()
        fires_runtime.append(stop_fires - start_fires)
        print("Clustering 10d, " + str(n) + " n, 2 sc by FIRES: done")
    plt.figure(figsize=(8, 4))
    #plt.rcParams.update({"font.size": 26})
    plt.plot(size, subclu_runtime, label="SUBCLU")
    plt.plot(size, fires_runtime, label="FIRES")
    plt.xlabel("Number of points")
    plt.ylabel("Runtime in seconds")
    plt.legend(loc="center right")
    plt.savefig("subspaceclustering/images/cluster_10d_xn_2sc.pdf")
    #plt.show()


def cluster_10d_xn_10sc():
    """Reproduces figure 9 in the thesis."""
    # We will consider only the first 10 dimensions. 20 is too much for Subclu.
    datasets = [1500, 2500, 3500, 4500, 5500]
    subclu_runtime = []
    fires_runtime = []
    subclu_runtime.append(subclu_example.cluster_S1500())
    print("Clustering S1500 by SUBCLU: done")
    subclu_runtime.append(subclu_example.cluster_S2500())
    print("Clustering S2500 by SUBCLU: done")
    subclu_runtime.append(subclu_example.cluster_S3500())
    print("Clustering S3500 by SUBCLU: done")
    subclu_runtime.append(subclu_example.cluster_S4500())
    print("Clustering S4500 by SUBCLU: done")
    subclu_runtime.append(subclu_example.cluster_S5500())
    print("Clustering S5500 by SUBCLU: done")
    fires_runtime.append(fires_example.cluster_S1500())
    print("Clustering S1500 by FIRES: done")
    fires_runtime.append(fires_example.cluster_S2500())
    print("Clustering S2500 by FIRES: done")
    fires_runtime.append(fires_example.cluster_S3500())
    print("Clustering S3500 by FIRES: done")
    fires_runtime.append(fires_example.cluster_S4500())
    print("Clustering S4500 by FIRES: done")
    fires_runtime.append(fires_example.cluster_S5500())
    print("Clustering S5500 by FIRES: done")
    plt.figure(figsize=(8, 4))
    #plt.rcParams.update({"font.size": 26})
    plt.plot(datasets, subclu_runtime, label="SUBCLU")
    plt.plot(datasets, fires_runtime, label="FIRES")
    plt.xlabel("Number of points")
    plt.ylabel("Runtime in seconds")
    plt.legend(loc="upper left")
    plt.savefig("subspaceclustering/images/cluster_10d_xn_10sc.pdf")
    #plt.show()


def cluster_xd_1000n_1sc():
    """Reproduces figure 8 in the thesis."""
    dimensionalities = [10, 15, 20, 25, 50]
    subclu_runtime = []
    fires_runtime = []
    fires_dbscan_params = [(0.1, 15), (0.07, 15), (0.07, 10), (0.08, 15), (0.08, 10)]
    for i, d in enumerate(dimensionalities):
        samples, labels = generate_samples(
            1600, d, 1.5, 1, [[range(700, 850), range(d - 7, d), 0.2]]
        )
        if i < 4:
            subclu_instance = subclu(samples, 0.49, 10)
            start_subclu = time.perf_counter()
            subclu_instance.process()
            stop_subclu = time.perf_counter()
            subclu_runtime.append(stop_subclu - start_subclu)
            print("Clustering " + str(d) + "d, 1000 n, 1 sc by SUBCLU: done")
        clustering_method = Clustering_By_dbscan(
            fires_dbscan_params[i][0], fires_dbscan_params[i][1]
        )
        fires_instance = fires(samples, 3, 4, 1, clustering_method)
        start_fires = time.perf_counter()
        fires_instance.process()
        stop_fires = time.perf_counter()
        fires_runtime.append(stop_fires - start_fires)
        print("Clustering " + str(d) + "d, 1000 n, 1 sc by FIRES: done")
    plt.figure(figsize=(8, 4))
    #plt.rcParams.update({"font.size": 26})
    plt.plot([10, 15, 20, 25], subclu_runtime, label="SUBCLU")
    plt.plot(dimensionalities, fires_runtime, label="FIRES")
    plt.xlabel("Data dimensionality")
    plt.ylabel("Runtime in seconds")
    plt.legend(loc="upper right")
    plt.savefig("subspaceclustering/images/cluster_xd_1000n_1sc.pdf")
    #plt.show()


def cluster_xd_xn_10sc():
    """Reproduces figure 7 in the thesis."""
    datasets = ["05", "10", "15", "20", "25", "50"]
    subclu_runtime = []
    fires_runtime = []
    subclu_dbscan_params = [(25, 25), (25, 20), (26, 20), (27, 20)]
    fires_dbscan_params = [(0.8, 8), (0.82, 8), (0.8, 8), (1, 8), (1, 10), (1, 10)]
    for i, dataset in enumerate(datasets):
        samples = read_csv("db_dimensionality_scale/D" + dataset)
        samples = samples[:, :-1]
        if i < 4:
            subclu_instance = subclu(
                samples, subclu_dbscan_params[i][0], subclu_dbscan_params[i][1]
            )
            start_subclu = time.perf_counter()
            subclu_instance.process()
            stop_subclu = time.perf_counter()
            subclu_runtime.append(stop_subclu - start_subclu)
            print("Clustering D" + dataset + " by SUBCLU: done")
        clustering_method = Clustering_By_dbscan(
            fires_dbscan_params[i][0], fires_dbscan_params[i][1]
        )
        fires_instance = fires(samples, 3, 4, 1, clustering_method)
        start_fires = time.perf_counter()
        fires_instance.process()
        stop_fires = time.perf_counter()
        fires_runtime.append(stop_fires - start_fires)
        print("Clustering D" + dataset + " by FIRES: done")
    plt.figure(figsize=(8, 4))
    #plt.rcParams.update({"font.size": 26})
    plt.plot([5, 10, 15, 20], subclu_runtime, label="SUBCLU")
    plt.plot([5, 10, 15, 20, 25, 50], fires_runtime, label="FIRES")
    plt.xlabel("Data dimensionality")
    plt.ylabel("Runtime in seconds")
    plt.legend(loc="upper left")
    plt.savefig("subspaceclustering/images/cluster_xd_xn_10sc.pdf")
    #plt.show()


def read_csv(filename):
    samples = None
    with open("subspaceclustering/samples/" + filename + ".csv", newline="") as csvfile:
        reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
        samples = np.array(list(reader)).astype(float)
    return samples


# compare_clustering_quality()
# cluster_10d_xn_2sc()
# cluster_10d_xn_10sc()
# cluster_xd_1000n_1sc()
# cluster_xd_xn_10sc()
