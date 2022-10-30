import numpy as np
import math
from itertools import chain
from scipy.optimize import linear_sum_assignment
import warnings


def overall_f1_recall(true_clusters, found_clusters):
    """
    Computes the overall F1-Recall evaluation measure for two clusterings. For
    more information please check
    https://dl.acm.org/doi/10.1145/2063576.2063774

    Parameters
    ----------
    true_clusters (dictionary): The ground truth. The keys are subspaces and the
                                values are lists of hidden clusters in these
                                subspaces.

    found_clusters (dictionary): Clustering result. The keys are subspaces that
                                contain clusters and the values are lists of
                                found clusters in these subspaces.
    """

    verify_arguments(true_clusters, found_clusters)
    f1_r = 0
    all_found_clusters = list(chain.from_iterable(found_clusters.values()))
    for subspace in true_clusters.keys():
        for cluster in true_clusters.get(subspace):
            intersections = np.array(
                list(map(lambda x: len(np.intersect1d(x, cluster)), all_found_clusters))
            )
            recalls = intersections / len(cluster)
            precisions = intersections / list(map(len, all_found_clusters))
            with np.errstate(divide="ignore", invalid="ignore"):
                f1s = (2 * recalls * precisions) / (recalls + precisions)
                f1s[np.isnan(f1s)] = 0
            f1_r = f1_r + max(f1s, default=0)
    return f1_r / sum(map(len, true_clusters.values()))


def overall_f1_precision(true_clusters, found_clusters):
    """
    Computes the overall F1-Precision evaluation measure for two clusterings. For
    more information please check
    https://dl.acm.org/doi/10.1145/2063576.2063774

    Parameters
    ----------
    true_clusters (dictionary): The ground truth. The keys are subspaces and the
                                values are lists of hidden clusters in these
                                subspaces.

    found_clusters (dictionary): Clustering result. The keys are subspaces that
                                contain clusters and the values are lists of
                                found clusters in these subspaces.
    """

    verify_arguments(true_clusters, found_clusters)
    f1_p = 0
    all_true_clusters = list(chain.from_iterable(true_clusters.values()))
    for subspace in found_clusters.keys():
        for cluster in found_clusters.get(subspace):
            intersections = np.array(
                list(map(lambda x: len(np.intersect1d(x, cluster)), all_true_clusters))
            )
            recalls = intersections / len(cluster)
            precisions = intersections / list(map(len, all_true_clusters))
            with np.errstate(divide="ignore", invalid="ignore"):
                f1s = (2 * recalls * precisions) / (recalls + precisions)
                f1s[np.isnan(f1s)] = 0
            f1_p = f1_p + max(f1s, default=0)
    return f1_p / sum(map(len, found_clusters.values()))


def overall_f1_merge(true_clusters, found_clusters):
    """
    Computes the overall F1-Merge evaluation measure for two clusterings. For
    more information please check
    https://dl.acm.org/doi/10.1145/2063576.2063774

    Parameters
    ----------
    true_clusters (dictionary): The ground truth. The keys are subspaces and the
                                values are lists of hidden clusters in these
                                subspaces.

    found_clusters (dictionary): Clustering result. The keys are subspaces that
                                contain clusters and the values are lists of
                                found clusters in these subspaces.
    """

    verify_arguments(true_clusters, found_clusters)
    f1_m = 0
    all_found_clusters = list(chain.from_iterable(found_clusters.values()))
    all_true_clusters = list(chain.from_iterable(true_clusters.values()))
    sizes_true_clusters = list(map(len, all_true_clusters))
    number_true_clusters = len(all_true_clusters)
    mapped_clusters = [[]] * number_true_clusters
    # merge found clusters if their best matching cluster in Ground is identical
    for cluster in all_found_clusters:
        intersections = np.array(
            list(map(lambda x: len(np.intersect1d(x, cluster)), all_true_clusters))
        )
        coverages = intersections / sizes_true_clusters
        if np.any(coverages):
            best_matching_clusters = np.where(coverages == max(coverages))[0]
            for best_matching_cluster in best_matching_clusters:
                mapped_clusters[best_matching_cluster] = np.union1d(
                    mapped_clusters[best_matching_cluster], cluster
                )
    for cluster_t, mapped_cluster in zip(all_true_clusters, mapped_clusters):
        try:
            intersection = len(np.intersect1d(cluster_t, mapped_cluster))
            recall = intersection / len(cluster_t)
            precision = intersection / len(mapped_cluster)
            f1 = (2 * recall * precision) / (recall + precision)
        except ZeroDivisionError:
            f1 = 0
        f1_m = f1_m + f1
    return f1_m / number_true_clusters


def overall_rnia(db_n, db_d, true_clusters, found_clusters):
    """
    Computes the overall RNIA evaluation measure for two clusterings. For
    more information please check
    https://dl.acm.org/doi/10.1145/2063576.2063774 and
    https://dl.acm.org/doi/10.1109/TKDE.2006.106

    Parameters
    ----------
    db_n (int): The number of points in the data.

    db_d (int): The dimensionality of the data.

    true_clusters (dictionary): The ground truth. The keys are subspaces and the
                                values are lists of hidden clusters in these
                                subspaces.

    found_clusters (dictionary): Clustering result. The keys are subspaces that
                                contain clusters and the values are lists of
                                found clusters in these subspaces.
    """

    verify_arguments(true_clusters, found_clusters, db_n, db_d)
    true_micro_objects, found_micro_objects = compute_true_and_found_micro_objects(
        db_n, db_d, true_clusters, found_clusters
    )
    cardinality_intersection_micro_objects = np.minimum(
        true_micro_objects, found_micro_objects
    ).sum()
    cardinality_union_micro_objects = np.maximum(
        true_micro_objects, found_micro_objects
    ).sum()
    rnia = (
        cardinality_union_micro_objects - cardinality_intersection_micro_objects
    ) / cardinality_union_micro_objects
    return 1 - rnia


def overall_ce(db_n, db_d, true_clusters, found_clusters):
    """
    Computes the overall CE evaluation measure for two clusterings. For
    more information please check
    https://dl.acm.org/doi/10.1145/2063576.2063774 and
    https://dl.acm.org/doi/10.1109/TKDE.2006.106

    Parameters
    ----------
    db_n (int): The number of points in the data.

    db_d (int): The dimensionality of the data.

    true_clusters (dictionary): The ground truth. The keys are subspaces and the
                                values are lists of hidden clusters in these
                                subspaces.

    found_clusters (dictionary): Clustering result. The keys are subspaces that
                                contain clusters and the values are lists of
                                found clusters in these subspaces.
    """

    verify_arguments(true_clusters, found_clusters, db_n, db_d)
    d_max = 0
    number_true_clusters = sum(map(len, true_clusters.values()))
    number_found_clusters = sum(map(len, found_clusters.values()))
    max_cluster_number = max(number_true_clusters, number_found_clusters)
    cost_matrix = np.zeros(
        shape=(max_cluster_number, max_cluster_number)
    )  # matching matrix
    i = 0
    j = 0
    for subspace_t in true_clusters.keys():
        for cluster_t in true_clusters.get(subspace_t):
            for subspace_r in found_clusters.keys():
                mutual_dimensions = len(np.intersect1d(subspace_t, subspace_r))
                if mutual_dimensions == 0:
                    j = j + len(found_clusters.get(subspace_r))
                else:
                    intersections_with_cluster_t = list(
                        map(
                            lambda x: len(np.intersect1d(x, cluster_t)),
                            found_clusters.get(subspace_r),
                        )
                    )
                    for intersection in intersections_with_cluster_t:
                        cost_matrix[i][j] = (
                            intersection * mutual_dimensions
                        )  # micro objects intersection
                        j = j + 1
            i = i + 1
            j = 0
    row_ind, col_ind = linear_sum_assignment(
        cost_matrix, maximize=True
    )  # find a permutation of clusters that maximaize the total sum over all cardinalities
    d_max = cost_matrix[row_ind, col_ind].sum()

    true_micro_objects, found_micro_objects = compute_true_and_found_micro_objects(
        db_n, db_d, true_clusters, found_clusters
    )
    cardinality_union_micro_objects = np.maximum(
        true_micro_objects, found_micro_objects
    ).sum()
    ce = (cardinality_union_micro_objects - d_max) / cardinality_union_micro_objects
    return 1 - ce


def compute_true_and_found_micro_objects(db_n, db_d, true_clusters, found_clusters):
    """
    Computes the representation of true and found clusters by their set of micro-
    objects.

    """

    true_micro_objects = np.zeros(shape=(db_n, db_d))
    found_micro_objects = np.zeros(shape=(db_n, db_d))
    for subspace in true_clusters.keys():
        for cluster_t in true_clusters.get(subspace):
            if isinstance(subspace, int):
                true_micro_objects[np.ix_(cluster_t, (subspace,))] += 1
            else:
                true_micro_objects[np.ix_(cluster_t, subspace)] += 1
    for subspace in found_clusters.keys():
        for cluster_r in found_clusters.get(subspace):
            if isinstance(subspace, int):
                found_micro_objects[np.ix_(cluster_r, (subspace,))] += 1
            else:
                found_micro_objects[np.ix_(cluster_r, subspace)] += 1
    return true_micro_objects, found_micro_objects


def e4sc(true_clusters, found_clusters):
    """
    Computes the E4SC evaluation measure for two clusterings. For
    more information please check
    https://dl.acm.org/doi/10.1145/2063576.2063774

    Parameters
    ----------
    true_clusters (dictionary): The ground truth. The keys are subspaces and the
                                values are lists of hidden clusters in these
                                subspaces.

    found_clusters (dictionary): Clustering result. The keys are subspaces that
                                contain clusters and the values are lists of
                                found clusters in these subspaces.
    """

    verify_arguments(true_clusters, found_clusters)
    f1_ground_res = compute_f1_sc_clus(true_clusters, found_clusters)
    f1_res_ground = compute_f1_sc_clus(found_clusters, true_clusters)
    warnings.filterwarnings("ignore")
    try:
        e4sc = (2 * f1_ground_res * f1_res_ground) / (f1_ground_res + f1_res_ground)
    except ZeroDivisionError:
        e4sc = 0
    if math.isnan(e4sc):
        e4sc = 0
    return e4sc


def compute_f1_sc_clus(p, q):
    """
    Computes the overall subspace_aware F1 measure for two clusterings.

    Parameters
    ----------
    p (dictionary): First clustering. The keys are subspaces and the
                    values are lists of clusters in these subspaces.

    q (dictionary): Second clustering. The keys are subspaces and the
                    values are lists of clusters in these subspaces.
    """

    f1_sc_clus = 0
    for subspace_p in p.keys():
        relevant_subspaces = [
            s for s in q.keys() if len(np.intersect1d(subspace_p, s)) != 0
        ]
        for cluster_p in p.get(subspace_p):
            relevant_f1_sc = []
            if isinstance(subspace_p, int):
                micro_objects_cluster_p = len(cluster_p)
            else:
                micro_objects_cluster_p = len(cluster_p) * len(subspace_p)
            for subspace_q in relevant_subspaces:
                clusters_in_subspace_q = q.get(subspace_q)
                mutual_dimensions = len(np.intersect1d(subspace_p, subspace_q))
                intersections_with_cluster_p = np.array(
                    list(
                        map(
                            lambda x: len(np.intersect1d(x, cluster_p)),
                            clusters_in_subspace_q,
                        )
                    )
                )
                micro_objects_intersections = (
                    intersections_with_cluster_p * mutual_dimensions
                )
                recalls_sc = micro_objects_intersections / micro_objects_cluster_p
                if isinstance(subspace_q, int):
                    precisions_sc = micro_objects_intersections / list(
                        map(lambda x: len(x), clusters_in_subspace_q)
                    )
                else:
                    precisions_sc = micro_objects_intersections / list(
                        map(lambda x: len(x) * len(subspace_q), clusters_in_subspace_q)
                    )
                with np.errstate(divide="ignore", invalid="ignore"):
                    f1_sc = (2 * recalls_sc * precisions_sc) / (
                        recalls_sc + precisions_sc
                    )
                    f1_sc[np.isnan(f1_sc)] = 0
                for f1_sc_measure in f1_sc:
                    relevant_f1_sc.append(f1_sc_measure)
            f1_sc_clus = f1_sc_clus + max(relevant_f1_sc, default=0)

    f1_sc_clus = f1_sc_clus / sum(map(len, p.values()))
    return f1_sc_clus


def verify_arguments(true_clusters, found_clusters, db_n=1, db_d=1):
    """
    Verifies input parameters.
    """

    if db_n <= 0:
        raise ValueError("Data can not have size 0.")

    if db_d <= 0:
        raise ValueError("Data dimensionality must be greater than 0.")

    if not found_clusters:
        raise ValueError("Found clusters are empty.")

    if not true_clusters:
        raise ValueError("True clusters are empty.")
