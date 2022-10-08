import numpy as np
from math import sqrt
from itertools import combinations, product
from collections import Counter
from functools import reduce
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from pyclustering.cluster.clique import clique


class Clustering_Method:
  """
  Parent class represents possible clustering methods used by the algorithm
  FIRES.
  """
  def __init__(self):
    pass


class Clustering_By_dbscan(Clustering_Method):
  def __init__(self, eps, minpts, distance_metric='euclidean'):
    """
    Parameters
    ----------
    eps (float): Connectivity radius between points.

    minpts (int): Minimum number of samples in a neighborhood for a point to be
                  considered as a core point. This includes the point itself.

    distance_metric (str): The metric to use when calculating distance between
                          instances in a feature array.
    """
    self.__eps = eps
    self.__minpts = minpts
    self.__distance_metric = distance_metric
  def process(self, data, new_eps = None):
    """
    Performs DBSCAN clustering.

    Parameters
    ----------
    data (ndarray): Input data.

    new_eps (float): Connectivity radius between points.
    """
    if new_eps is None:
      dbscan_instance = DBSCAN(eps=self.__eps, min_samples= self.__minpts,
                               metric=self.__distance_metric, n_jobs=-1).fit(data)
    else:
      neigh = NearestNeighbors(n_neighbors=1, radius=new_eps, metric='euclidean', n_jobs=-1)
      neigh.fit(data)
      distances = neigh.radius_neighbors_graph(data, mode='distance', sort_results=True)
      dbscan_instance = DBSCAN(eps= new_eps, min_samples= self.__minpts,
                               metric='precomputed', n_jobs=-1).fit(distances)
    return dbscan_instance
  def get_eps(self):
    return self.__eps



class Clustering_By_kmeans(Clustering_Method):
  def __init__(self, amount_centers):
    """
    Parameters
    ----------
    amount_centers (int): The number of clusters to form as well as the number
                          of centroids to generate.
    """
    self.__amount_centers = amount_centers
  def process(self, data):
    """
    Performs KMeans clustering.

    Parameters
    ----------
    data (ndarray): Input data.
    """
    kmeans_instance = KMeans(n_clusters=self.__amount_centers, init='k-means++',
                             n_init=10, max_iter=300, tol=0.0001, verbose=0,
                             random_state=None, copy_x=True, algorithm='lloyd').fit(data)
    return kmeans_instance



class Clustering_By_clique(Clustering_Method):
  def __init__(self, amount_intervals, density_threshold):
    """
    Parameters
    ----------
    amount_intervals (int): Amount of intervals in each dimension that defines
                            amount of CLIQUE blocks.

    density_threshold (int): Minimum number of points that should contain
                          CLIQUE block to consider its points as non-outliers.
    """
    self.__intervals = amount_intervals
    self.__threshold = density_threshold
    self.labels_ = None
  def process(self, data):
    """
    Performs CLIQUE clustering.

    Parameters
    ----------
    data (list): Input data.
    """
    list_of_points = data.tolist()
    clique_instance = clique(list_of_points, self.__intervals,self.__threshold)
    clusters = clique_instance.get_clusters()
    self.labels_ = np.zeros(len(list_of_points),)-1
    for i in range(len(clusters)):
      self.labels_[clusters[i]] = i
    return self



class fires:
  """
  Class represents clustering algorithm FIRES.
  For clustering example please check '/examples/fires_example.py'.
  """

  def __init__(self, data, mu, k, minClu, clustering_method: Clustering_Method):
    """
    Constructor of clustering algorithm FIRES.

    Parameters
    ----------
    data (ndarray): Input data.

    mu (int): Number of most similar clusters in which two clusters must
              overlap in order to be considered best-merge-clusters of each
              other.

    k (int): Amount of base-clusters that every base-cluster is compared to for
            merging purposes. For computing the k-most-similar-clusters.

    minClu (int): Minimum number of best-merge-candidates a cluster must have
                  to be considered a best-merge-cluster.

    clustering_method (object): The clustering method used to cluster data
    objects in a subspace at the preclustering and postprocessing steps.
    This object should be an instance of one of the following three available
    classes, Clustering_By_dbscan/ Clustering_By_kmeans/ Clustering_By_clique.
    """

    self.__data = data
    self.__mu = mu
    self.__k = k
    self.__minClu = minClu
    self.__clustering_method = clustering_method

    self.__clusters = {}  # clustering(subspaces to corresponding clusters)
    self.__pruned_C1 = []  #pruned base-clusters, above 25% of the average size
    self.__unpruned_C1 = []#all base-clusters resulting from the pre-clustering
    self.__baseCluster_to_dimension = {}
    self.__k_most_similar_clusters = {}
    self.__best_merge_candidates = {}
    self.__best_merge_clusters = []
    self.__subspace_cluster_approximations = []
    self.__verify_arguments()


  def __verify_arguments(self):
    """
    Verifies input parameters for the algorithm.
    """

    if self.__data is None or self.__data.size == 0:
      raise ValueError("Input data is empty.")
    if self.__minClu < 1:
      raise ValueError("minClu should be a natural number.")
    if self.__mu < 1:
      raise ValueError("mu should be a natural number.")
    if self.__k < 1:
      raise ValueError("k should be a natural number.")
    if not isinstance(self.__clustering_method, (Clustering_By_dbscan, Clustering_By_kmeans, Clustering_By_clique)):
      raise ValueError("clustering_method should be instance of Clustering_By_dbscan or Clustering_By_kmeans or Clustering_By_clique.")


  def get_clusters(self):
    """
    Returns a result of the postprocessing step.
    'self.__clusters' is a dictionary whereby keys are corresponding subspaces
    of the pruned subspace-cluster approximations and values are refined
    subspace clusters.
    """

    return self.__clusters



  def get_pruned_C1(self):
    """
    Returs a list of base-clusters their sizes are above 25% of the average
    size of all base clusters.
    """

    return self.__pruned_C1


  def get_baseCluster_to_dimension(self):
    """
    Returns a dictionary whereby keys are indicies of base-clusters in the list
    'self.__pruned_C1' and values are the corresponding 1-D subspaces in which
    the clusters exist.
    """

    return self.__baseCluster_to_dimension


  def get_k_most_similar_clusters(self):
    return self.__k_most_similar_clusters

  def get_best_merge_candidates(self):
    return self.__best_merge_candidates

  def get_best_merge_clusters(self):
    return self.__best_merge_clusters

  def get_subspace_cluster_approximations(self):
    return self.__subspace_cluster_approximations



  def process(self):
    """
    Performs cluster analysis in line with rules of FIRES algorithm and
    returns a FIRES instance(itself).
    """

    self.generate_base_clusters()
    self.prune_irrelevant_base_clusters()
    self.check_for_base_clusters_splits()
    self.compute_k_most_similar_clusters()
    self.compute_best_merge_candidates()
    self.compute_best_merge_clusters()
    self.generate_subspace_cluster_approximations()
    #self.clean_subspace_cluster_approximations()
    self.prune_irrelevant_merge_candidate_of_subspace_cluster_approximations()
    self.refine_cluster_approximations()
    return self


  def generate_base_clusters(self):
    """
    Generates 1-dimensional subspace clusters by applying one of the available
    clustering methods to each 1-dimensional subspace and computes the
    dictionary '__baseCluster_to_dimension' that tells, in which dimension
    every base-cluster exists.
    """

    for dimension in range(self.__data.shape[1]):
      vector = self.__data[:,dimension].reshape(-1, 1)
      clustering = self.__clustering_method.process(vector)
      labels = clustering.labels_
      for label in np.setdiff1d(labels, -1):
        self.__unpruned_C1.append(np.where(labels == label)[0])
        self.__baseCluster_to_dimension[len(self.__unpruned_C1) -1] = dimension



  def prune_irrelevant_base_clusters(self):
    """
    Removes base-clusters whose size is smaller than 25% of the average size of
    all base-clusters. The wanted base-clusters are stored in the list
    'self.__pruned_C1'.
    """

    _25_percent_of_s_avg = fires.compute_s_avg(self.__unpruned_C1) / 4
    for i, cluster in enumerate(self.__unpruned_C1):
      if len(cluster) >= _25_percent_of_s_avg:
        self.__pruned_C1.append(cluster)
        self.__baseCluster_to_dimension[len(self.__pruned_C1) -1] = self.__baseCluster_to_dimension.pop(i)
      else:
        self.__baseCluster_to_dimension.pop(i)


  @staticmethod
  def compute_s_avg(C1):
    """
    Computes the average size of all base-clusters and throws exception in case
    no base-clusters were found.

    Parameters
    ----------
    C1 (list): All found base-clusters.
    """

    try:
      return sum(map(len, C1)) / len(C1)
    except ZeroDivisionError:
      print('no base clusters were found')



  def check_for_base_clusters_splits(self):
    """
    Split a base-cluster to merge it with two different clusters, if the size
    of both clusters resulting from the split is at least two-thirds the
    average size of base-clusters. The two clusters resulting from the split
    are the intersection with the most_similar_cluster and the remaining points
    in the base-cluster(split candidate).
    """

    two_thirds_of_s_avg = 2 * fires.compute_s_avg(self.__pruned_C1) / 3
    all_clusters_checked = False
    while not all_clusters_checked:
      splits = {}
      all_clusters_checked = True
      for i, cluster in enumerate(self.__pruned_C1):
        most_similar_cluster = self.compute_most_similar_cluster(i, cluster)
        intersection = np.intersect1d(self.__pruned_C1[most_similar_cluster],cluster)
        difference = np.setdiff1d(cluster, self.__pruned_C1[most_similar_cluster])
        #print('cluster-index', i, 'msc ', most_similar_cluster, 'two_thirds_of_s_avg ', two_thirds_of_s_avg,'intersection ', len(intersection), 'difference ', len(difference))
        if(len(intersection)>= two_thirds_of_s_avg and len(difference)>= two_thirds_of_s_avg):
          all_clusters_checked = False
          splits[i] = [intersection, difference]
      for j, new_clusters in splits.items():
        self.__pruned_C1.append(new_clusters[1])
        self.__baseCluster_to_dimension[len(self.__pruned_C1) -1] = self.__baseCluster_to_dimension.get(j)
        self.__pruned_C1[j] = new_clusters[0]




  def compute_most_similar_cluster(self, cluster_index, cluster):
    """
    Searches the most_similar_cluster of a base-cluster. That means the cluster
    with the largest intersection. If there are more than one
    most_similar_cluster, the one with the least difference is chosen.

    Parameters
    ----------
    cluster_index (int): Index of the base-cluster in the list of all pruned
                          base-clusters 'self.__pruned_C1'.

    cluster (array): The base-cluster for which we search the
                      most_similar_cluster.
    """

    intersections = list(map(lambda x: len(np.intersect1d(x, cluster)), self.__pruned_C1))
    intersections[cluster_index] = -1
    arr = np.array(intersections)
    mscs = np.where(arr == np.amax(arr))[0]
    unique_of_mscs = list(map(lambda v: len(np.setdiff1d(self.__pruned_C1[v], cluster)), mscs))
    return mscs[np.argmin(unique_of_mscs)]



  def compute_k_most_similar_clusters(self):
    """
    Searches the k most_similar_clusters for each base-cluster. If there are
    more than k most_similar_clusters, the ones with the least difference
    are chosen. The results are stored in the dictionary
    'self.__k_most_similar_clusters' whereby the keys are indicies of
    base-clusters in the list of all pruned base-cltsers and the values are
    lists containing indicies of the k most_similar_clusters.
    """

    if self.__k < len(self.__pruned_C1):
      for i, cluster in enumerate(self.__pruned_C1):
        self.__k_most_similar_clusters[i] = []
        intersections = list(map(lambda x: len(np.intersect1d(x, cluster)), self.__pruned_C1))
        differences = list(map(lambda v: len(np.setdiff1d(v, cluster)), self.__pruned_C1))
        intersections[i] = -1
        arr = np.array(intersections)
        arr2 = np.array(differences)
        counter = Counter(arr)
        k = self.__k
        while k > 0:
          maxnew_clusters = np.amax(arr)
          number_maxnew_clusters = counter[maxnew_clusters]
          if number_maxnew_clusters == k:
            self.__k_most_similar_clusters.get(i).extend(list(np.where(arr == maxnew_clusters)[0]))
            k = 0
          elif number_maxnew_clusters > k:
            indices_maxnew_clusterss = np.where(arr == maxnew_clusters)[0]
            self.__k_most_similar_clusters.get(i).extend(list(indices_maxnew_clusterss[np.argpartition(arr2[indices_maxnew_clusterss], k)[:k]]))
            k = 0
          else:
            self.__k_most_similar_clusters.get(i).extend(list(np.where(arr == maxnew_clusters)[0]))
            arr[arr == maxnew_clusters] = -1
            k = k - number_maxnew_clusters
    else:
      raise ValueError("k should not be equal or greater than the number of base clusters('%d')"% len(self.__pruned_C1))



  def compute_best_merge_candidates(self):
    """
    Searches all best_merge_candidates for each base-cluster. The results are
    stored in the dictionary 'self.__best_merge_candidates' whereby the keys
    are indicies of base-clusters in the list of all pruned base-cltsers and
    the values are lists containing indicies of the best_merge_candidates.
    """

    number_base_clusters = len(self.__pruned_C1)
    sorted_combinations = list(combinations(range(number_base_clusters), r=2))
    for combination in sorted_combinations:
      if (len(set(self.__k_most_similar_clusters.get(combination[0])) & set(self.__k_most_similar_clusters.get(combination[1]))) >= self.__mu):
        self.__best_merge_candidates[combination[0]] = self.__best_merge_candidates.get(combination[0], [])
        self.__best_merge_candidates.get(combination[0]).append(combination[1])
        self.__best_merge_candidates[combination[1]] = self.__best_merge_candidates.get(combination[1], [])
        self.__best_merge_candidates.get(combination[1]).append(combination[0])



  def compute_best_merge_clusters(self):
    """
    Computes the list of all best_merge_clusters. The resulting list contains
    indicies of the best_merge_clusters in the list of all pruned base-cltsers
    'self.__pruned_C1'.
    """

    for cluster_index, bm_candidates in self.__best_merge_candidates.items():
      if len(bm_candidates) >= self.__minClu:
        self.__best_merge_clusters.append(cluster_index)



  def generate_subspace_cluster_approximations(self):
    """
    Generates subspace_cluster_approximations by grouping every pair of
    best-merge-clusters with all of their best-merge-candidates together if
    they are best-merge-candidates of each other. The results are stored in the
    list 'self.__subspace_cluster_approximations'.
    """

    merged_list = None
    sorted_combinations = list(combinations(self.__best_merge_clusters, r=2))
    for combination in sorted_combinations:
      if (combination[0] in self.__best_merge_candidates.get(combination[1])
          and combination[1] in self.__best_merge_candidates.get(combination[0])):
        merged_list = list(set(self.__best_merge_candidates.get(combination[0]) + self.__best_merge_candidates.get(combination[1])))
        if all(Counter(x) != Counter(merged_list) for x in self.__subspace_cluster_approximations):
          self.__subspace_cluster_approximations.append(merged_list.copy())



  def clean_subspace_cluster_approximations(self):
    for approximation in self.__subspace_cluster_approximations:
      scores = self.compute_quality_of_approximation_ignoring_one_cluster_eachTime(approximation)
      while (sum(scores) == 0):
        dimensions = list(map(self.__baseCluster_to_dimension.get, approximation))
        counter = Counter(dimensions)
        most_repeated_dimension = counter.most_common()[0][0]
        relevant_clusters = np.where(np.array(dimensions) == most_repeated_dimension)[0]
        clusters_of_mrd = []
        for i in relevant_clusters:
          clusters_of_mrd.append(self.__pruned_C1[approximation[i]])
        sizes = list(map(len, clusters_of_mrd))
        del approximation[relevant_clusters[np.argmin(sizes)]]
        scores = self.compute_quality_of_approximation_ignoring_one_cluster_eachTime(approximation)



  def prune_irrelevant_merge_candidate_of_subspace_cluster_approximations(self):
    """
    Removes repeatedly irrelevant merge_candidates from a
    subspace_cluster_approximation if their removal improves the quality of the
    subspace cluster. This is a top-down pruning. For more details please check
    the Fires paper page 5.
    """

    for index, approximation in enumerate(self.__subspace_cluster_approximations):
      score_approximation = self.compute_quality_of_subspace_cluster_approximation(approximation)
      score_approximation_ignoring_one_cluster_eachTime = self.compute_quality_of_approximation_ignoring_one_cluster_eachTime(approximation)
      if np.any(np.greater(score_approximation_ignoring_one_cluster_eachTime, score_approximation) == True):
        i = 0
        count_remaining_base_clusters = len(approximation)
        while (i < len(approximation) and len(approximation) > 1):
          #print(approximation,score_approximation, score_approximation_ignoring_one_cluster_eachTime, list(map(self.__baseCluster_to_dimension.get, approximation) ))
          if len(approximation) < count_remaining_base_clusters:
            score_approximation_ignoring_one_cluster_eachTime = self.compute_quality_of_approximation_ignoring_one_cluster_eachTime(approximation)
            count_remaining_base_clusters = count_remaining_base_clusters - 1

          remaining_scores = [x for z,x in enumerate(score_approximation_ignoring_one_cluster_eachTime) if z!=i]
          if (score_approximation_ignoring_one_cluster_eachTime[i] > score_approximation
              and np.all(np.less_equal(remaining_scores,score_approximation_ignoring_one_cluster_eachTime[i]) == True)):
            approximation.pop(i)
            score_approximation = score_approximation_ignoring_one_cluster_eachTime[i]
            i = 0
          else:
            i = i + 1
    #check if any approximation appears twice after pruning?
    amount_approximations = len(self.__subspace_cluster_approximations)
    counters_of_approximations = list(map(Counter, self.__subspace_cluster_approximations))
    sorted_combinations = list(combinations(range(amount_approximations), r=2))
    must_deleted = []
    for j in range(amount_approximations-1):
      for combi in [w for w in sorted_combinations if w[0] == j]:
        if (counters_of_approximations[combi[0]] == counters_of_approximations[combi[1]]):
          must_deleted.append(combi[0])
          break
    for h in reversed(must_deleted):
      del self.__subspace_cluster_approximations[h]
    #check if any approximation is a strict subset of any other one?and delete it?
    """amount_approximations = len(self.__subspace_cluster_approximations)
    sorted_combinations = list(combinations(range(amount_approximations), r=2))
    must_deleted = set()
    for j in range(amount_approximations-1):
      if j not in must_deleted:
        first_approximation = self.__subspace_cluster_approximations[j]
        for combi in [w for w in sorted_combinations if w[0] == j]:
          if (set(first_approximation).issubset(self.__subspace_cluster_approximations[combi[1]])):
            must_deleted.add(combi[0])
            break
          elif (set(self.__subspace_cluster_approximations[combi[1]]).issubset(first_approximation)):
            must_deleted.add(combi[1])
    for h in sorted(must_deleted, reverse=True):
      del self.__subspace_cluster_approximations[h]"""



  def compute_quality_of_subspace_cluster_approximation(self, approximation):
    """
    Computes the quality of a subspace_cluster_approximation w.r.t. its
    dimensionality and the number of objects shared by all its
    base-clusters(merge_candidates).

    Parameters
    ----------
    approximation (list): A subspace_cluster_approximation.
    """

    base_clusters_of_approximation = [self.__pruned_C1[i] for i in approximation]
    cluster_size = len(reduce(np.intersect1d, base_clusters_of_approximation))
    return sqrt(cluster_size) * len(approximation)



  def compute_quality_of_approximation_ignoring_one_cluster_eachTime(self, approximation):
    """
    Computes the quality of a subspace_cluster_approximation ignoring one
    base-cluster(merge_candidate) eachTime.

    Parameters
    ----------
    approximation (list): A subspace_cluster_approximation.
    """

    size_approximation = len(approximation)
    score_approximation_ignoring_one_cluster_eachTime = []
    for i in range(size_approximation):
      approximation_without_one_cluster = np.array(approximation)[np.arange(size_approximation)!=i]
      score = self.compute_quality_of_subspace_cluster_approximation(approximation_without_one_cluster)
      score_approximation_ignoring_one_cluster_eachTime.append(score)
    return score_approximation_ignoring_one_cluster_eachTime


  def refine_cluster_approximations(self):
    """
    Generates the definitive clusters from the subspace_cluster_approximations.
    For each approximation all base-clusters in the approximation are combined,
    then a clustering algorithm is performed on these points.
    """

    for approximation in self.__subspace_cluster_approximations:
      features = tuple(set(self.__baseCluster_to_dimension.get(k) for k in approximation))
      base_clusters = [self.__pruned_C1[i] for i in approximation]
      base_clusters_union = reduce(np.union1d, base_clusters)
      projections_into_subspace = self.get_points_values_in_subspace(base_clusters_union, features)
      #print('////', approximation, features, base_clusters_union)
      if isinstance(self.__clustering_method, Clustering_By_dbscan):
        new_eps = self.adjust_density_threshold(self.__clustering_method.get_eps(), len(features))
        clustering = self.__clustering_method.process(projections_into_subspace, new_eps)
      else:
        clustering = self.__clustering_method.process(projections_into_subspace)
      labels = clustering.labels_
      if 0 in labels:
        self.__clusters[features] = self.__clusters.get(features, [])
        for label in np.setdiff1d(labels, -1):
          self.__clusters[features].append(base_clusters_union[list(np.where(labels == label)[0])])



  def get_points_values_in_subspace(self, points, features):
    """
    Extracts the projection of objects into a specific subspace.

    Parameters
    ----------
    points (array): Indices of the objects.

    features (tuple): Indices of the dimensions forming the subspace.
    """

    return self.__data[np.ix_(points,features)]



  def adjust_density_threshold(self, eps, d):
    """
    Computes the post_dbscan_epsilon by adjusting the density threshold to the
    subspace dimensionality.

    Parameters
    ----------
    eps (float): The epsilon-value in the 1D subspace(pre_dbscan_epsilon).

    d (int): Dimensionality of the subspace.
    """

    return (eps * self.__data.shape[0]) / (pow(self.__data.shape[0], (1/d)))





