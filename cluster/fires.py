import numpy as np
from math import sqrt
from itertools import combinations
from collections import Counter
from functools import reduce
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from pyclustering.cluster.clique import clique


class Clustering_Method:
  def __init__(self):
    pass

class Clustering_By_dbscan(Clustering_Method):
  def __init__(self, eps, minpts, distance_metric='euclidean'):
    self.__eps = eps
    self.__minpts = minpts
    self.__distance_metric = distance_metric
  def process(self, data, new_eps = None):
    if new_eps is None:
      dbscan_instance = DBSCAN(eps=self.__eps, min_samples= self.__minpts,
                               metric=self.__distance_metric, n_jobs=-1).fit(data)
    else:
      dbscan_instance = DBSCAN(eps= new_eps, min_samples= self.__minpts,
                               metric=self.__distance_metric, n_jobs=-1).fit(data)
    return dbscan_instance
  def get_eps(self):
    return self.__eps



class Clustering_By_kmeans(Clustering_Method):
  def __init__(self, amount_centers):
    self.__amount_centers = amount_centers
  def process(self, data):
    kmeans_instance = KMeans(n_clusters=self.__amount_centers, init='k-means++',
                             n_init=10, max_iter=300, tol=0.0001, verbose=0,
                             random_state=None, copy_x=True, algorithm='lloyd').fit(data)
    return kmeans_instance



class Clustering_By_clique(Clustering_Method):
  def __init__(self, amount_intervals, density_threshold):
    self.__intervals = amount_intervals
    self.__threshold = density_threshold
    self.labels_ = None
  def process(self, data):
    list_of_points = data.tolist()
    clique_instance = clique(list_of_points, self.__intervals,self.__threshold)
    clusters = clique_instance.get_clusters()
    self.labels_ = np.zeros(len(list_of_points),)-1
    for i in range(len(clusters)):
      self.labels_[clusters[i]] = i
    return self



class fires:

  def __init__(self, data, mu, k, minClu, clustering_method: Clustering_Method):
    self.__data = data
    self.__mu = mu
    self.__k = k
    self.__minClu = minClu
    self.__clustering_method = clustering_method

    self.__clusters = {}
    self.__pruned_C1 = []
    self.__unpruned_C1 = []
    self.__cluster_to_dimension = {}
    self.__split_clusters = {}
    self.__k_most_similar_clusters = {}
    self.__best_merge_candidates = {}
    self.__best_merge_clusters = []
    self.__subspace_cluster_approximations = []
    self.__verify_arguments()


  def __verify_arguments(self):

    if self.__minClu < 1:
      raise ValueError("minClu should be a natural number.")

    if len(self.__data) == 0:
      raise ValueError("Input data is empty (size: '%d')." % len(self.__data))



  def get_clusters(self):
    return self.__clusters

  def get__pruned_C1(self):
    return self.__pruned_C1

  def get_cluster_to_dimension(self):
    return self.__cluster_to_dimension

  def get_k_most_similar_clusters(self):
    return self.__k_most_similar_clusters

  def get__best_merge_candidates(self):
    return self.__best_merge_candidates

  def get__best_merge_clusters(self):
    return self.__best_merge_clusters

  def get__subspace_cluster_approximations(self):
    return self.__subspace_cluster_approximations



  def process(self):
    self.generate_base_clusters()
    self.prune_irrelevant_base_clusters()
    self.check_for_base_clusters_splits()
    self.compute_k_most_similar_clusters()
    self.compute_best_merge_candidates()
    self.compute_best_merge_clusters()
    self.generate_subspace_cluster_approximations()
    self.prune_irrelevant_merge_candidate_of_subspace_cluster_approximations()
    self.refine_cluster_approximations()
    return self


  def generate_base_clusters(self):
    for dimension in range(len(self.__data[0,:])):
      vector = self.__data[:,dimension].reshape(-1, 1)
      clustering = self.__clustering_method.process(vector)
      labels = clustering.labels_
      for label in range(len(set(labels)) - (1 if -1 in labels else 0)):
        self.__unpruned_C1.append(np.where(labels == label)[0])
        self.__cluster_to_dimension[len(self.__unpruned_C1) -1] = dimension



  def prune_irrelevant_base_clusters(self):
    _25_percent_of_s_avg = fires.compute_s_avg(self.__unpruned_C1) / 4
    for i, cluster in enumerate(self.__unpruned_C1):
      if len(cluster) >= _25_percent_of_s_avg:
        self.__pruned_C1.append(cluster)
        self.__cluster_to_dimension[len(self.__pruned_C1) -1] = self.__cluster_to_dimension.pop(i)
      else:
        self.__cluster_to_dimension.pop(i)


  @staticmethod
  def compute_s_avg(C1):
    try:
      return sum(map(len, C1)) / len(C1)
    except ZeroDivisionError:
      print('no base clusters were found')



  def check_for_base_clusters_splits(self):
    two_thirds_of_s_avg = 2 * fires.compute_s_avg(self.__pruned_C1) / 3
    for i, cluster in enumerate(self.__pruned_C1):
      most_similar_cluster = self.compute_most_similar_cluster(i, cluster)
      intersection = np.intersect1d(self.__pruned_C1[most_similar_cluster],cluster)
      difference = np.setdiff1d(cluster, self.__pruned_C1[most_similar_cluster])
      if(len(intersection)>= two_thirds_of_s_avg and len(difference)>= two_thirds_of_s_avg):
        self.__pruned_C1[i] = intersection
        self.__pruned_C1.append(difference)
        self.__cluster_to_dimension[len(self.__pruned_C1) -1] = self.__cluster_to_dimension.get(i)
        #is this correct or should it be Top-Down check?
        #self.check_for_base_clusters_splits()
        #break


  def compute_most_similar_cluster(self, cluster_index, cluster):
    intersections = list(map(lambda x: len(np.intersect1d(x, cluster)), self.__pruned_C1))
    intersections[cluster_index] = -1
    arr = np.array(intersections)
    mscs = np.where(arr == np.amax(arr))[0]
    unique_of_msc = list(map(lambda v: len(np.setdiff1d(self.__pruned_C1[v], cluster)), mscs))
    return mscs[unique_of_msc.index(min(unique_of_msc))]



  def compute_k_most_similar_clusters(self):
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
          max_intersection = np.amax(arr)
          number_max_intersection = counter[max_intersection]
          if number_max_intersection == k:
            self.__k_most_similar_clusters.get(i).extend(list(np.where(arr == max_intersection)[0]))
            k = 0
          elif number_max_intersection > k:
            indices_max_intersections = np.where(arr == max_intersection)[0]
            self.__k_most_similar_clusters.get(i).extend(list(indices_max_intersections[np.argpartition(arr2[indices_max_intersections], k)[:k]]))
            k = 0
          else:
            self.__k_most_similar_clusters.get(i).extend(list(np.where(arr == max_intersection)[0]))
            arr[arr == max_intersection] = -1
            k = k - number_max_intersection
    else:
      raise ValueError("k should not be greater than the number of base clusters('%d')"% len(self.__pruned_C1))



  def compute_best_merge_candidates(self):
    number_base_clusters = len(self.__pruned_C1)
    sorted_combinations = list(combinations(range(number_base_clusters), r=2))
    for combination in sorted_combinations:
      if (len(set(self.__k_most_similar_clusters.get(combination[0])) & set(self.__k_most_similar_clusters.get(combination[1]))) >= self.__mu):
        self.__best_merge_candidates[combination[0]] = self.__best_merge_candidates.get(combination[0], [])
        self.__best_merge_candidates.get(combination[0]).append(combination[1])
        self.__best_merge_candidates[combination[1]] = self.__best_merge_candidates.get(combination[1], [])
        self.__best_merge_candidates.get(combination[1]).append(combination[0])



  def compute_best_merge_clusters(self):
    for cluster_index, bm_candidates in self.__best_merge_candidates.items():
      if len(bm_candidates) >= self.__minClu:
        self.__best_merge_clusters.append(cluster_index)



  def generate_subspace_cluster_approximations(self):
    merged_list = None
    sorted_combinations = list(combinations(self.__best_merge_clusters, r=2))
    for combination in sorted_combinations:
      if (combination[0] in self.__best_merge_candidates.get(combination[1])
          and combination[1] in self.__best_merge_candidates.get(combination[0])):
        merged_list = list(set(self.__best_merge_candidates.get(combination[0]) + self.__best_merge_candidates.get(combination[1])))
        if all(Counter(x) != Counter(merged_list) for x in self.__subspace_cluster_approximations):
          self.__subspace_cluster_approximations.append(merged_list.copy())



  def prune_irrelevant_merge_candidate_of_subspace_cluster_approximations(self):

    for index, approximation in enumerate(self.__subspace_cluster_approximations):
      score_approximation = self.compute_quality_of_subspace_cluster_approximation(approximation)
      score_approximation_without_base_cluster = self.compute_quality_of_approximation_without_one_cluster(approximation)
      i = 0
      count_remaining_base_clusters = len(approximation)
      while (i < len(approximation) and len(approximation) > 1):
        if len(approximation) < count_remaining_base_clusters:
          score_approximation_without_base_cluster = self.compute_quality_of_approximation_without_one_cluster(approximation)
          count_remaining_base_clusters = count_remaining_base_clusters - 1

        remaining_scores = [x for z,x in enumerate(score_approximation_without_base_cluster) if z!=i]
        if (score_approximation_without_base_cluster[i] > score_approximation
            and all(v <= score_approximation_without_base_cluster[i] for v in remaining_scores)):
          approximation.pop(i)
          score_approximation = score_approximation_without_base_cluster[i]
          i = 0
        else:
          i = i + 1



  def compute_quality_of_subspace_cluster_approximation(self, approximation):
    cluster_dimensionality = len(approximation)
    #cluster_dimensionality = len(set([self.__cluster_to_dimension.get(j) for j in approximation]))
    base_clusters_of_approximation = [self.__pruned_C1[i] for i in approximation]
    cluster_size = len(reduce(np.intersect1d, base_clusters_of_approximation))
    return sqrt(cluster_size) * cluster_dimensionality



  def compute_quality_of_approximation_without_one_cluster(self, approximation):
    score_approximation_without_base_cluster = []
    for cluster_index in approximation:
      temp_approximation = approximation.copy()
      temp_approximation.remove(cluster_index)
      score = self.compute_quality_of_subspace_cluster_approximation(temp_approximation)
      score_approximation_without_base_cluster.append(score)
    return score_approximation_without_base_cluster




  def refine_cluster_approximations(self):

    for approximation in self.__subspace_cluster_approximations:
      features = list(set(self.__cluster_to_dimension.get(k) for k in approximation))
      features.sort()
      base_clusters = [self.__pruned_C1[i] for i in approximation]
      base_clusters_union = reduce(np.union1d, base_clusters)
      points_in_new_subspace = self.get_cluster_members_values(base_clusters_union, features)
      if isinstance(self.__clustering_method, Clustering_By_dbscan):
        new_eps = fires.adjust_density_threshold(self.__clustering_method.get_eps(), len(base_clusters_union), len(features))
        clustering = self.__clustering_method.process(points_in_new_subspace, new_eps)
      else:
        clustering = self.__clustering_method.process(points_in_new_subspace)
      labels = clustering.labels_
      if 0 in labels:
        self.__clusters[tuple(features)] = self.__clusters.get(tuple(features), [])
        for label in range(len(set(labels)) - (1 if -1 in labels else 0)):
          self.__clusters[tuple(features)].append(base_clusters_union[list(np.where(labels == label)[0])])



  def get_cluster_members_values(self, points, features):
    return self.__data[np.ix_(points,features)]

  @staticmethod
  def adjust_density_threshold(eps, n, d):
    return (eps * n) / (pow(n,(1/d)))






