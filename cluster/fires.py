import numpy as np
from math import sqrt
from itertools import combinations
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
    self.split_base_clusters()
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
    for i in range(len(self.__unpruned_C1)):
      if len(self.__unpruned_C1[i]) >= _25_percent_of_s_avg:
        self.__pruned_C1.append(self.__unpruned_C1[i])
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
    for cluster in self.__pruned_C1:
      list_of_intersections = list(map(lambda x: len(np.intersect1d(x, cluster)), self.__pruned_C1))
      if list_of_intersections.count(len(cluster)) == 1:
        most_similar_clusters = self.compute_most_similar_cluster(cluster)
        #for msc_index in most_similar_clusters:
          #if(len(set(self.__pruned_C1[msc_index]) & set(cluster)) >= two_thirds_of_s_avg
            # and set(cluster).difference(set(self.__pruned_C1[msc_index])) >= two_thirds_of_s_avg):
        first_condition = len(set(self.__pruned_C1[most_similar_clusters]) & set(cluster)) >= two_thirds_of_s_avg
        second_condition = len(set(cluster).difference(set(self.__pruned_C1[most_similar_clusters]))) >= two_thirds_of_s_avg
        if(first_condition and second_condition):
          self.__split_clusters[self.__pruned_C1.index(cluster)] = list(set(self.__pruned_C1[most_similar_clusters]) & set(cluster))




  def compute_most_similar_cluster(self, cluster):
    similarities = []
    for c in self.__pruned_C1:
      if np.array_equal(c, cluster):
        similarities.append(-1)
      else:
        similarities.append(len(set(c) & set(cluster)))
    #return np.where(np.array(similarities) == np.max(similarities))[0]
    return similarities.index(np.max(similarities))




  def split_base_clusters(self):
    for cluster_index, clusters_intersection in self.__split_clusters.items():
      cluster = self.__pruned_C1[cluster_index].copy()
      self.__pruned_C1[cluster_index] = np.array(clusters_intersection)
      self.__pruned_C1.append(np.array(list(set(cluster).difference(set(clusters_intersection)))))
      self.__cluster_to_dimension[len(self.__pruned_C1) -1] = self.__cluster_to_dimension.get(cluster_index)




  def compute_k_most_similar_clusters(self):
    number_of_base_clusters = len(self.__pruned_C1)
    if self.__k < number_of_base_clusters:
      similarities = []
      indices_of_sorted_similarities = []
      for c1 in range(number_of_base_clusters):
        for c2 in range(number_of_base_clusters):
          if c1 == c2:
            similarities.append(-1)
          else:
            similarities.append(len(set(self.__pruned_C1[c1]) & set(self.__pruned_C1[c2])))

        indices_of_sorted_similarities = sorted(range(len(similarities)), key=lambda k: similarities[k])
        self.__k_most_similar_clusters[c1] = indices_of_sorted_similarities[-self.__k:]
        similarities.clear()
    else:
      raise ValueError("k should not be greater than the number of base clusters('%d')"% len(self.__pruned_C1))



  def compute_best_merge_candidates(self):
    number_base_clusters = len(self.__pruned_C1)
    for i in range(number_base_clusters):
      self.__best_merge_candidates[i] = []
    sorted_combinations = list(combinations(range(number_base_clusters), r=2))
    for combination in sorted_combinations:
      if (len(set(self.__k_most_similar_clusters.get(combination[0], [])) & set(self.__k_most_similar_clusters.get(combination[1], []))) >= self.__mu):
        self.__best_merge_candidates.get(combination[0]).append(combination[1])
        self.__best_merge_candidates.get(combination[1]).append(combination[0])

    for key in list(self.__best_merge_candidates):
      if self.__best_merge_candidates[key] == []:
        self.__best_merge_candidates.pop(key)



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
        if merged_list not in self.__subspace_cluster_approximations:
          self.__subspace_cluster_approximations.append(merged_list.copy())



  def prune_irrelevant_merge_candidate_of_subspace_cluster_approximations(self):

    for index, approximation in enumerate(self.__subspace_cluster_approximations):
      score_approximation = self.compute_quality_of_subspace_cluster_approximation(approximation)
      score_approximation_without_base_cluster = self.compute_quality_of_approximation_without_one_cluster(approximation)
      i = 0
      count_remaining_base_clusters = len(approximation)
      while (i < len(approximation) and len(approximation) > 1):
        if len(approximation) < count_remaining_base_clusters:
          score_approximation = self.compute_quality_of_subspace_cluster_approximation(approximation)
          score_approximation_without_base_cluster = self.compute_quality_of_approximation_without_one_cluster(approximation)
          count_remaining_base_clusters = count_remaining_base_clusters - 1

        l = [x for z,x in enumerate(score_approximation_without_base_cluster) if z!=i]
        if (score_approximation_without_base_cluster[i] > score_approximation
            and all(v <= score_approximation_without_base_cluster[i] for v in l)):
          approximation.pop(i)
        else:
          i = i + 1



  def compute_quality_of_subspace_cluster_approximation(self, approximation):
    cluster_dimensionality = len(approximation)
    base_clusters_of_approximation = [self.__pruned_C1[i] for i in approximation]
    cluster_size = len(set.intersection(*map(set,base_clusters_of_approximation)))
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







