import numpy as np
from itertools import combinations
from more_itertools import locate
from sklearn.cluster import DBSCAN


class subclu:

  def __init__(self, data, eps, m, distance_metric = 'euclidean'):
    self.__data = data
    self.__eps = eps
    self.__m = m
    self.__distance_metric = distance_metric

    self.__S1_indices = []  #set of 1-D subspaces containing clusters
    self.__C1 = {}  # set of all sets of clusters in 1-D subspaces
    self.__clusters = {}
    self.__noise = {}

    self.__verify_arguments()



  def __verify_arguments(self):

    if self.__m < 1:
      raise ValueError("minpts should be a natural number.")

    if len(self.__data) == 0:
      raise ValueError("Input data is empty (size: '%d')." % len(self.__data))

    #if np.min(self.__data) < 0:
      #raise ValueError("All values should be greater or equal to 0.")



  def get_clusters(self):
    return self.__clusters

  def get_noise(self):
    return self.__noise



  def process(self):
    self.generate_all_1_D_clusters()
    self.generate_kplus1_D_clusters_from_k_D_clusters()
    return self


  def generate_all_1_D_clusters(self):
    for dimension in range(len(self.__data[0,:])):
      vector = self.__data[:,dimension].reshape(-1, 1)
      dbscan_instance = DBSCAN(eps = self.__eps, min_samples = self.__m,
                               metric=self.__distance_metric, algorithm='auto',
                               n_jobs=-1).fit(vector)
      labels = dbscan_instance.labels_
      if 0 in labels:   # at least one cluster was found
        self.__S1_indices.append(dimension)
        self.__C1[dimension] = []
        for label in range(len(set(labels)) - (1 if -1 in labels else 0)):
           self.__C1.get(dimension).append(np.where(labels == label)[0])
        self.__noise[dimension] = np.where(labels == -1)[0]
    self.__clusters = self.__C1



  def generate_kplus1_D_clusters_from_k_D_clusters(self):
    k = 1
    Ck = self.__C1.copy()
    Sk = self.__S1_indices.copy()
    while Ck:
      cand_S_k_plus_1 = subclu.generate_candidate_subspaces(Sk, k)
      Sk.clear()
      Ck.clear()
      for cand in cand_S_k_plus_1:
        bestSubspaces = self.find_min_cluster(cand, k)
        c_cand = []
        for bestSubspace in bestSubspaces:
          self.__noise[tuple(cand)] = []
          for cluster in self.__clusters.get(bestSubspace, []):
            points = self.get_points_values_in_subspace(cluster, cand)
            dbscan_instance = DBSCAN(eps = self.__eps, min_samples = self.__m,
                                     metric=self.__distance_metric, algorithm='auto',
                                     n_jobs=-1).fit(points)
            labels = dbscan_instance.labels_
            for label in range(len(set(labels)) - (1 if -1 in labels else 0)):
              c_cand.append(cluster[list(np.where(labels == label)[0])])
            self.__noise.get(tuple(cand)).append(cluster[list(np.where(labels == -1)[0])])
        if c_cand:
          Sk.append(cand.copy())
          Ck[tuple(cand)] = c_cand.copy()

      self.__clusters = self.__clusters | Ck.copy() #change this
      k = k + 1



  @staticmethod
  def generate_candidate_subspaces(Sk, k):
    candSkplus1 = []
    if k == 1:
      sorted_combinations = list(combinations(Sk, r=2))
      candSkplus1 = [list(c) for c in sorted_combinations]
    elif k > 1:
      for s1 in Sk:
        for s2 in Sk:
          if (s1[:-1] == s2[:-1] and s1[-1] < s2[-1]):
            candSkplus1.append(s1.copy() + [s2[-1]])
      subclu.prune_irrelevant_candidates_subspaces(candSkplus1, Sk, k)
    return candSkplus1


  @staticmethod
  def prune_irrelevant_candidates_subspaces(candSkplus1, Sk, k):
    for cand in candSkplus1:
      sorted_combinations = list(combinations(cand, r=k))
      list_combinations = [list(c) for c in sorted_combinations]
      for subspace in list_combinations:
        if subspace not in Sk and cand in candSkplus1:
          candSkplus1.remove(cand)



  def find_min_cluster(self, cand, k, consider_all_bestsubspaces = False):
    cand_combinations = cand.copy()
    if k > 1:
      cand_combinations = list(combinations(cand, r=k))
    size = []
    bestsubspace = ()
    for key in cand_combinations:
      clusters = self.__clusters.get(key, [])
      size.append(sum(map(len, clusters)))

    if consider_all_bestsubspaces == False:
      min_cluster_index = size.index(np.min(size))
      bestsubspace = [cand_combinations[min_cluster_index]]
    else:
      min_clusters_indcies = list(locate(size, lambda a: a == np.min(size)))
      bestsubspace = [cand_combinations[i] for i in min_clusters_indcies]

    return bestsubspace



  def get_points_values_in_subspace(self, points, features):
    return self.__data[np.ix_(points,features)]



