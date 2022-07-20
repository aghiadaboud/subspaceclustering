import numpy as np
from itertools import combinations
from more_itertools import locate

from subspaceclustering.cluster.dbscan import dbscan

class subclu:

  def __init__(self, data, eps, m):
    self.__data = data
    self.__eps = eps
    self.__m = m

    self.__S1_indices = []  #set of 1-D subspaces containing clusters
    self.__C1 = {}  # set of all sets of clusters in 1-D subspaces
    self.__clusters = {}

    self.__verify_arguments()



  def __verify_arguments(self):

    if self.__m < 1:
      raise ValueError("minpts should be a natural number.")

    if len(self.__data) == 0:
      raise ValueError("Input data is empty (size: '%d')." % len(self.__data))

    if np.min(self.__data) < 0:
      raise ValueError("All values should be greater or equal to 0.")



  def get_clusters(self):
    return self.__clusters



  def process(self):
    self.generate_all_1_D_clusters()
    self.generate_kplus1_D_clusters_from_k_D_clusters()
    return self


  def generate_all_1_D_clusters(self):
    for subspace in range(len(self.__data[0])):
      column = [row[subspace] for row in self.__data]
      dbscan_instance = dbscan([[v] for v in column], self.__eps, self.__m)
      dbscan_instance = dbscan_instance.process()
      subspace_clusters = dbscan_instance.get_clusters()
      if subspace_clusters:
        self.__S1_indices.append(subspace)
        self.__C1[subspace] = subspace_clusters.copy()
    self.__clusters = self.__C1.copy()




  def generate_kplus1_D_clusters_from_k_D_clusters(self):
    k = 1
    Ck = self.__C1.copy()
    Sk = self.__S1_indices.copy()
    while Ck:
      cand_S_k_plus_1 = self.generate_candidate_subspaces(Sk, k)
      Sk.clear()
      Ck.clear()
      for cand in cand_S_k_plus_1:
        bestSubspaces = self.find_min_cluster(cand, k)
        c_cand = []
        for bestSubspace in bestSubspaces:
          for cluster in self.__clusters.get(bestSubspace):
            points = self.get_cluster_members_values(cluster, cand)
            dbscan_instance = dbscan(points, self.__eps, self.__m)
            dbscan_instance = dbscan_instance.process()
            for cluster_in_higher_dim in dbscan_instance.get_clusters():
              c_cand.append([cluster[i] for i in cluster_in_higher_dim])
        if c_cand:
          Sk.append(cand.copy())
          Ck[tuple(cand)] = c_cand.copy()

      self.__clusters = self.__clusters | Ck.copy()
      k = k + 1




  def generate_candidate_subspaces(self, Sk, k):
    candSkplus1 = []
    if k == 1:
      sorted_combinations = list(combinations(Sk, r=2))
      candSkplus1 = [list(c) for c in sorted_combinations]
    elif k > 1:
      for s1 in Sk:
        for s2 in Sk:
          if (s1[:-1] == s2[:-1] and s1[-1] < s2[-1]):
            candSkplus1.append(s1.copy() + [s2[-1]])
      self.prune_irrelevant_candidates_subspaces(candSkplus1, Sk, k)
    return candSkplus1



  def prune_irrelevant_candidates_subspaces(self, candSkplus1, Sk, k):
    for cand in candSkplus1:
      sorted_combinations = list(combinations(cand, r=k))
      list_combinations = [list(c) for c in sorted_combinations]
      for subspace in list_combinations:
        if subspace not in Sk:
          candSkplus1.remove(cand)




  def find_min_cluster(self, cand, k, consider_all_bestsubspaces = False):
    cand_combinations = cand.copy()
    if k > 1:
      cand_combinations = list(combinations(cand, r=k))
    size = []
    bestsubspace = ()
    for key in cand_combinations:
      count_objects = 0
      clusters = self.__clusters.get(key)
      for cluster in clusters:
        count_objects = count_objects + len(cluster)
      size.append(count_objects)

    if consider_all_bestsubspaces == False:
      min_cluster_index = size.index(np.min(size))
      bestsubspace = [cand_combinations[min_cluster_index]]
    else:
      min_clusters_indcies = list(locate(size, lambda a: a == np.min(size)))
      bestsubspace = [cand_combinations[i] for i in min_clusters_indcies]

    return bestsubspace



  def get_cluster_members_values(self, points, features):
    temp_data = np.array(self.__data)
    newValues = temp_data[np.ix_(points,features)]
    return newValues.tolist()






