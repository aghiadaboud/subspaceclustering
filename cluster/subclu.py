import numpy as np
from itertools import combinations
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

class subclu:
  """
  Class represents clustering algorithm SUBCLU.
  For clustering example please check '/examples/subclu_example.py'.
  """

  def __init__(self, data, eps, m):
    """
    Constructor of clustering algorithm SUBCLU.

    Parameters
    ----------
    data (ndarray): Input data.

    eps (float): Connectivity radius between points. For DBSCAN algorithm.

    m (int): Minimum number of samples in a neighborhood for a point to be
            considered as a core point. This includes the point itself.
            For DBSCAN algorithm.
    """

    self.__data = data
    self.__eps = eps
    self.__m = m

    self.__S1_indices = []  #list of 1-D subspaces containing clusters
    self.__C1 = {}  #1-D subspaces to corresponding 1-D clusters
    self.__clusters = {}  #clustering(subspaces to clusters)
    self.__noise = {} #subspaces to noise

    self.__verify_arguments()



  def __verify_arguments(self):
    """
    Verifies input parameters for the algorithm.
    """

    if self.__data is None or self.__data.size == 0:
      raise ValueError("Input data is empty.")

    if self.__data.shape[1] < 2:
      raise ValueError("Please provide multivariate data.")

    if self.__m < 1:
      raise ValueError("minpts should be a natural number.")

    if self.__eps <= 0:
      raise ValueError("epsilon should be a greater than 0.")

    #if np.min(self.__data) < 0:
      #raise ValueError("All values should be greater or equal to 0.")



  def get_clusters(self):
    """
    Returns a dictionary whereby keys are subspaces and values are detected
    clusters.
    Keys are tuples and values are lists.
    """

    return self.__clusters


  def get_noise(self):
    """
    Returns a dictionary whereby keys are subspaces and values are noise.
    Keys are tuples and values are lists.
    """

    return self.__noise



  def process(self):
    """
    Performs cluster analysis in line with rules of SUBCLU algorithm and
    returns a SUBCLU instance(itself).
    """

    self.generate_all_1_D_clusters()
    self.generate_kplus1_D_clusters_from_k_D_clusters()
    return self



  def generate_all_1_D_clusters(self):
    """
    Generates 1-dimensional subspace clusters by applying DBSCAN to each
    1-dimensional subspace and computes the list of all 1-dimensional subspaces
    containing clusters.
    """

    for dimension in range(self.__data.shape[1]):
      vector = self.__data[:,dimension].reshape(-1, 1)
      dbscan_instance = DBSCAN(eps = self.__eps, min_samples = self.__m,
                               metric='euclidean', algorithm='auto',
                               n_jobs=-1).fit(vector)
      labels = dbscan_instance.labels_
      if 0 in labels:   # at least one cluster was found
        self.__S1_indices.append(dimension)
        self.__C1[dimension] = []
        for label in np.setdiff1d(labels, -1):
           self.__C1.get(dimension).append(np.where(labels == label)[0])
        self.__noise[dimension] = np.where(labels == -1)[0]
    self.__clusters = self.__C1



  def generate_kplus1_D_clusters_from_k_D_clusters(self):
    """
    Generates the (k+1)-dimensional clusters and the corresponding
    (k+1)-dimensional subspaces containing these clusters using the
    k-dimensional subclusters and the list of (k+1)-dimensional candidate
    subspaces.
    """

    Sk = None
    k = 1
    cand_S_k_plus_1 = subclu.generate_candidate_subspaces(self.__S1_indices, k)
    while Sk != []:
      Sk = []
      for cand in cand_S_k_plus_1:
        bestSubspace = self.find_bestsubspace(cand, k)
        c_cand = []
        self.__noise[cand] = []
        for cluster in self.__clusters.get(bestSubspace, []):
          points = self.get_points_values_in_subspace(cluster, cand)

          neigh = NearestNeighbors(n_neighbors=1, radius=self.__eps, metric='euclidean', n_jobs=-1)
          neigh.fit(points)
          distances = neigh.radius_neighbors_graph(points, mode='distance', sort_results=True)

          dbscan_instance = DBSCAN(eps = self.__eps, min_samples = self.__m,
                                   metric='precomputed', algorithm='auto',
                                   n_jobs=-1).fit(distances)

          labels = dbscan_instance.labels_
          for label in np.setdiff1d(labels, -1):
            c_cand.append(cluster[list(np.where(labels == label)[0])])
          self.__noise.get(cand).append(cluster[list(np.where(labels == -1)[0])])

        if c_cand:
          Sk.append(cand)
          self.__clusters[cand] = c_cand
      k = k + 1
      cand_S_k_plus_1 = subclu.generate_candidate_subspaces(Sk, k)


  @staticmethod
  def generate_candidate_subspaces(Sk, k):
    """
    Generates (k+1)-dimensional candidate subspaces by joining k-dimensional
    subspaces having (k-1) features in common.

    Parameters
    ----------
    Sk (list): k-dimensional subspaces containing clusters.

    k (int): Dimensionality of subspaces within Sk.
    """

    cand_S_kplus1 = []
    if k == 1:
      cand_S_kplus1 = list(combinations(Sk, r=2))
    elif k > 1:
      for i, s1 in enumerate(Sk):
        for s2 in Sk[i+1:]:
          if (s1[:-1] != s2[:-1]):
            break
          else:
            cand_S_kplus1.append(s1 + (s2[-1],))
      subclu.prune_irrelevant_candidates_subspaces(cand_S_kplus1, Sk, k)
    return cand_S_kplus1



  @staticmethod
  def prune_irrelevant_candidates_subspaces(cand_S_kplus1, Sk, k):
    """
    Removes irrelevant (k+1)-dimensional subspaces from cand_S_kplus1 in place
    if any k-dimensional subspace ⊂ (k+1)-dimensional subspaces contains no
    clusters.

    Parameters
    ----------
    cand_S_kplus1 (list): (k+1)-dimensional candidate subspaces.

    Sk (list): k-dimensional subspaces containing clusters.

    k (int): Dimensionality of subspaces within Sk.
    """

    i = 0
    while i < len(cand_S_kplus1):
      if any(x not in Sk for x in list(combinations(cand_S_kplus1[i], r=k))):
        del cand_S_kplus1[i]
        i = i - 1
      i = i + 1



  def find_bestsubspace(self, cand, k):
    """
    Searches k-dimensional subspace of cand with minimal number of objects in
    the clusters.

    Parameters
    ----------
    cand (tuple): (k+1)-dimensional candidate subspace.

    k (int): Dimensionality of the searched subspace.
    """

    cand_combinations = cand
    if k > 1:
      cand_combinations = list(combinations(cand, r=k))
    size = []
    bestsubspace = ()
    for key in cand_combinations:
      clusters = self.__clusters.get(key, [])
      size.append(sum(map(len, clusters)))
    min_cluster_index = np.argmin(size)
    bestsubspace = cand_combinations[min_cluster_index]
    return bestsubspace



  def get_points_values_in_subspace(self, points, features):
    """
    Extracts the projection of objects into a specific subspace.

    Parameters
    ----------
    points (array): Indices of the objects.

    features (tuple): Indices of the dimensions forming the subspace.
    """

    return self.__data[np.ix_(points,features)]



