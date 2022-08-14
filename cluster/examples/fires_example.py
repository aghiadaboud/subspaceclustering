import numpy as np
from subspaceclustering.cluster.fires import fires
from subspaceclustering.cluster.fires import *
from pyclustering.utils import read_sample
from subspaceclustering.utils.sample_generator import generate_sample
from sklearn import preprocessing
import pandas as pd
from sklearn.datasets import make_blobs




def cluster_3d_250n_3sc_dbscan():
  """sample, labels = generate_sample(250, 3, [[tuple(chain(range(50), range(70, 100))), (0,2), 80, 2, 0.6],
                                            [(range(130, 220)), (range(1, 2)), 90, 1, 0.1],
                                            [range(230, 250), (0,2), 20, 2, 0.3]])
  scale_and_save_data(sample, labels, '3d_250n_3sc')"""

  sample = read_sample("subspaceclustering/samples/3d_250n_3sc.data")
  clustering_method = Clustering_By_dbscan(0.1, 5)
  fires_instance = fires(sample, 2, 2, 1, clustering_method)
  fires_instance.process()
  print_clustering_info(fires_instance)



def cluster_4d_500n_4sc():
  """sample, labels = generate_sample(500, 4, [[(range(0,160, 2)), (0, 3), 80, 2, 0.3], [(range(180, 300)), (1,2,3), 120, 3, 0.9],
                                            [(range(350, 400)), (0, 2), 50, 2, 0.6], [(range(450, 500)), range(2, 3), 50, 1, 0.1]])
  scale_and_save_data(sample, labels, '4d_500n_4sc')"""
  #note:last cluster is not found
  sample = read_sample("subspaceclustering/samples/4d_500n_4sc.data")
  clustering_method = Clustering_By_dbscan(0.1, 6)
  fires_instance = fires(sample, 2, 2, 1, clustering_method)
  fires_instance.process()
  print_clustering_info(fires_instance)



def cluster_8d_1000n_9sc():
  """sample, labels = generate_sample(1000, 8, [[(range(270)), (1,2,3), 270, 3, 0.6], [(range(50, 150)), (4,5),100, 2, 0.2],
                                             [(range(200, 340)), (5,6,7),140, 3, 0.7], [(range(290, 390)), (1,2),100, 2, 0.3],
                                             [(range(380, 510)), (range(4, 8)),130, 4, 0.9], [(range(520, 640)), (0,1,6),120, 3, 0.5],
                                             [(range(670, 780)), (range(2, 6)),110, 4, 0.6], [(range(880, 950)), (0,1),70, 2, 0.2],
                                             [(range(970, 1000)), (0,3,4,6,7),30, 5, 0.8]])
  scale_and_save_data(sample, labels, '8d_1000n_9sc')"""

  sample = read_sample("subspaceclustering/samples/8d_1000n_9sc.data")
  clustering_method = Clustering_By_dbscan(0.2, 7)
  fires_instance = fires(sample, 2, 2, 4, clustering_method)
  fires_instance.process()
  print_clustering_info(fires_instance)




def cluster_3d_250n_3sc_kmeans():
  sample = read_sample("subspaceclustering/samples/3d_250n_3sc.data")
  clustering_method = cluster_3d_250n_3sc_kmeans()
  fires_instance = fires(sample, 1, 2, 2, clustering_method)
  fires_instance.process()
  print_clustering_info(fires_instance)



def test_with_clique():
  clique_instance = clique(self.__data, self.__intervals, self.__threshold)



def scale_and_save_data(data, labels, name):

  labels_filepath = "subspaceclustering/samples/"+name+"_labels.csv"
  pd.DataFrame(labels).to_csv(labels_filepath)
  scaler = preprocessing.StandardScaler().fit(data)
  sample_scaled = scaler.transform(data)
  data_filepath = "subspaceclustering/samples/"+name+".data"
  file = open(data_filepath, "w+")
  for i in sample_scaled:
    for f in i:
     file.write("%f " % f)
    file.write("\n")
  file.close()



def print_clustering_info(fires_instance):
  print(list(map(sorted, fires_instance.get__pruned_C1())))
  print(fires_instance.get_cluster_to_dimension())
  print(fires_instance.get_k_most_similar_clusters())
  print(fires_instance.get__best_merge_candidates())
  print(fires_instance.get__best_merge_clusters())
  print(fires_instance.get__subspace_cluster_approximations())
  clusters = fires_instance.get_clusters()
  for subspace, list_of_clusters in clusters.items():
   print(subspace, list(map(sorted, list_of_clusters)))


test()
#cluster_3d_250n_3sc_dbscan()
#cluster_4d_500n_4sc()
#cluster_8d_1000n_9sc()
