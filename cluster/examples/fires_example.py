import numpy as np
from itertools import chain
import csv
from subspaceclustering.cluster.fires import *
from subspaceclustering.utils.sample_generator import generate_sample





def cluster_2d_100n_2sc():
  """samples, labels = generate_sample(100, 2, 1, [[range(40), range(0,1), 0.2],
                                               [range(50, 95), range(1, 2), 0.3]])
  save_data(samples, labels, '2d_100n_2sc')"""

  samples = read_data('2d_100n_2sc')
  clustering_method = Clustering_By_dbscan(0.3, 5)
  fires_instance = fires(samples, 1, 1, 2, clustering_method)
  fires_instance.process()
  print_clustering_info(fires_instance)


def cluster_3d_250n_3sc():
  """samples, labels = generate_sample(250, 3, 1, [[tuple(chain(range(50), range(70, 100))), (0,2), 0.3],
                                                [range(120, 190), range(1, 2), 0.3],
                                                [range(200, 235), (0,2), 0.3]])
  save_data(samples, labels, '3d_250n_3sc')"""

  samples = read_data('3d_250n_3sc')
  clustering_method = Clustering_By_dbscan(0.4, 6)
  fires_instance = fires(samples, 1, 1, 1, clustering_method)
  fires_instance.process()
  print_clustering_info(fires_instance)





def cluster_4d_500n_4sc():
  """samples, labels = generate_sample(500, 4, 1, [[range(0,160, 2), (0, 3), 0.3], [range(180, 300), (1,2,3), 0.3],
                                                [range(320, 400), (0, 2), 0.4], [range(425, 475), range(2, 3), 0.4]])
  save_data(samples, labels, '4d_500n_4sc')"""

  samples = read_data('4d_500n_4sc')
  clustering_method = Clustering_By_dbscan(0.3, 6)
  fires_instance = fires(samples, 2, 2, 1, clustering_method)
  fires_instance.process()
  print_clustering_info(fires_instance)



def cluster_32d_4000n_15sc():
  sample, labels = generate_sample(4000, 32, [[(range(200)), (0,1,2), 200, 3, 0.4], [(range(150)), (4,5,6,7),150, 4, 0.3],
                                             [(range(50)), range(11,16),50, 5, 0.6], [(range(240, 340)), (2,3,4,8,9,10),100, 6, 0.5],
                                             [(range(400, 530)), (0,1,5,6,7,8,9),130, 7, 0.5], [(range(590, 710)), range(7,16),120, 9, 0.5],
                                             [(range(820, 930)), tuple(chain(range(7), range(10, 14))),110, 11, 0.8],
                                             [(range(1080, 1180)), tuple(chain(range(3), range(7, 16))),100, 12, 0.4],
                                             [(range(1490,1600)), range(3,16),110, 13, 0.7],
                                             [(range(1860, 1910)), tuple(chain(range(7), range(9, 16))),50, 14, 0.5],
                                             [(range(2000, 2100)), range(15,32), 100, 17, 0.4],
                                             [(range(2500, 2600)), tuple(chain(range(4), range(16,32))), 100, 20, 0.6],
                                             [(range(3000, 3100)), tuple(chain(range(7,16), range(14,29))), 100, 24, 0.5],
                                             [(range(3400, 3500)), tuple(chain(range(11), range(16,32))), 100, 27, 0.7],
                                             [(range(3700, 3800)), range(2,32), 100, 30, 0.3]])
  save_data(sample, labels, '32d_4000n_15sc')

  sample = read_sample("subspaceclustering/samples/32d_4000n_15sc.data")




def cluster_3d_250n_3sc_kmeans():
  sample = read_sample("subspaceclustering/samples/3d_250n_3sc.data")
  clustering_method = cluster_3d_250n_3sc_kmeans()
  fires_instance = fires(sample, 3, 1, 3, clustering_method)
  fires_instance.process()
  print_clustering_info(fires_instance)



def test_with_clique():
  clique_instance = clique(self.__data, self.__intervals, self.__threshold)



def read_data(filename):
  samples = []
  with open('subspaceclustering/samples/'+filename+'.csv', newline='') as csvfile:
      reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
      samples = np.array(list(reader)).astype(float)
  return samples


def save_data(samples, labels, filename):
  with open('subspaceclustering/samples/'+filename+'.csv', 'w', newline='') as csvfile:
      writer = csv.writer(csvfile)
      for row in samples:
          writer.writerow(row)

  with open('subspaceclustering/samples/'+filename+'_labels.csv', 'w', newline='') as csvfile:
      writer = csv.writer(csvfile)
      for row in labels:
          writer.writerow(row)


def print_clustering_info(fires_instance):
  for cluster_index, dimension in fires_instance.get_cluster_to_dimension().items():
    print(cluster_index , dimension)
    print(sorted(fires_instance.get__pruned_C1()[cluster_index]), '\n')

  print(fires_instance.get_k_most_similar_clusters())
  print(fires_instance.get__best_merge_candidates())
  print(fires_instance.get__best_merge_clusters())
  print(fires_instance.get__subspace_cluster_approximations())
  clusters = fires_instance.get_clusters()
  for subspace, list_of_clusters in clusters.items():
   print(subspace, list(map(sorted, list_of_clusters)))

test()
#cluster_2d_100n_2sc()
#cluster_3d_250n_3sc()
#cluster_4d_500n_4sc()
#cluster_3d_250n_3sc_dbscan()
#cluster_4d_500n_4sc()
#cluster_8d_1000n_9sc()
