import numpy as np
from itertools import chain
import csv
from subspaceclustering.cluster.fires import *
from subspaceclustering.utils.sample_generator import generate_samples
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def cluster_D05():
  true_clusters = {
    (0,1,4) : [range(0, 302)],
    (0,1,3,4) : [range(0, 156), range(1008,1158), range(1159, 1309)],
    (1,2,3) : [range(303, 604)],
    (1,2,3,4) : [range(303, 459)],
    (0,2,3) : [range(605, 755), range(1310, 1460)],
    (1,2,4) : [tuple(chain(range(605,655), range(756, 854)))],
    (0,2,3,4) : [range(855, 1007)]
    }
  samples = read_csv('db_dimensionality_scale/D05')
  samples = np.delete(samples, -1, 1)
  clustering_method = Clustering_By_dbscan(1, 7)
  fires_instance = fires(samples, 2, 5, 1, clustering_method)
  fires_instance.process()
  print_clustering_info(fires_instance)


def cluster_D10():
  true_clusters = {
    (1,2,3,5,9) : [range(0, 300)],
    (0,1,2,3,5,6,7,9) : [range(0, 150)],
    (0,2,4,6,9) : [range(301, 602)],
    (0,1,2,3,4,5,6,9) : [range(301, 454)],
    (1,2,4,5,7,8) : [range(603, 753)],
    (0,2,3,4,5,9) : [tuple(chain(range(603,653), range(754, 853)))],
    (1,2,3,4,5,6,7,9) : [range(854, 1003)],
    (0,2,3,4,5,6,8,9) : [range(1004, 1154)],
    (0,1,2,3,6,7,8) : [range(1155, 1305)],
    (0,3,4,6,7,8) : [range(1306,1457)]
    }
  samples = read_csv('db_dimensionality_scale/D10')
  samples = np.delete(samples, -1, 1)
  clustering_method = Clustering_By_dbscan(1, 7)
  fires_instance = fires(samples, 2, 4, 2, clustering_method)
  fires_instance.process()
  print_clustering_info(fires_instance)


def test():
  samples, labels = generate_samples(200, 10, 1.5, 0, [[range(50, 120), range(5), 0.2]])

  """pca = PCA(n_components=3)
  samples = pca.fit_transform(samples)
  fig = plt.figure()
  ax = fig.add_subplot(projection='3d')
  ax.scatter(samples[:,0], samples[:,1], samples[:,2])
  ax.set_xlabel('0')
  ax.set_ylabel('1')
  ax.set_zlabel('2')
  plt.show()"""

  clustering_method = Clustering_By_dbscan(0.2, 6)
  fires_instance = fires(samples, 1, 2, 2, clustering_method)
  fires_instance.process()
  print_clustering_info(fires_instance)



def cluster_2d_100n_2sc():
  samples, labels = generate_samples(100, 2, 1, 0, [[range(40), range(0,1), 0.2],
                                                   [range(50, 95), range(1, 2), 0.3]])
  clustering_method = Clustering_By_dbscan(0.3, 5)
  fires_instance = fires(samples, 1, 1, 2, clustering_method)
  fires_instance.process()
  print_clustering_info(fires_instance)



def cluster_3d_250n_3sc():
  samples, labels = generate_samples(250, 3, 1.5, 0, [[tuple(chain(range(50), range(70, 100))), (0,2), 0.3],
                                                     [range(120, 190), range(1, 2), 0.3],
                                                     [range(200, 235), (0,2), 0.3]])
  clustering_method = Clustering_By_dbscan(0.3, 6)
  fires_instance = fires(samples, 1, 1, 1, clustering_method)
  fires_instance.process()
  print_clustering_info(fires_instance)



def cluster_4d_500n_4sc():
  samples, labels = generate_samples(500, 4, 1, 0, [[range(0,160, 2), (0, 3), 0.3], [range(180, 300), (1,2,3), 0.3],
                                                    [range(320, 400), (0, 2), 0.4], [range(425, 475), range(2, 3), 0.4]])
  clustering_method = Clustering_By_dbscan(0.2, 6)
  fires_instance = fires(samples, 1, 1, 1, clustering_method)
  fires_instance.process()
  print_clustering_info(fires_instance)


def cluster_8d_1000n_7sc():
  samples, labels = generate_samples(1000, 8, 1.5, 2, [[range(250), (1,2,3), 0.25], [range(200, 340), (5,6,7), 0.25],
                                                       [range(290, 390), (1,2), 0.3], [range(380, 510), range(4, 8), 0.25],
                                                       [range(670, 780), range(2, 6), 0.3], [range(800, 890), (0,1), 0.3],
                                                       [range(890, 950), (0,3,4,6,7), 0.3]])
  clustering_method = Clustering_By_dbscan(0.2, 6)
  fires_instance = fires(samples, 3, 4, 1, clustering_method)
  fires_instance.process()
  print_clustering_info(fires_instance)


def cluster_16d_2000n_6sc():
  samples, labels = generate_samples(2000, 16, 2, 0, [[range(150), range(7), 0.2],
                                                      [tuple(chain(range(50), range(200, 250))), range(11,16), 0.2],
                                                      [range(350,450), range(7,11), 0.2], [range(700, 780), range(9), 0.2],
                                                      [range(760, 860), (13,14,15), 0.2], [range(1500,1600), range(2,16), 0.2]])


def read_csv(filename):
  samples = None
  with open('subspaceclustering/samples/'+filename+'.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
    samples = np.array(list(reader)).astype(float)
  return samples


def save_csv(samples, labels, filename):
  with open('subspaceclustering/samples/'+filename+'.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for row in samples:
        writer.writerow(row)

  with open('subspaceclustering/samples/'+filename+'_labels.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for row in labels:
        writer.writerow(row)


def print_clustering_info(fires_instance):
  for cluster_index, dimension in fires_instance.get_baseCluster_to_dimension().items():
    cluster = fires_instance.get_pruned_C1()[cluster_index]
    print('cluster_index ',cluster_index ,'dim ', dimension, len(np.intersect1d(np.arange(50,120), cluster))/len(cluster), 'cluster size ', len(cluster))
    print(sorted(cluster), '\n')

  print('k most similar clusters', '\n', fires_instance.get_k_most_similar_clusters())
  print('best merge candidates', '\n',fires_instance.get_best_merge_candidates())
  print('best merge clusters', '\n',fires_instance.get_best_merge_clusters())
  print('subspace cluster approximations', '\n',fires_instance.get_subspace_cluster_approximations(), '\n')
  clusters = fires_instance.get_clusters()
  for subspace, list_of_clusters in clusters.items():
   print(subspace, list(map(sorted, list_of_clusters)), '\n')

cluster_D05()
#cluster_D10()
#test()
#cluster_2d_100n_2sc()
#cluster_3d_250n_3sc()
#cluster_4d_500n_4sc()
#cluster_8d_1000n_7sc()
