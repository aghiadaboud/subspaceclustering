import numpy as np
from itertools import chain
from subspaceclustering.cluster.subclu import subclu
from pyclustering.utils import read_sample
from pyclustering.cluster import cluster_visualizer
from pyclustering.cluster import cluster_visualizer_multidim
from subspaceclustering.utils.sample_generator import generate_sample
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn import preprocessing
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns



def cluster_8d_1000n_9sc():
  """sample, labels = generate_sample(1000, 8, [[(range(270)), (1,2,3), 270, 3, 0.6], [(range(50, 150)), (4,5),100, 2, 0.2],
                                             [(range(200, 340)), (5,6,7),140, 3, 0.7], [(range(290, 390)), (1,2),100, 2, 0.3],
                                             [(range(380, 510)), (range(4, 8)),130, 4, 0.9], [(range(520, 640)), (0,1,6),120, 3, 0.5],
                                             [(range(670, 780)), (range(2, 6)),110, 4, 0.6], [(range(880, 950)), (0,1),70, 2, 0.2],
                                             [(range(970, 1000)), (0,3,4,6,7),30, 5, 0.8]])
  scale_and_save_data(sample, labels, '8d_1000n_9sc')"""

  sample = read_sample("subspaceclustering/samples/8d_1000n_9sc.data")
  subclu_instance = subclu(sample, 0.2, 7)
  subclu_instance.process()
  clusters = subclu_instance.get_clusters()


cluster_8d_1000n_9sc()
