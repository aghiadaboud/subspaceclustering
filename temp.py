from pyclustering.utils import read_sample
from pyclustering.samples.definitions import SIMPLE_SAMPLES
import numpy as np
import itertools
import more_itertools
import math
import matplotlib.pyplot as plt
from itertools import chain, groupby, product,zip_longest
from collections import Counter
from functools import reduce
from itertools import combinations, takewhile
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
import time
import timeit
from scipy.optimize import linear_sum_assignment
import csv
from sklearn.decomposition import PCA
import json



class temp:

    def __init__(self):
      self.x = 1

    def visualize_samples(samples):
      """mapper = umap.UMAP().fit(samples.data)
      umap.plot.points(mapper)
      plt.show()"""

      """pca = PCA(n_components=3)
      samples = pca.fit_transform(samples)
      fig = plt.figure()
      ax = fig.add_subplot(projection='3d')
      colours = ['green', 'b', 'r', 'c', 'm', 'y', 'k', 'w', 'orange', 'tomato']
      for indicies in true_clusters.values():
        for cluster in indicies:
          ax.scatter(samples[cluster,0], samples[cluster,1], samples[cluster,2], c = colours.pop())
      plt.show()"""
      pass

c = temp()
c.visualize_samples()





