"""import os
import sys

root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)
print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!', sys.path)"""


from ..subclu import subclu
from ...utils import read_sample
from ...samples.definitions import SIMPLE_SAMPLES
from ...samples.generator import generate_subspacedata


def test():
  path = SIMPLE_SAMPLES.SAMPLE_SIMPLE1
  sample = read_sample(path)
  print(sample)
  subclu_instance = subclu(sample, 0.4, 2)
  subclu_instance.process()
  clusters = subclu_instance.get_clusters()
  for key, value in clusters.items():
   print(key, value)






test()
