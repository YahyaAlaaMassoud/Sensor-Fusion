
import deepdish as dd

from pprint import pprint

loaded_stats = dd.io.load('kitti_stats/stats.h5')

pprint(loaded_stats)
