import scipy.io as sio
import numpy as np
import skimage.io as skio

from collections import defaultdict

import math
import heapq
 
import networkx as nx
import matplotlib.pyplot as plt


def ProcessPQ(joints, marg, feature_length):
  """
  Populates a priority queue on reversed sorted mutual informations.
  This is used to build the maximum spanning tree.
  """
  pq = []

  for i in range(feature_length):
    for j in range(i+1, feature_length):
      I = 0
      for x_u, p_x_u in marg[i].iteritems():
        for x_v, p_x_v in marg[j].iteritems():
          if (x_u, x_v) in joints[(i, j)]:
            p_x_uv = joints[(i, j)][(x_u, x_v)]
            I += p_x_uv * (math.log(p_x_uv, 2) - math.log(p_x_u, 2) - math.log(p_x_v, 2))
      heapq.heappush(pq, (-I, i, j))
  return pq


def findSet(parent, i):
  while i != parent[i]:
    i = parent[i]

  return i

def buildMST(pq, feature_length):
  """
  Builds the MST using the pq generated above.
  """
  parent = range(feature_length)
  size = [1]*feature_length

  count = 0
  edges = set()
  while count < feature_length-1:
    item = heapq.heappop(pq)
    i = item[1]
    j = item[2]
    seti = findSet(parent, i)
    setj = findSet(parent, j)
    if seti != setj:
      if size[seti] < size[setj]:
        size[setj] += size[seti]
        parent[seti] = setj
      else:
        size[seti] += size[setj]
        parent[setj] = seti
      edges.add((i, j))
      count += 1

  return edges

G2 = None
pos2 = None

def buildVisual(edges, feature_length, labels, fname, title=None):
  """
  Graphs the tree and saves the figures.
  """
  global G2
  global pos2

  if type(G2) == type(None):
    G = nx.Graph()
    for i in range(feature_length):
      G.add_node(i)
    pos = nx.spring_layout(G, k=10., scale = 10)
    G2 = G
    pos2 = pos
  else:
    G = G2
    pos = pos2

  nx.draw_networkx_nodes(G, pos, node_size=1000)

  nx.draw_networkx_labels(G, pos,labels,font_size=8)
  nx.draw_networkx_edges(G, pos, edgelist=list(edges))
  if title:
    plt.title(title)
  plt.savefig(fname)
  plt.close()

labels = {0: "Age",
          1: "Workclass",
          2: "education",
          3: "education-num",
          4: "marital-status",
          5: "occupation",
          6: "relationships",
          7: "race",
          8: "sex",
          9: "capital-gain",
          10: "capital-loss",
          11: "hours-per-week",
          12: "native-country",
          13: "salary",
         }

f = open("data.csv", "r")
joints = {}
marg = {}

feature_length = 14
data_size = 32561

for i in range(feature_length):
  marg[i] = defaultdict(float)

  for j in range(i+1, feature_length):
    joints[(i, j)] = defaultdict(float)

count_aggr = 0

for line in f:
  n = line.strip().split(",")
  # Delete the second element because it's some weird finalweight sample
  del n[2]
  count_aggr += 1
  for i in range(feature_length):
    marg[i][n[i]] += 1./data_size

    for j in range(i+1, feature_length):
      joints[(i,j)][(n[i], n[j])] += 1./data_size

  if count_aggr%1000 == 10:
    pq = ProcessPQ(joints, marg, feature_length)
    edges = buildMST(pq, feature_length)
    fname = ("graphs/%d.jpg")%count_aggr
    buildVisual(edges, feature_length, labels, fname, title="%d samples"%count_aggr)

pq = ProcessPQ(joints, marg, feature_length)
edges = buildMST(pq, feature_length)
fname = ("graphs/%d.jpg")%count_aggr
buildVisual(edges, feature_length, labels, "graphs/final.jpg", title="%d samples"%data_size)
