from SNARS_Bogucki_P10 import my_community_detection
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os
import time

# load data and create networks
files = os.listdir("data/competition")
networks = []
for file in files:
    df = pd.read_csv(f"data/competition/{file}", header=None)
    networks.append(nx.from_numpy_array(df.to_numpy()))

# calculate communities
communities = []
times = []
for net in networks:
    t1 = time.time()
    my_com = my_community_detection(net, seed=124)
    t2 = time.time()
    communities.append(my_com)
    times.append(t2-t1)


nx.draw(networks[5], node_color=communities[5], cmap=plt.get_cmap("Set1"))
plt.show()

# create description file
with open("description.txt", "w") as f:
    f.write("Wojciech Bogucki\n")
    f.write("https://github.com/WojtekBogucki/SNARS-community-detection\n")
    f.write(",".join(["{" +"{0}, {1}".format(file, time)+"}" for file, time in zip(files, times)]))

for file, com in zip(files, communities):
    df = pd.DataFrame(com+1)
    df.index += 1
    df.to_csv(f"results/{file}", header=False)


# tests
my_com = my_community_detection(networks[0], seed=124, resolution=0.85, resolution2=0.7)
nx.draw(networks[0], node_color=my_com, cmap=plt.get_cmap("Set1"))
plt.show()

my_com = my_community_detection(networks[2], seed=124, resolution=1.2, resolution2=1.25)
nx.draw(networks[2], node_color=my_com, cmap=plt.get_cmap("Set1"))
plt.show()