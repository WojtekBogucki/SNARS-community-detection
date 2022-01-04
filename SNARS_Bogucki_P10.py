import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import copy

# test_net = nx.stochastic_block_model([5, 5], [[0.7, 0.05], [0.05, 0.7]], seed=123)


def modularity_gain2(G, communities, i, new_com, weight=None):
    new_communities = copy.deepcopy(communities)
    A = nx.to_numpy_array(G, weight=weight)
    C = np.where(new_communities == new_com)[0]
    m = A.sum() / 2
    sum_tot = A[C].sum()
    k_i = G.degree(i)
    k_i_in = A[np.ix_([i], C)].sum()
    denom = 2 * m
    print(k_i, k_i_in, sum_tot)
    delta_q = k_i_in/m - (2*sum_tot*k_i)/denom**2
    C2 = np.where(new_communities == communities[i])[0]
    sum_tot2 = A[C2].sum()
    k_i_in2 = A[np.ix_([i], C2)].sum()
    delta_q2 = -k_i_in2 / m + (2 * sum_tot2 * k_i) / denom ** 2
    return delta_q, delta_q2

def transform_comm(communities):
    df = pd.DataFrame({'com': communities})
    gg = df.groupby(by=df.loc[:, 'com'], as_index=False)
    comm = [i.tolist() for i in list(gg.groups.values())]
    return comm


def extract_comm(communities_t, n):
    new_communities = np.zeros(n, dtype='int')
    for i, com in enumerate(communities_t):
        for j, com_node in enumerate(com):
            new_communities[com_node] = int(i)
    return new_communities


def translate_comm(old_communities, new_communities):
    comm = copy.deepcopy(old_communities)
    for i, new_c in enumerate(new_communities):
        for j, new_c_node in enumerate(new_c):
            comm[old_communities==new_c_node] = i
    return comm


def modularity_gain(G, communities, i, new_com, weight=None):
    communities = copy.deepcopy(communities)
    comm = transform_comm(communities)
    old_mod = nx.community.modularity(G, comm, weight=weight)
    communities[i] = new_com
    comm2 = transform_comm(communities)
    new_mod = nx.community.modularity(G, comm2, weight=weight)
    return new_mod - old_mod


def my_community_detection_pass(G, communities, weight=None):
    nodes = list(G)
    # phase 1
    while True:
        change = False
        nodes_perm = np.random.permutation(nodes)
        for i in nodes_perm:
            i_neighbors = [neigh for neigh in list(G.neighbors(i)) if communities[neigh] != communities[i] and neigh != i]
            mod_gains = []
            for neigh in i_neighbors:
                mod_gains.append(modularity_gain(G, communities, i, communities[neigh], weight=weight))
            if not len(mod_gains): continue
            mod_gains = np.array(mod_gains)
            max_mod_gain = mod_gains.max()
            if max_mod_gain > 0 + 1e-10:
                new_com_idx = np.random.choice(np.where(mod_gains == max_mod_gain)[0], 1)[0]
                new_com = i_neighbors[new_com_idx]
                communities[i] = communities[new_com]
                change = True
            # print(communities, max_mod_gain, nx.community.modularity(G, transform_comm(communities)))
        if not change: break
    # phase 2
    A = nx.to_numpy_array(G)
    communities = transform_comm(communities)
    n_com = len(communities)
    new_adj = np.zeros((n_com, n_com))
    for i in range(n_com):
        for j in range(n_com):
            ci = communities[i]
            if i != j:
                cj = communities[j]
                new_adj[i, j] = A[np.ix_(ci, cj)].sum()
            else:
                new_adj[i, j] = A[np.ix_(ci, ci)].sum()/2
    new_G = nx.from_numpy_array(new_adj)
    return new_G, communities


def my_community_detection(G, weight=None):
    n = nx.number_of_nodes(G)
    communities = np.arange(n)
    mod = nx.community.modularity(G, transform_comm(communities))
    new_g = G.copy()
    initial_com = copy.deepcopy(communities)
    while True:
        new_g, communities_t = my_community_detection_pass(new_g, communities, weight=weight)
        initial_com = translate_comm(initial_com, communities_t)
        new_mod = nx.community.modularity(G, transform_comm(initial_com))
        print(mod, new_mod)
        if new_mod > mod:
            mod = new_mod
        else:
            break
        n = nx.number_of_nodes(new_g)
        communities = np.arange(n)
    return initial_com

# test_g, my_comm = my_community_detection_pass(test_net, np.arange(len(test_net)))
test_net = nx.karate_club_graph()
init_com = my_community_detection(test_net)

nx.draw(test_net, with_labels=True, node_color=init_com, cmap=plt.get_cmap("Set1"))
plt.show()

test_net2 = nx.stochastic_block_model([10,10, 10],[[0.7,0.05, 0.1],[0.05,0.7, 0.2],[0.1, 0.2,0.7]], seed=123)
init_com2 = my_community_detection(test_net2)

nx.draw(test_net2, with_labels=True, node_color=init_com2, cmap=plt.get_cmap("Set1"))
plt.show()