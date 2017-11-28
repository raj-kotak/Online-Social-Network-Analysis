"""
cluster.py
"""
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import networkx as nx
import sys
import time


def readFromFile(filename):
    users_friends_dict = defaultdict(lambda:0)
    f = open(filename, 'r+')
    data = f.readlines()
    for d in data:
        split_list = d.split(':')
        split_list[1] = split_list[1].strip(" ")
        split_list[1] = split_list[1].replace("[","")
        split_list[1] = split_list[1].replace("]","")
        split_list[1] = split_list[1].replace(" ","")
        split_list[1] = split_list[1].split(",")
        users_friends_dict[split_list[0]] = split_list[1]
    return users_friends_dict

def createGraph(users_friends_dict):
    graph = nx.Graph()
    for user, friends in users_friends_dict.items():
        graph.add_node(user)
        for friend in friends:
            graph.add_edge(friend, user)

    return graph

def girvan_newman(G, depth=0):

    if G.order() == 1:
        return [G.nodes()]
    
    def find_best_edge(G0):
        eb = nx.edge_betweenness_centrality(G0)
        return sorted(eb.items(), key=lambda x: x[1], reverse=True)[0][0]

    components = [c for c in nx.connected_component_subgraphs(G)]
    indent = '   ' * depth  # for printing
    while len(components) == 1:
        edge_to_remove = find_best_edge(G)
        G.remove_edge(*edge_to_remove)
        components = [c for c in nx.connected_component_subgraphs(G)]

    result = [c.nodes() for c in components]

    return result

def main():
    users_friends_dict = readFromFile('users_friends.txt')
    graph = createGraph(users_friends_dict)
    components = girvan_newman(graph)
    total_users = 0
    for c in components:
        total_users += len(c)
    f = open('clusters.txt', 'w+')
    f.write("Number of communities discovered: "+str(len(components))+"\n")
    f.write("Average number of users per community: "+str(total_users/len(components))+"\n")
    f.close()

if __name__ == '__main__':
    main()
