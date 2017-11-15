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
    
    removenodes = []
    for node, degree in graph.degree().items():
       if degree <= 1:
          removenodes.append(node)          
    graph.remove_nodes_from(removenodes)

    return graph

def findMaxEdge(graph):
    betweenness = nx.edge_betweenness_centrality(graph)
    print(max(betweenness.keys()))
    return max(betweenness.keys())

def createComponents(graph):
    components = [c for c in nx.connected_component_subgraphs(graph)]

    while len(components) == 1:
        edge_to_remove = findMaxEdge(graph)
        graph.remove_edge(*edge_to_remove)
        components = [c for c in nx.connected_component_subgraphs(graph)]
    return components

def main():
    users_friends_dict = readFromFile('users_friends.txt')
    graph = createGraph(users_friends_dict)
    components = createComponents(graph)
    # print(len(graph.nodes()), len(graph.edges()))

if __name__ == '__main__':
    main()
