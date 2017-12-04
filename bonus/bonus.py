import networkx as nx
import urllib.request

def download_data():
    urllib.request.urlretrieve('http://cs.iit.edu/~culotta/cs579/a1/edges.txt.gz', 'edges.txt.gz')

def read_graph():
    return nx.read_edgelist('edges.txt.gz', delimiter='\t')



def jaccard_wt(graph, node):
  """
  The weighted jaccard score, defined above.
  Args:
    graph....a networkx graph
    node.....a node to score potential new edges for.
  Returns:
    A list of ((node, ni), score) tuples, representing the 
              score assigned to edge (node, ni)
              (note the edge order)
  """
  score_list = []
  
  nn = graph.neighbors(node)

  unique_nn = set(nn)

  for n in graph.nodes():
    if n not in unique_nn:
      degree_sum_A = 0
      degree_sum_B = 0
      degree_sum_AB = 0
      
      potential_nodes = []
      potential_nodes.append(node)
      potential_nodes.append(n)
      potential_tuple = tuple(potential_nodes)

      degree_sum_A = degree_sum_A + graph.degree(n)

      neighbours_friends = graph.neighbors(n)
      for y in neighbours_friends:
        degree_sum_B = degree_sum_B+(graph.degree(y))

      mutual_friends = list(set(nn).intersection(neighbours_friends))
      for x in mutual_friends:
        degree_sum_AB = degree_sum_AB + (1 / graph.degree(x))

      sim = (degree_sum_AB / ((1 / degree_sum_A) + (1 / degree_sum_B)))

      potential_edges_list = []
      potential_edges_list.append(potential_tuple)
      potential_edges_list.append(sim)

      sim_tuple = tuple(potential_edges_list)
      score_list.append(sim_tuple)
    
  return score_list

  pass

def main():
  download_data()
  graph = read_graph()

  node_deg_dict = {}
  for node in graph.nodes():
    node_deg_dict.update({node: graph.degree(node)})

  max_deg_node = []
  max_deg_node.append(max(node_deg_dict, key=node_deg_dict.get))
  
  score_list = jaccard_wt(graph, max_deg_node[0])
  for score_edge in score_list:
    print(score_edge)

if __name__ == '__main__':
  main()
