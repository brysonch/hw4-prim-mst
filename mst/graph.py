import numpy as np
import heapq
from typing import Union

class Graph:

    def __init__(self, adjacency_mat: Union[np.ndarray, str]):
        """
    
        Unlike the BFS assignment, this Graph class takes an adjacency matrix as input. `adjacency_mat` 
        can either be a 2D numpy array of floats or a path to a CSV file containing a 2D numpy array of floats.

        In this project, we will assume `adjacency_mat` corresponds to the adjacency matrix of an undirected graph.
    
        """
        if type(adjacency_mat) == str:
            self.adj_mat = self._load_adjacency_matrix_from_csv(adjacency_mat)
        elif type(adjacency_mat) == np.ndarray:
            self.adj_mat = adjacency_mat
        else: 
            raise TypeError('Input must be a valid path or an adjacency matrix')
        self.mst = None

    def _load_adjacency_matrix_from_csv(self, path: str) -> np.ndarray:
        with open(path) as f:
            return np.loadtxt(f, delimiter=',')

    def construct_mst(self):
        """
    
        TODO: Given `self.adj_mat`, the adjacency matrix of a connected undirected graph, implement Prim's 
        algorithm to construct an adjacency matrix encoding the minimum spanning tree of `self.adj_mat`. 
            
        `self.adj_mat` is a 2D numpy array of floats. Note that because we assume our input graph is
        undirected, `self.adj_mat` is symmetric. Row i and column j represents the edge weight between
        vertex i and vertex j. An edge weight of zero indicates that no edge exists. 
        
        This function does not return anything. Instead, store the adjacency matrix representation
        of the minimum spanning tree of `self.adj_mat` in `self.mst`. We highly encourage the
        use of priority queues in your implementation. Refer to the heapq module, particularly the 
        `heapify`, `heappop`, and `heappush` functions.

        """
        #self.mst = None
        self.mst = np.zeros(self.adj_mat.shape)

        if not self.adj_mat: raise Exception("Input was an empty graph")
        if np.any(self.adj_mat.diagonal()): raise Exception("There should be no self-edges in the graph")
        if np.any(self.adj_mat < 0): raise Exception("All edge weights in the graph must be positive values")
        if np.any(self.adj_mat != self.adj_mat.T): raise Exception("The input graph must be a symmetric adjacency matrix")

        # Initialize a list of visited nodes, priority queue of edge weights, and append vertex 0 to the visited list
        visited = []
        visited.append(0)
        h = []

        # Start with the first row of the adjacency matrix, row_ind indicates the row number (starting node) of the adjacency matrix
        # Remove all zeros from the row (no edges between those vertices)
        row = self.adj_mat[0,:]
        row_ind = 0
        insert = row[np.nonzero(row)]
        
        # Push the first set of edge weights from vertex 0 to the priority queue
        for i in insert: 
            heapq.heappush(h, i)
        
        # Loop until we have visited all nodes in the graph
        while len(visited) < self.adj_mat.shape[0]:

            # Pop the lowest edge weight and find the destination node of that edge
            # Although multiple identical MSTs may exist, we'll take the first destination node with that (unique) edge weight
            lowest = heapq.heappop(h)
            dest = np.where(row == lowest)[0][0]

            # If the destination node has not been visited, add that lowest edge weight to the mst
            # The row index becomes the destination node, and we visit that node
            # The new row becomes the desintation node's row index of the adjacency matrix, push outgoing edges to the priority queue
            # If the desintation node has already been visited, then go back to the row containing the lowest edge weight from the priority queue
            if dest not in visited:
                self.mst[row_ind,dest] = lowest
                row_ind = dest
                visited.append(dest)
                
                row = self.adj_mat[dest,:]
                insert = row[np.nonzero(row)]
                for i in insert: 
                    heapq.heappush(h, i)
            else: 
                row = self.adj_mat[np.where(self.adj_mat == lowest)[0][0]]
                row_ind = np.where(self.adj_mat == lowest)[0][0]

            if (not h) and (len(visited) < self.adj_mat.shape[0]): raise Exception("The input graph is disconnected")

