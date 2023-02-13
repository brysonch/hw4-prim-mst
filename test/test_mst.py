import pytest
import numpy as np
from mst import Graph
from sklearn.metrics import pairwise_distances


def check_mst(adj_mat: np.ndarray, 
              mst: np.ndarray, 
              expected_weight: int, 
              allowed_error: float = 0.0001):
    """
    
    Helper function to check the correctness of the adjacency matrix encoding an MST.
    Note that because the MST of a graph is not guaranteed to be unique, we cannot 
    simply check for equality against a known MST of a graph. 

    Arguments:
        adj_mat: adjacency matrix of full graph
        mst: adjacency matrix of proposed minimum spanning tree
        expected_weight: weight of the minimum spanning tree of the full graph
        allowed_error: allowed difference between proposed MST weight and `expected_weight`

    TODO: Add additional assertions to ensure the correctness of your MST implementation. For
    example, how many edges should a minimum spanning tree have? Are minimum spanning trees
    always connected? What else can you think of?

    """

    def approx_equal(a, b):
        return abs(a - b) < allowed_error

    total = 0
    for i in range(mst.shape[0]):
        for j in range(i+1):
            total += mst[i, j]
    assert approx_equal(total, expected_weight), 'Proposed MST has incorrect expected weight'

    # See test_mst_student() for additional assertions
    
    assert (np.count_nonzero(mst) / 2) + 1 == adj_mat.shape[0], 'Proposed MST has incorrect number of edges for the given graph'
    assert mst.shape[0] == mst.shape[1], 'MST should be an n x n matrix'
    assert mst.shape == adj_mat.shape, 'MST should include the same number of nodes as the adjacency matrix'


def test_mst_small():
    """
    
    Unit test for the construction of a minimum spanning tree on a small graph.
    
    """
    file_path = './data/small.csv'
    g = Graph(file_path)
    g.construct_mst()
    check_mst(g.adj_mat, g.mst, 8)


def test_mst_single_cell_data():
    """
    
    Unit test for the construction of a minimum spanning tree using single cell
    data, taken from the Slingshot R package.

    https://bioconductor.org/packages/release/bioc/html/slingshot.html

    """
    file_path = './data/slingshot_example.txt'
    coords = np.loadtxt(file_path) # load coordinates of single cells in low-dimensional subspace
    dist_mat = pairwise_distances(coords) # compute pairwise distances to form graph
    g = Graph(dist_mat)
    g.construct_mst()
    check_mst(g.adj_mat, g.mst, 57.263561605571695)


def test_mst_student():
    """
    
    TODO: Write at least one unit test for MST construction.
    
    """
    file_path_empty = './data/empty.csv'
    g_empty = Graph(file_path_empty)

    rand_adjmat = np.arange(3, 35, 2)
    rand_adjmat = np.reshape(rand_adjmat,(4,4))
    g_rand = Graph(rand_adjmat)

    neg_adjmat = np.arange(-35, -3, 2)
    neg_adjmat = np.reshape(neg_adjmat,(4,4))
    g_neg = Graph(neg_adjmat)

    disconnected_adjmat = np.zeros((4,4))
    disconnected_adjmat[0][1] = 8
    disconnected_adjmat[1][0] = 8
    g_disc = Graph(disconnected_adjmat)

    try:
        g_empty.construct_mst()
        test_empty = False
    except:
        test_empty = True

    try:
        g_rand.construct_mst()
        test_rand = False
    except:
        test_rand = True

    try:
        g_nonpos.construct_mst()
        test_neg = False
    except:
        test_neg = True

    try:
        g_disc.construct.mst()
        disc_test = False
    except:
        disc_test = True


    assert test_empty == True, "Empty graph should not return an MST"
    assert test_rand == True, "Random adjacency matrix with self-edges is not a valid graph"
    assert test_neg == True, "All edge weights must be positive values"
    assert disc_test == True, "The input adjacency matrix must be symmetric and a connected graph"

