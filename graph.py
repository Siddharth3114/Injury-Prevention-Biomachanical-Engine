import numpy as np

class Graph():
    """
    Constructs the spatial graph based on MediaPipe's 33 pose landmarks.
    """
    def __init__(self):
        self.num_node = 33
        self.edges = self.get_edge()
        
        # Adjacency matrices
        self.A = self.get_adjacency(self.edges, self.num_node)
        
    def get_edge(self):
        # Define the connections based on MediaPipe pose topology.
        # Format: (source, target)
        
        edges = [
            # Face/Head (0-10)
            (0, 1), (1, 2), (2, 3), (3, 7), # Left eye/ear
            (0, 4), (4, 5), (5, 6), (6, 8), # Right eye/ear
            (9, 10), # Mouth
            
            # Torso connection to face
            (11, 0), (12, 0),
            
            # Torso (11, 12, 23, 24)
            (11, 12), # shoulders
            (23, 24), # hips
            (11, 23), # left torso
            (12, 24), # right torso
            
            # Left Arm (11, 13, 15, 17, 19, 21)
            (11, 13), # Left shoulder -> elbow
            (13, 15), # Left elbow -> wrist
            (15, 17), # wrist -> left pinky
            (15, 19), # wrist -> left index
            (15, 21), # wrist -> left thumb
            (17, 19),
            
            # Right Arm (12, 14, 16, 18, 20, 22)
            (12, 14), # Right shoulder -> elbow
            (14, 16), # Right elbow -> wrist
            (16, 18), # wrist -> right pinky
            (16, 20), # wrist -> right index
            (16, 22), # wrist -> right thumb
            (18, 20),
            
            # Left Leg (23, 25, 27, 29, 31)
            (23, 25), # hip -> knee
            (25, 27), # knee -> ankle
            (27, 29), # ankle -> heel
            (27, 31), # ankle -> foot index
            (29, 31),
            
            # Right Leg (24, 26, 28, 30, 32)
            (24, 26), # hip -> knee
            (26, 28), # knee -> ankle
            (28, 30), # ankle -> heel
            (28, 32), # ankle -> foot index
            (30, 32)
        ]
        
        # Make edges bidirectional for undirected graph
        bidirectional_edges = []
        for i, j in edges:
            bidirectional_edges.append((i, j))
            bidirectional_edges.append((j, i))
            
        # Add self-loops
        for i in range(self.num_node):
            bidirectional_edges.append((i, i))
            
        return bidirectional_edges
        
    def get_adjacency(self, edges, num_node):
        """
        Creates the adjacency matrix and applies spatial configuration partitioning.
        In standard ST-GCN, nodes are partitioned into:
        0: root node itself
        1: neighborhood closer to gravity center (here, average of shoulders and hips, conceptually we can use proximity to node 0 or center)
        2: neighborhood further from gravity center
        
        For simplicity in this initial blueprint, we will use a normalized static adjacency matrix A.
        A (1, V, V) where V = 33
        """
        A = np.zeros((num_node, num_node), dtype=np.float32)
        for i, j in edges:
            A[i, j] = 1.0

        # Normalize the adjacency matrix
        # D^(-1/2) * A * D^(-1/2)
        node_degrees = A.sum(axis=1)
        
        # Prevent division by zero
        node_degrees[node_degrees == 0] = 1.0
        
        D_inv_sqrt = np.diag(np.power(node_degrees, -0.5))
        A_normalized = np.dot(np.dot(D_inv_sqrt, A), D_inv_sqrt)
        
        # We wrap it in an extra dimension to represent the "spatial configuration" partition.
        # Since we use a simple normalized graph for now, the partition size is 1.
        # A.shape = (1, V, V)
        A_final = np.expand_dims(A_normalized, axis=0)
        
        return A_final
