import numpy as np

class RBFNetwork:
    # class of calculate RBF
    def __init__(self, num_nodes=7, input_dim=2, width=np.sqrt(2), center_range=(-2.0, 2.0)):
        """
        :param num_nodes: hidden nodes
        :param input_dim: The dimension of the vector Z
        :param width: pi_l
        """
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.width = width
        np.random.seed(42)
        self.centers = np.random.uniform(center_range[0], center_range[1], (self.num_nodes, self.input_dim))
        
    def compute_basis(self, Z):
        """
        S_i(Z) = exp(- ||Z - \nu_i||^2 / pi^2)
        :param Z: ex. Z1 = [x1, dot_yd])
        :return: num_nodes, )
        """
        # Convert Z to numpy
        Z = np.asarray(Z)
        distances_sq = np.sum((self.centers - Z) ** 2, axis=1)
        S = np.exp(-distances_sq / (self.width ** 2))
        return S

if __name__ == "__main__":
    # Logic test
    print("TEST init")
    rbf_net_1 = RBFNetwork(num_nodes=7, input_dim=2)
    test_Z = [0.18, 0.05]
    S_out = rbf_net_1.compute_basis(test_Z)
    print(f"RBF initialized")
    print(f" Z = {test_Z}")
    print(f" S(Z) = \n{S_out}")
    print(f"The dimension of the output: {S_out.shape} ( {rbf_net_1.num_nodes},)")