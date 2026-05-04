import numpy as np
from rbf_nn import RBFNetwork

class CSTRController:
    def __init__(self, delta=0.3):
        # refrigerant
        self.delta = delta
        """
         1. Controller gain parameters 
        ( the larger *k*₁ and *k*₂ are, the faster the tracking speed; 
        however, it tends to induce oscillatory behaviour).
        """
        self.k1 = 5.0
        self.k2 = 10.0
        
        # Learning rate and robustness correction coefficient of NN
        self.Gamma1 = 15.0  # learning rate W1
        self.Gamma2 = 15.0  # learning rate W2
        self.sigma1 = 0.05  # Forgetting factor of W1
        self.sigma2 = 0.05  # Forgetting factor of W2
        
        # Two RBF NNs instantiation
        self.rbf1 = RBFNetwork(num_nodes=7, input_dim=2, center_range=(-1.5, 1.5))
        # Temp. x2 varies over a wide range
        self.rbf2 = RBFNetwork(num_nodes=7, input_dim=3, center_range=(-5.0, 5.0))

    def compute(self, x1, x2, yd, dot_yd, W1, W2, kb, dot_kb=0.0):
        """
        :param x1, x2: conc. and temp.
        :param yd, dot_yd: reference signal and its derivative
        :param W1, W2: NN weight vector
        :param kb: Time-varying constraint limit kb(t)
        :param dot_kb: Time derivative
        :return: (u, dW1_dt, dW2_dt, alpha1)
        """
        # Step 1
        # 1. Tracking error
        z1 = x1 - yd
        
        # 2. Safeguarding of BLF
        eps = 1e-6
        if abs(z1) >= kb:
            z1 = np.sign(z1) * (kb - eps)
        v1 = z1 / (kb**2 - z1**2)
        
        # 3. NN1
        Z1 = np.array([x1, yd])
        S1 = self.rbf1.compute_basis(Z1)
        f_hat_1 = np.dot(W1, S1)
        
        # 4. alpha1
        alpha1 = -self.k1 * v1 - f_hat_1 + dot_yd + (dot_kb / kb) * z1
        
        # 5. W1 updating
        dW1_dt = self.Gamma1 * (v1 * S1 - self.sigma1 * W1)
        
        # Step 2
        # 1. Tracking error
        z2 = x2 - alpha1
        
        # 2. NN2
        Z2 = np.array([x1, x2, alpha1])
        S2 = self.rbf2.compute_basis(Z2)
        f_hat_2 = np.dot(W2, S2)
        
        # 3. Control law u (v1 included)
        u = (1.0 / self.delta) * (-self.k2 * z2 - f_hat_2 - v1)
        
        # 4. W2 updating
        dW2_dt = self.Gamma2 * (z2 * S2 - self.sigma2 * W2)
        
        return u, dW1_dt, dW2_dt, alpha1