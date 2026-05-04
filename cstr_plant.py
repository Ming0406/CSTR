import numpy as np

class CSTRSystem:
    # CSTR dynamical model
    def __init__(self):
        # parameter setting
        self.Ga = 0.072  # Dam-kohl number
        self.phi = 20.0  # Activation energy parameter
        self.delta = 0.3 # Heat transfer coefficient
        self.B = 8.0     # Reactive heat parameters
        
    def reaction_rate(self, x1, x2):
        # reaction rate
        # index
        exponent = x2 / (1.0 + x2 / self.phi)
        # rate returning
        return self.Ga * (1.0 - x1) * np.exp(exponent)

    def dynamics(self, t, state, u):
        """
        :param t: current time
        :param state: [x1, x2]， conc. = x1，temp = x2
        :param u: flow of coolant
        :return: [dx1_dt, dx2_dt]
        """
        x1, x2 = state
        rate = self.reaction_rate(x1, x2)
        dx1_dt = -x1 + rate
        dx2_dt = -x2 * (1.0 + self.delta) + self.B * rate + self.delta * u
        return np.array([dx1_dt, dx2_dt])

    @staticmethod
    def get_reference_signal(t):
        """
        :param t: current time
        :return: tuple (yd, dot_yd)
        """
        yd = 0.1 * np.sin(t) + 0.2
        dot_yd = 0.1 * np.cos(t)
        return yd, dot_yd

if __name__ == "__main__":
    # Logic test
    plant = CSTRSystem()
    test_state = [0.18, 0.5]  # Set Initial State
    test_u = 0.0              # zero input
    
    derivatives = plant.dynamics(0.0, test_state, test_u)
    print('CSTR SYSTEM init succeed')
    print(f"x=[0.18, 0.5], u=0 : dx/dt = {derivatives}")