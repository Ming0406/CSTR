# CSTR Intelligent Adaptive Constraint Control

## Overview
This project provides a Python-based numerical simulation framework for the intelligent adaptive control of a Continuous Stirred Tank Reactor (CSTR). The system addresses the challenges of strong non-linearity and state constraints using a combination of **Radial Basis Function Neural Networks (RBFNNs)**, **Backstepping Control**, and **Barrier Lyapunov Functions (BLF)**.

The primary objective is to ensure that the reactor's concentration tracks a desired reference signal while strictly adhering to time-varying state constraints, thereby ensuring operational safety and stability in chemical processing.

## Mathematical Modelling
The simulation is based on a dimensionless dynamic model of a CSTR. The system state is defined by the reactant concentration ($x_1$) and the mixture temperature ($x_2$). The non-linear behaviour arises from the Arrhenius reaction rate term.

## Control Strategy
1.  **Non-linear Approximation**: RBF Neural Networks are utilised to identify and compensate for unknown system dynamics online [DOI: 10.1109/TNNLS.2021.3107600].
2.  **Constraint Handling**: Logarithmic Barrier Lyapunov Functions (BLF) are integrated into the control design to prevent the violation of time-varying output constraints [DOI: 10.1109/TSMC.2019.2956769].
3.  **Stability**: The closed-loop stability and finite-time convergence properties are analysed through the Lyapunov stability theory [DOI: 10.1109/TFUZZ.2018.2882173].

## Project Structure
* `cstr_plant.py`: Defines the CSTR physical parameters and non-linear ODEs.
* `rbf_nn.py`: Implements the Gaussian basis function calculations for the RBFNN.
* `controller.py`: Contains the backstepping control logic and the adaptive weight update laws.
* `main.py`: The entry point for initialising the global state vector, executing the ODE solver, and visualising the results.

## Performance Visualisation
The simulation generates plots illustrating:
* Real-time tracking of the concentration against the reference signal.
* Strict adherence to upper and lower time-varying constraint boundaries.
* The evolution of the control input (coolant flow).

## References
* Liu, Y. J., et al. "Adaptive Neural Network Control for a Class of Nonlinear Systems With Function Constraints on States." *IEEE Transactions on Neural Networks and Learning Systems*. DOI: 10.1109/TNNLS.2021.3107600.
* Li, D., et al. "Adaptive Finite-Time Tracking Control for Continuous Stirred Tank Reactor With Time-Varying Output Constraint." *IEEE Transactions on Systems, Man, and Cybernetics: Systems*. DOI: 10.1109/TSMC.2019.2956769.
* Liu, L., et al. "Fuzzy Based Multi-Error Constraint Control for Switched Nonlinear Systems and Its Applications." *IEEE Transactions on Fuzzy Systems*. DOI: 10.1109/TFUZZ.2018.2882173.