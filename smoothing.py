# smoothing.py

import numpy as np

class KalmanFilter:
    """
    A simple Kalman Filter for 2D object tracking (position and velocity).
    Assumes a constant velocity model.
    """
    def __init__(self, dt, measurement_noise, process_noise):
        """
        Initializes the Kalman Filter.
        :param dt: Time step of the simulation.
        :param measurement_noise: The uncertainty of the position measurement.
        :param process_noise: The uncertainty of the motion model (e.g., unexpected accelerations).
        """
        self.dt = dt

        # State vector [x, y, vx, vy]
        self.x = np.zeros((4, 1))

        # State Transition Matrix F
        # Projects the state forward one time step
        self.F = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])

        # Measurement Matrix H
        # Maps the state vector to the measurement space (we only measure position)
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])

        # Process Noise Covariance Q
        # Represents the uncertainty in our motion model
        self.Q = np.eye(4) * process_noise

        # Measurement Noise Covariance R
        # Represents the uncertainty in our measurement
        self.R = np.eye(2) * measurement_noise

        # Initial State Covariance P
        # Our initial uncertainty about the agent's state
        self.P = np.eye(4) * 500

    def predict(self):
        """Performs the prediction step."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x

    def update(self, z):
        """
        Performs the update step with a new measurement.
        :param z: The measurement vector (observed [x, y] position).
        """
        # Measurement residual (error)
        y = z - self.H @ self.x

        # Residual covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman Gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Update state estimate
        self.x = self.x + K @ y

        # Update state covariance
        I = np.eye(4)
        self.P = (I - K @ self.H) @ self.P

def smooth_trajectories(simulated_agents, config):
    """
    Applies a Kalman filter to smooth the trajectories of all agents.
    :param simulated_agents: The list of agent objects after simulation.
    :param config: The simulation configuration dictionary.
    :return: A dictionary mapping agent_id to its smoothed trajectory.
    """
    print("\n--- Smoothing agent trajectories using Kalman Filter ---")
    smoothed_trajectories = {}
    dt = config['dt']
    
    # Tuning parameters for the filter
    measurement_noise = 0.5  # How much we trust the measurement (lower is more trust)
    process_noise = 0.1      # How much we expect the agent to deviate from a constant velocity path

    for agent in simulated_agents:
        if not agent.history_pos:
            continue

        kf = KalmanFilter(dt, measurement_noise, process_noise)
        
        # Initialize the filter with the first known position and zero velocity
        initial_pos = agent.history_pos[0]
        kf.x = np.array([[initial_pos[0]], [initial_pos[1]], [0], [0]])
        
        smoothed_path = []
        for pos in agent.history_pos:
            # Predict the next state
            kf.predict()
            # Update the state with the actual measurement
            kf.update(np.array([[pos[0]], [pos[1]]]))
            # Store the smoothed position
            smoothed_path.append(kf.x[0:2].flatten().tolist() + [pos[2]]) # Keep original z-coord

        smoothed_trajectories[agent.id] = smoothed_path
    
    print("Trajectory smoothing complete.")
    return smoothed_trajectories