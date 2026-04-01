import numpy as np
# Kalman Filter (Skeleton) for Time to Transit prediction 
# Written by Manuel Morteo

dt = 1.0 # Time between observations

F = np.array([[1, dt], [0,1]])

H = np.array([[1,0]]) # Only measuring tau directly

Q = np.diag([0.01,0.005])

noise_std = 0.05 # noise standard for kalman?
R = np.array([noise_std**2])


def kf_predict(z, P):
    """
    z is state vector (tau, tau_dot)
    P is 2x2 Covariance matrix
    """ 
    z_pred = F @ z
    P_pred = F @ P @ F.T + Q

    return z_pred, P_pred

def kf_update(z, P, measurement):
    """
        Calculating Innovation 
        S = innovation covariance
        K = kalman gain
    """
    innovation = measurement - (H @ z).item() # item() extracts resulting value when matrix operator @  produces 1x1 matrix or a single scalar results

    S = H @ P @ H.T + R 
    K = P @ H.T @ np.linalg.inv(S)

    z_updated = z + (K * innovation).flatten()
    P_updated = (np.eye(2) - K @ H) @ P  
    
    return z_updated, P_updated
    
def run_filter(noise_measurements):
    # Initialize with first measurement
    z = np.array([noise_measurements[0], 0.0]) # [tau, tau_dot = 0]
    P = np.diag([noise_std**2, 0.5])

    estimates = []

    for y in noise_measurements: 
        
        # Predict next state z 
        z, P = kf_predict(z,P)
        
        # Correct with actual observation
        z, P = kf_update(z, P, y)
       
        # Call z_ahead again for KF+1
        z_ahead, P_ahead = kf_predict(z,P)
        tau_ahead = z_ahead[0] 
        sigma_ahead = np.sqrt(P_ahead[0,0])

        estimates.append({
            "tau_estimated": z[0],
            "tau_velocity": z[1], # Hard to measure in practice sense, refer to Chiara Boretti and Philip Bich Phd Thesis
            "uncertainty": np.sqrt(P[0,0]), # 1 -sigma
            "tau_k+1": z_ahead[0],
            "sigma_ahead": sigma_ahead
        })
    return estimates

if __name__ == "__main__":

    true_tau = np.linspace(5.0,1.0,40)
    # print(true_tau, "\n") 
    noisy_tau = true_tau + np.random.normal(0, noise_std, size=40)
    # print(noisy_tau)
    
    results = run_filter(noisy_tau)
   
    for k, (true, noisy, est) in enumerate(zip(true_tau, noisy_tau, results)):
        print(f"k={k:2d} | true={true:.3f} | noisy={noisy:.3f} "
              f"| KF={est['tau_estimated']:.3f} "
              f"| KF+1={est['tau_k+1']:.3f}"
              f"| σ={est['uncertainty']:.4f}")
    
