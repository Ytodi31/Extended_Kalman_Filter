import pickle
import numpy as np
import matplotlib.pyplot as plt

with open('data/data.pickle', 'rb') as f:
    data = pickle.load(f)

t = data['t']  # timestamps [s]

x_init  = data['x_init'] # initial x position [m]
y_init  = data['y_init'] # initial y position [m]
th_init = data['th_init'] # initial theta position [rad]

# input signal
v  = data['v']  # translational velocity input [m/s]
om = data['om']  # rotational velocity input [rad/s]

# bearing and range measurements, LIDAR constants
b = data['b']  # bearing to each landmarks center in the frame attached to the laser [rad]
r = data['r']  # range measurements [m]
l = data['l']  # x,y positions of landmarks [m]
d = data['d']  # distance between robot center and laser rangefinder [m]

v_var = 0.01  # translation velocity variance  
om_var = 0.01  # rotational velocity variance 
r_var = 0.1  # range measurements variance
b_var = 0.1 # bearing measurement variance

Q_km = np.diag([v_var, om_var]) # input noise covariance 
cov_y = np.diag([r_var, b_var])  # measurement noise covariance 

x_est = np.zeros([len(v), 3])  # estimated states, x, y, and theta
P_est = np.zeros([len(v), 3, 3])  # state covariance matrices

x_est[0] = np.array([x_init, y_init, th_init]) # initial state
P_est[0] = np.diag([1, 1, 0.1]) # initial state covariance


def wraptopi(x):
    if x > np.pi:
        x = x - (np.floor(x / (2 * np.pi)) + 1) * 2 * np.pi
    elif x < -np.pi:
        x = x + (np.floor(x / (-2 * np.pi)) + 1) * 2 * np.pi
    return x




def measurement_update(lk, rk, bk, P_check, x_check):
    
    # 1. Compute measurement Jacobian
    x_check[2] = wraptopi(x_check[2])
    r = np.sqrt((lk[0] - x_check[0] - d*np.cos(x_check[2]))**2 + (lk[1] - x_check[1] - d*np.sin(x_check[2]))**2) 
    phi = np.arctan2(lk[1] - x_check[1] - d*np.sin(x_check[2]), lk[0] - x_check[0] - d*np.cos(x_check[2])) - x_check[2]
    phi = wraptopi(phi)
    yk = np.reshape(np.array([r, phi]), (1,2)) 
    n = np.random.multivariate_normal([0, 0], cov_y, 1)
    yk = yk

    dh1_dx1 = -(lk[0] - x_check[0] - d*np.cos(x_check[2]))/r
    dh1_dx2 = -(lk[1] - x_check[1] - d*np.sin(x_check[2]))/r
    dh1_dx3 = ((lk[0] -  x_check[0] - d*np.cos(x_check[2]))*np.sin(x_check[2]) + 
               (lk[1] - x_check[1] - d*np.sin(x_check[2]))*-np.cos(x_check[2]))*d/r
    dh2_dx1 = (lk[1] - x_check[1] - d*np.sin(x_check[2]))/r**2
    dh2_dx2 = -(lk[0] -  x_check[0] - d*np.cos(x_check[2]))/r**2
    dh2_dx3 = -((lk[1] - x_check[1] - d*np.sin(x_check[2]))*d*np.sin(x_check[2]) + 
                  (lk[0] -  x_check[0] - d*np.cos(x_check[2]))*d*np.cos(x_check[2]))/r**2

    H = np.array([[dh1_dx1, dh1_dx2, dh1_dx3], [dh2_dx1, dh2_dx2, dh2_dx3]])
    H = np.reshape(H, (2,3))
    M = np.eye(2)

    # 2. Compute Kalman Gain

    # 3. Correct predicted state

    # 4. Correct covariance
	
    return x_check, P_check
