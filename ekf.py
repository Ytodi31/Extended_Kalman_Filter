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

    # 1. Computing measurement Jacobian
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

    # 2. Computing Kalman Gain
    K = np.matmul(np.matmul(P_check, H.T), np.linalg.inv(np.matmul(H, np.matmul(P_check, H.T))+
                                                  np.matmul(M, np.matmul(cov_y, M.T))))

    # 3. Correcting predicted state (remember to wrap the angles to [-pi,pi])
    ym = np.array([rk, wraptopi(bk)])
    yk = np.reshape(yk, ym.shape)
    x_check +=  np.matmul(K, ym - yk)
    x_check[2] = wraptopi(x_check[2])

    # 4. Correcting covariance
    I = np.eye(3)
    P_check = np.matmul(I - np.matmul(K, H), P_check)

    return x_check, P_check

x_check = x_est[0, :]
P_check = P_est[0]
for k in range(1, len(t)):
	# 1. Updating state with odometry readings
	x_check[2] = wraptopi(x_check[2])
	F = np.array([[np.cos(x_check[2]), 0],
		              [np.sin(x_check[2]), 0],
		              [0,1]])
	Input = np.array([[v[k-1]], [om[k-1]]])
	w = np.reshape(np.random.multivariate_normal([0, 0], Q_km, 1), Input.shape)
	x_check += np.reshape(delta_t*np.matmul(F, Input), x_check.shape)
	x_check[2] = wraptopi(x_check[2])

	# 2. Motion model jacobian with respect to last state
	F_km = np.array([[1, 0, -np.sin(x_check[2])*(v[k-1])*(delta_t)],
		                 [0, 1,  np.cos(x_check[2])*(v[k-1])*(delta_t)],
		                 [0, 0, 1]])

	# 3. Motion model jacobian with respect to noise
	L_km = np.array([[np.cos(x_check[2])*delta_t, 0],
		                [np.sin(x_check[2])*delta_t, 0],
		                [0, delta_t]])

	# 4. Propagating uncertainty
	P_check = np.matmul(F_km, np.matmul(P_check, F_km.T)) + np.matmul(L_km, np.matmul(Q_km, L_km.T))

	# 5. Updating state estimate using available landmark measurements
	x_check, P_check = measurement_update(l[i], r[k, i], b[k, i], P_check, x_check)


	x_est[k, 0] = x_check[0]
    x_est[k, 1] = x_check[1]
    x_est[k, 2] = x_check[2]
    P_est[k, :, :] = P_check


# Plotting estimated trajectory
e_fig = plt.figure()
ax = e_fig.add_subplot(111)
ax.plot(x_est[:, 0], x_est[:, 1])
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_title('Estimated trajectory')
plt.show()

e_fig = plt.figure()
ax = e_fig.add_subplot(111)
ax.plot(t[:], x_est[:, 2])
ax.set_xlabel('Time [s]')
ax.set_ylabel('theta [rad]')
ax.set_title('Estimated trajectory')
plt.show()
