import numpy as np
import pandas as pd
import pylab as pl
from pykalman import KalmanFilter
from data_filter import load_data

rnd = np.random.RandomState(0)

# generate a noisy sine wave to act as our fake observations
n_timesteps = 100
x = np.linspace(0, 3 * np.pi, n_timesteps)
xobservations = 20 * (np.sin(x) + 0.5 * rnd.randn(n_timesteps))

# generate actual observations

observations = load_data()
x1 = observations['ACCELEROMETER_X'].values
x2 = observations['ACCELEROMETER_Y'].values
x3 = observations['ACCELEROMETER_Z'].values
#y = df_train['state'].values
X = np.column_stack([x1, x2, x3])

# create a Kalman Filter by hinting at the size of the state and observation space.
# If you already have good guesses for the initial parameters, put them in here.
# The Kalman filter will try to learn the values of all variables

# The Kalman Filter is parameterized by 3 arrays for state transitions, 3 for measurements, and 2 more for initial conditions.

kf = KalmanFilter(transition_matrices=np.array([[1, 1], [0, 1]]),
					transition_covariance=0.01 * np.eye(2))
					
# You can use the kalman Filter immediately without fitting, but its estimates
# may not be as good as if you fit first

# The KalmanFilter class however can learn parameters using KalmanFilter.em() (fitting is optional).
# Then the hidden sequence of states can be predicted using KalmanFilter.smooth():

# Parameters :	Z : [n_timesteps, n_dim_state] array
states_pred = kf.em(X).smooth([X])[0]

#print('fitted model: {0}'.format(kf))

# Plot lines for the observations without noise, the estimated position of the
# target before fitting, and the estimated position after fitting.
#pl.figure(figsize=(16, 6))
#obs_scatter = pl.scatter(x, observations, marker='x', color='b',
#                         label='observations')
#position_line = pl.plot(x, states_pred[:, 0],
#                        linestyle='-', marker='o', color='r',
#                        label='position est.')
#velocity_line = pl.plot(x, states_pred[:, 1],
#                        linestyle='-', marker='o', color='g',
#                        label='velocity est.')
#pl.legend(loc='lower right')
#pl.xlim(xmin=0, xmax=x.max())
#pl.xlabel('time')
# pl.show()
a = xobservations
print a
print X