import numpy as np
import pandas as pd
import pylab as pl
from pykalman import KalmanFilter
from data_filter import load_data
from ggplot import ggplot, aes, geom_point

rnd = np.random.RandomState(0)

# generate actual observations
observations = load_data()
x1 = observations['ACCELEROMETER_X'].values
x2 = observations['ACCELEROMETER_Y'].values
x3 = observations['ACCELEROMETER_Z'].values

X = np.column_stack([x1, x2, x3])

#### specify parameters ####

# random_state
# transition_offset
# observation_offset
# initial_state_mean
# n_timesteps
transition_matrix = np.array([[1, 1, 0], [0, 1, 0],[0, 0, 1]])
observation_matrix = [X[0],X[1],X[2]]



# create a Kalman Filter by hinting at the size of the state and observation space.
# If you already have good guesses for the initial parameters, put them in here.
# The Kalman filter will try to learn the values of all variables

# The Kalman Filter is parameterized by 3 arrays for state transitions, 3 for measurements, and 2 more for initial conditions.
## This worked
kf = KalmanFilter(transition_matrices=transition_matrix,
                  observation_matrices=observation_matix,
				  transition_covariance=0.01 * np.eye(3)
)

kf = kf.em(X, n_iter=5)

(filtered_state_means, filtered_state_covariances) = kf.filter(X)
(smoothed_state_means, smoothed_state_covariances) = kf.smooth(X)

# kf.em(X).smooth([X]) ## this gives the shape error
kf = kf.em(X).smooth(X) # This returns large tuple
# print len(kf_test)

# Plot lines for the observations without noise, the estimated position of the
# target before fitting, and the estimated position after fitting.

states_pred = kf

n_timesteps = X.shape[0]
x = np.linspace(0, 3 * np.pi, n_timesteps)
xobservations = 20 * (np.sin(x) + 0.5 * rnd.randn(n_timesteps))

observations['noise'] = x
observations['sin'] = xobservations

pl.figure(figsize=(16, 6))
obs_scatter = pl.scatter(x, x1, marker='x', color='b',
                         label='observations')
position_line = pl.plot(x, states_pred[0],linestyle='-', marker='o', color='r',label='position est.')
# velocity_line = pl.plot(x, states_pred[1], linestyle='-', marker='o', color='g',label='velocity est.')
pl.legend(loc='lower right')
pl.xlim(xmin=0, xmax=x.max())
pl.xlabel('time')
pl.show()






"""QUESTIONS

- Should the observation matrix be the data? Can I add more data?
- How do we come up with the transition matrix values?
- How do we determine the transition_covariance

"""