import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from markov_model import MarkovModel
from scipy.io import savemat

""" 
Generates 100 random sequences using the Markov model. Each sequene consists
of either a 0 or 1 at a given time step. Given sigma_min and sigma_max 
values. Each sequence, s, can therefore be converted to a calving stress 
threshold input for the model via:
sigma = (sigma_max - sigma_min)*s + sigma_min
"""

mm = MarkovModel()
runoff = mm.runoff
ts = mm.ts

indexes = ts >= 2007.
ts = ts[indexes]
runoff = runoff[indexes]
states = []

for i in range(100):
    states_i = mm.get_sequence(runoff)

    indexes_A = states_i == 'A'
    numeric_states = np.zeros(len(states_i))
    numeric_states[indexes_A] = 1.

    states.append(numeric_states)
    print(numeric_states)

states = np.array(states)

plt.subplot(2,1,1)
plt.plot(ts, states.sum(axis=0))

plt.subplot(2,1,2)
plt.plot(ts, runoff)

np.save('data/sequence_ts', ts)
np.save('data/markov_sequences', states)
savemat('data/markov_sequences.mat', {'markov_sequences' : states, 'ts' : ts})
plt.show()
    




