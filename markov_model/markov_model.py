import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label
from scipy.interpolate import interp1d

class MarkovModel():

    def __init__(self):
        
        """ 
        Determine transition probabilities of Markov model from terminus + runoff data.
        """

        ### Load terminus position and runoff data
        ##############################################################
        
        runoff = np.loadtxt('data/runoff_data.txt')
        terminus = np.loadtxt('data/terminus_data.txt')
        ts = np.linspace(2003., 2019., 16*52+1)
        runoff = interp1d(runoff[:,0], runoff[:,1])(ts)
        terminus = interp1d(terminus[:,0], terminus[:,1])(ts)

        self.ts = ts
        self.terminus = terminus
        self.runoff = runoff

        
        # Compute states (advancing / retreating) at each step
        ##############################################################
        
        dl = terminus[1:] - terminus[:-1]
        dl[dl >= 0.] = 1.
        dl[dl < 0.] = 0.
        states = np.array(np.insert(dl, 0, 0), dtype = int)
        self.states_data = states

        ### Compute transition pobabilities (Maximum likelihood estimation)
        ##############################################################
        
        # Number of bins
        N = 9
        self.N = N
        # Load the runof data
        # Runoff bins
        max_threshold = 700.
        self.max_threshold = max_threshold
        bins = np.linspace(0., max_threshold, N)
        bins = np.append(bins, runoff.max()+1000.)
        self.bins = bins
        # Divide the runoff data into bins
        runoff_bins = (runoff / max_threshold) * float(N)
        runoff_bins = np.array(runoff_bins, dtype = int)
        runoff_bins[runoff_bins  > (N-1)] = N-1

        # Compute transition probabilities for each runoff bin
        p_A_Rs = np.zeros(N)
        p_R_As = np.zeros(N)

        for i in range(N):
            groups = np.zeros_like(runoff_bins)
            indexes = runoff_bins == i
            groups[indexes] = 1
            groups, n = label(groups)

            n_1_0 = 0
            n_1_1 = 0
            n_0_1 = 0
            n_0_0 = 0

            for j in range(1,n):
                indexes_j = groups == j
                sequence = states[indexes_j]
                pairs = np.c_[sequence[1:], sequence[:-1]]
                transitions = pairs[:,0]*2 + pairs[:,1]

                n_1_0 += (transitions == 2).sum()
                n_1_1 += (transitions == 3).sum()

                n_0_1 += (transitions == 1).sum()
                n_0_0 += (transitions == 0).sum()

            #print(n_0_1 + n_0_0)
            #print(n_1_0 + n_1_1)
            #print()
            
            p_R_As[i] = n_0_1 / (n_0_1 + n_0_0 + 1e-16)
            p_A_Rs[i] = n_1_0 / (n_1_0 + n_1_1 + 1e-16)

        # Manually assign probabilities for last bin
        p_A_Rs[-1] = 1.
        p_R_As[-1] = 0.
        
        self.P_A_As = 1. - p_A_Rs
        self.P_R_Rs = 1. - p_R_As
        self.P_A_Rs = p_A_Rs
        self.P_R_As = p_R_As


    def get_sequence(self, runoff):
        xs = np.random.rand(len(runoff))
        S_i = 'A'
        states = []
        
        for i in range(len(xs)):
            r_i = runoff[i]

            bin_index = int((r_i / self.max_threshold) * float(self.N))
            bin_index = min(bin_index, self.N-1)
            x = xs[i]

            if S_i == 'A':
                if x <= self.P_A_Rs[bin_index]:
                    S_i = 'R'
            else:
                if x <= self.P_R_As[bin_index]:
                    S_i = 'A'
                
            states.append(S_i)
        return np.array(states)
