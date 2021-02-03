import numpy as np
from scipy.signal import resample
from scipy.io import loadmat

class Base:
    def remove_prefix(self, text, prefix):
        if text.startswith(prefix):
            return text[len(prefix):]
        return text 

def custom_loadmat(file):
    """
    Simple auxiliary function to load .mat files without
    the unnecesary MATLAB keys and squeezing unnecesary
    dimensions
    """
    d = loadmat(file)
    del d['__globals__']
    del d['__header__']
    del d['__version__']
    d = {k: d[k].squeeze() for k in d}
    return d

def get_transitions(states):
    """
    Computes transitions given a state array

    Args:
        states : numpy array
            States array of the form
            ...,4,1,1,...,1,2,2,...,2,3,3,....,3,4,...,4,1,...
    Returns:
        transitions : numpy array
            Contains indices of all the transitions in the states array
    """

    # Removes any single-dimensional entries from the array
    states = np.squeeze(states)

    # Edge cases when starts in 1 and/or ends in 4.
    if states[0] == 1:
        states = np.concatenate(([4], states))
    if states[-1] == 4:
        states = np.concatenate((states, [1]))
    
    # Determine all indices where there is a change in the state. We add 1 to
    # account for the offset associated with diff in numpy.
    transitions = np.where(np.diff(states) != 0)[0] + 1

    # Grab the first and last indices of S1 and S4 states.
    first = np.where(states == 1)[0][0]
    last = np.where(states == 4)[0][-1] + 1

    # Ensure all transitions are within range of first and last indices associated
    # with S1 and S4 cycle, respectively - discard any outside this range.
    transitions = transitions[np.logical_and(transitions >= first, transitions <= last)]
    return transitions

def boundaries(transitions, interval='RR'):
    """
    Given array transitions and a interval type
    computes an array of starting indices and ending indices for that
    given type of interval.

    E.g S1 will return two lists of indices
        * Indices where ...,4,1,... happens. Index will be where 1
        * Indices where ...,1,2,... happens. Index will be where 2

    Args:
        transitions : numpy array
            Array of state transitions. Computed by get_transitions()
        interval : string
            Type of interval [RR, S1, Sys, S2, Dia]. Defaults to RR
    Returns:
        pair_transitions : tuple( numpy array begin, numpy array end)
            begin - indices where the intervals start
            end   - indices where the intervals end
    """
    pair_transitions = {
        'RR':  lambda transitions: (transitions[:-1:4],  transitions[4::4]), # Returns indices four apart, starting from index 0 and 4.
        'S1':  lambda transitions: (transitions[0:-1:4], transitions[1::4]),
        'Sys': lambda transitions: (transitions[1::4],   transitions[2::4]),
        'S2':  lambda transitions: (transitions[2::4],   transitions[3::4]),
        'Dia': lambda transitions: (transitions[3::4],   transitions[4::4]),
    }
    return pair_transitions[interval](transitions)

def get_intervals(pcg, transitions=None, interval='RR', resize=None):
    """
    Given array transitions and a interval type computes an array of 
    amplitude values.

    Args:
        pcg : numpy array
        transitions : tuple( numpy array begin, numpy array end)
            begin - indices where the intervals start
            end   - indices where the intervals end
        interval : string
            Type of interval [RR, S1, Sys, S2, Dia]. Defaults to RR
        resize : int
            resample the interval to a specified length
    Returns:
        intervals : list<numpy array>
            list of intervals of the specified type
    """

    # Extract the amplitude values of the PCG at each interval, and store as a list
    # in the output intervals list.
    intervals = [pcg[i:j] for i, j in zip(*boundaries(transitions, interval))]
    if resize:
        intervals = [resample(i, resize) for i in intervals]
    return intervals

