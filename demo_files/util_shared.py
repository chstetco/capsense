import glob, os, sys, pickle
import numpy as np

def timewindow(x, window=50, flatten=True):
    return [(x[i:i+window].flatten() if flatten else x[i:i+window]) for i in range(len(x)-window+1)]

def load_data_from_file(files='*.pkl', directory='trainingdata', window=1, flatten=True, verbose=False):
    allCapacitance = []
    allPosition = []
    allTimes = []
    actions = []

    if type(files) == str:
        filenames = glob.glob(os.path.join(directory, files))
    else:
        filenames = []
        for f in files:
            filenames.extend(glob.glob(os.path.join(directory, f)))
    for filename in filenames:
        if verbose:
            print(filename)
        with open(filename, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        capacitanceReadings, limbCircumference = data
        capacitance = np.array([d[0] for d in capacitanceReadings])
        # [+ away from robot, + left (robot's perspective) towards center of robot, + up towards the sky]
        position = np.array([[d[1][0], d[1][2], d[2][0], d[2][2]] for d in capacitanceReadings])
        times = np.array([d[-9] for d in capacitanceReadings])
        if window > 1:
            # Window the data
            capacitance = timewindow(capacitance, window=window, flatten=flatten)
            position = position[window-1:]
            times = times[window-1:]
            # Sometimes we generate a random position close to the current position, so there is less data that the size of a window.
            if len(capacitance) == 0:
                # Skip this data
                continue
        # For windowed models (SVM and NN)
        allCapacitance.extend(capacitance)
        allPosition.extend(position)
        allTimes.extend(times)
    return np.array(allCapacitance), np.array(allPosition), np.array(allTimes)

