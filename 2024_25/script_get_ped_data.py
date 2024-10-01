#import numpy as np
import pandas as pd
#import math as math
from matplotlib import pyplot as plt 

import scipy.io as spio

# Load matlab files
#fullFinal = spio.loadmat('FullFinal.mat', struct_as_record=False, squeeze_me=True)
frameFullFinal = spio.loadmat('frameFullFinal.mat', struct_as_record=False, squeeze_me=True)

# Prepare empty structure
ped_frames = pd.DataFrame({'t': 0,                              # to experiment time within each round
                           'x': [0],                            # to be list of recorded x-positions
                           'y': [0],                            # to be list of recorded y-positions
                           'ped_id': [0],                       # to be list of recorded pedestrian ids
                           'r': 0                               # to experiment round id
                         }, index = [0])


# Read and save data frame by frame [3 min]
rep = range(len(frameFullFinal['frameStruct']))
for i in rep: 
    if i % 10000 == 0:
        print(i)
        
    ped_frames_n = pd.DataFrame({'t': frameFullFinal['frameStruct'][i].t,                           
                               'x': [frameFullFinal['frameStruct'][i].x],                        
                               'y': [frameFullFinal['frameStruct'][i].y],    
                               'ped_id': [frameFullFinal['frameStruct'][i].ID],
                               'r': frameFullFinal['frameStruct'][i].round
                             }, index = [i])  
    ped_frames = pd.concat([ped_frames, ped_frames_n])


# Examples
p = plt.plot(ped_frames.x[10489], ped_frames.y[10489],'bo')

# Save to pickle
ped_frames.to_pickle('ped_frames.pkl')

# Load from pickle
ped_frames_loaded = pd.read_pickle('ped_frames.pkl')