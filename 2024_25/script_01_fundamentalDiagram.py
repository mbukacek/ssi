import numpy as np
import pandas as pd
import math as math
from matplotlib import pyplot as plt 


# Load from pickle
ped_frames = pd.read_pickle('ped_frames.pkl')

#
dens = []
time = []
flow = []

n_in = []
n_left = []

ped_left = []
r7start = 53739
r7last = 64811

i = 0


for i in range(r7last-r7start+1): 
#for i in range(100): 
   
   if type(ped_frames.ped_id[r7start+i]) == int:
       
       if ped_frames.ped_id[r7start+i] < 3:
           ped_out = ped_frames.ped_id[r7start+i]
           n_in.append(0)
       else:
           ped_out = []
           n_in.append(1)
           
   else:
       ped_out = ped_frames.ped_id[r7start+i][ped_frames.x[r7start+i]<3]
       n_in.append(sum(ped_frames.x[r7start+i]>3))
       
   ped_out = np.setdiff1d(ped_out, ped_left)
   n_left.append(len(ped_out))
   
   if len(ped_out) > 0:
       ped_left = np.concatenate((ped_left, ped_out))

    
plt.plot(n_in)

core = [1] * 500

n_left_smooth = np.convolve(n_left,core, 'same')
plt.plot(n_left_smooth)

plt.plot(n_in, n_left_smooth, 'k.')



