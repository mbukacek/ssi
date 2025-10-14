import numpy as np
import pandas as pd
import math as math
from matplotlib import pyplot as plt 
import matplotlib.gridspec as gridspec

#  SCRIPT ILLUSTRATES PED DATA VISUALIZATION TOOLS 
#


# Prepare figure for plots
fig = plt.figure(figsize=(10, 6))
gs = gridspec.GridSpec(2, 3, figure=fig)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])
ax4 = fig.add_subplot(gs[1, 0])
ax5 = fig.add_subplot(gs[1, 1])
ax6 = fig.add_subplot(gs[1, 2])

# Load from pickle
print('.. loading od data')
ped_frames = pd.read_pickle('ped_frames.pkl')

# Drop init row
ped_frames = ped_frames[ped_frames.r > 0]


#----------------------------------------- add occupancy in monitored area

print('.. calculating occumapncy')
ped_frames['N'] = np.nan

for i in range(len(ped_frames)):
    if type(ped_frames.ped_id[i]) == int:
        ped_frames.loc[i,'N'] = 1
    else:
        ped_frames.loc[i,'N'] = len(ped_frames.x[i])
        
ax1.plot(ped_frames.N)
ax1.set_title("occupancy in monitored area")

#----------------------------------------- plot situation in random time

ax2.plot(ped_frames.x[63000], ped_frames.y[63000], 'bo')
ax2.plot(ped_frames.x[63010], ped_frames.y[63010], 'ro')
ax2.set_title("ped positions on frame 63 000 (blue) and 63 010 (red)")


#----------------------------------------- analyze flow and occupancy in round 7

print('.. analyzing round 7')
ped_frames_loc = ped_frames[ped_frames.r == 7].copy()
ped_frames_loc.reset_index(drop=True, inplace=True)

ax3.plot(ped_frames_loc.t, ped_frames_loc.N)
ax3.set_title("occupancy in monitored area during round 7 in sec")

ped_out = []    # list of pedestrians in exit area in given frame
ped_left = []   # list of pedestrians in exit area any tiome up to fiven frame
n_in = []       # number of person in detector .. series over frame frame
n_left = []     # number of person that left detector from previous to present frame .. series over frame frame


for i in range(len(ped_frames_loc)): 

   if type(ped_frames_loc.ped_id[i]) == int:
       
       if ped_frames_loc.x[i] < 3:
           ped_out = ped_frames_loc.ped_id[i]
           n_in.append(0)
       else:
           ped_out = []
           n_in.append(1)
           
   else:
       ped_out = ped_frames_loc.ped_id[i][ped_frames_loc.x[i]<3]
       n_in.append(sum(ped_frames_loc.x[i] > 3))
       
   ped_out = np.setdiff1d(ped_out, ped_left)                # keep only peds that are new in exit area
   n_left.append(len(ped_out))                        
   
   if len(ped_out) > 0:
       ped_left = np.concatenate((ped_left, ped_out))

# convert to flow
j_out = [i * 48 for i in n_left]        # 48 FPS

# two version of smoothing
core_cylinder = [1] * 500
j_out_smooth_cyl = np.convolve(j_out, core_cylinder, 'same')/sum(core_cylinder)        

core_triangle = np.convolve(core_cylinder, core_cylinder, 'same')
j_out_smooth_tri = np.convolve(j_out, core_triangle, 'same')/sum(core_triangle)    

ax4.plot(j_out_smooth_tri, 'g.')
ax4.plot(j_out_smooth_cyl, 'r.')
ax4.set_title("outflow during round 7 smoothed by cylinder (red) or triangle (green)")

ax5.plot(n_in, j_out_smooth_tri, 'g-')
ax5.set_title("outflow vs occupancy during round 7")











