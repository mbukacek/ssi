import numpy as np
import pandas as pd
import math as math
from matplotlib import pyplot as plt 



#  SCRIPT ILLUSTRATES PED DATA VISUALIZATION TOOLS 
#

#pocet lidi na case

#graf x na y + hustotni profil

#zavislost toku na hustote divat se jak rychle lide mizi (bod je mensi nez 3)
#pred vychodem x=3-4 a y=2-4
#stav pred vychodem, tok


# Load from pickle
ped_frames = pd.read_pickle('ped_frames.pkl')

ped_frames_loc = ped_frames[ped_frames.r ==7].copy()
ped_frames_loc.reset_index(drop=True, inplace=True)

n_in = []
n_out = []
ped_out = []
ped_left = []
start=53739
finish=64811

for i in range(len(ped_frames_loc)):
    
    if type(ped_frames_loc.ped_id[i])== int:
        
        if ped_frames_loc.x[i]<3:
            ped_out = ped_frames_loc.ped_id[i]
            n_in.append(0)
        else:
            ped_out = []
            n_in.append(1)
    else:
        ped_out = ped_frames_loc.ped_id[i][ped_frames_loc.x[i]<3]
        n_in.append(sum((ped_frames_loc.x[i]>3)))
        
    ped_out = np.setdiff1d(ped_out, ped_left)
    n_out.append(len(ped_out))
    
    if len(ped_out)>0:
        ped_left = np.concatenate((ped_left, ped_out))
        
j_out = [i * 48 for i in n_out]

core = [1] * 500
j_out_smooth = np.convolve(j_out, core, "same")/500
      
        
plt.figure
#plt.plot(j_out, "b.")
#plt.plot(j_out_smooth, 'r.')

plt.figure 
#plt.plot(n_in, j_out, 'b.')
plt.plot(n_in, j_out_smooth, 'r-')

# for i in range(start,finish,1):
#             n.append(sum(ped_frames.ped_id[(ped_frames.x[i]>3) and (ped_frames.x[i]<4) and (ped_frames.y[i]>2) and (ped_frames.y[i]<4)]))



# plt.plot(n)

# start=56300
# finish=56400

# plt.plot(ped_frames.x[finish],ped_frames.y[finish],"bo")

# for i in range(start,finish,1):
#     plt.plot(ped_frames.x[i],ped_frames.y[i],"bo")

