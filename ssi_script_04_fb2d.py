import pandas as pd
import numpy as np
import math as math
from matplotlib import pyplot as plt 

# https://pedpy.readthedocs.io/en/stable/

def init_ped_data(t,x,y,vx,vy):
    # initialize ped data container
    # INPUT: vector of init time, init position and init velocity
    
    # init dataframe with the first ped
    ped_data = pd.DataFrame({'ped_id': 0,
                             't': [[t[0]]],                         # to be list of recorded time
                             'x': [[x[0]]],                         # to be list of recorded x position
                             'y': [[y[0]]],                         # to be list of recorded y position
                             'vx': [[vx[0]]],                       # to be list of recorded x velocity
                             'vy': [[vy[0]]]                        # to be list of recorded y velocity
                             }, index = [0])

    # add peds one by one
    rep = range(len(x)-1)
    for i in rep:
        ped_data_n = pd.DataFrame({'ped_id': i+1,
                                't': [[t[i+1]]],                         
                                'x': [[x[i+1]]],   
                                'y': [[y[i+1]]],      
                                'vx': [[vx[i+1]]],
                                'vy': [[vy[i+1]]]  
                                }, index = [i+1])
        ped_data = ped_data.append(ped_data_n)
    
    return ped_data


def one_ped_step(ped_data, ped_idx, act_t, act_vx, act_vy, const):
    # performs one step of one ped with respect to the new velocity
    
    ped_data.t[ped_idx] = ped_data.t[ped_idx] + [act_t]             # '+' works as append   
    
    ped_data.vx[ped_idx] = ped_data.vx[ped_idx] + [act_vx]  
    ped_data.vy[ped_idx] = ped_data.vy[ped_idx] + [act_vy]       
    
    new_x = ped_data.x[ped_idx][-1] + act_vx*const['dt']             # [-1] reffers to the last value            
    ped_data.x[ped_idx] = ped_data.x[ped_idx] + [new_x]
    
    new_y = ped_data.y[ped_idx][-1] + act_vy*const['dt']                         
    ped_data.y[ped_idx] = ped_data.y[ped_idx] + [new_y]
    
    return ped_data


def update_v(ped_data, ped_idx, model_name, const):
    # With respect to the model and situation, new velocity is calculated here
    
    if model_name == 'no_change':
        new_vx = ped_data.vx[ped_idx][-1]
        new_vy = ped_data.vy[ped_idx][-1]
        
 #   elif model_name == 'xxx':
       
    else:
        new_vx = np.nan
        new_vy = np.nan
    
    return new_vx, new_vy


#============================================#
#              SCRIPT STARTS HERE            #
#============================================#

#======================#
#     PRELIMINARIES    #
#======================#

# Constants - dictionary
const = {'N_ped': 5,                # numer of peds in the system
         'N_step': 10,              # number of steps
         'dt': 0.1,                 # diffrential step length [s]
         'attractor_x': 10,         # x position of attractor [m]
         'attractor_y': 5           # y position of attractor [m]
        }

# Init time, positions and velocities
t = [0, 0, 0, 0, 0]
x = [1.1, 0.9, 1, 3, 3.05]
y = [1, 2.5, 4, 1.1, 2.6]
vx = [1.5, 1.5, 1.5, 1.5, 1.5]
vy = [0,0,0,0,0]

# Init data containers
ped_data = init_ped_data(t,x,y,vx,vy)



#======================#
#         MODEL        #
#======================#

# Model loop over time
rep = range(const['N_step'])
for i in rep: 

    act_t = (i+1)*const['dt']                                       # i+1 is current itteration as i = 0  was defined in init step 
    
    # model loop over other peds 
    rep2 = range(const['N_ped'])
    for j in rep2: 
    
        act_vx, act_vy = update_v(ped_data, j, 'no_change', const)
        ped_data = one_ped_step(ped_data, j, act_t, act_vx, act_vy, const)
    


#======================#
#     POSTPROCESSING   #
#======================#


# Timespace fundamental diagram
plt.figure()
plt.plot(ped_data.t[0], ped_data.x[0], 'r-', label = 'ped 1')
plt.plot(ped_data.t[1], ped_data.x[1], 'g-', label = 'ped 2')
plt.plot(ped_data.t[2], ped_data.x[2], 'b-', label = 'ped 3')
plt.plot(ped_data.t[3], ped_data.x[3], 'k-', label = 'ped 4')
plt.plot(ped_data.t[4], ped_data.x[4], 'm-', label = 'ped 5')
plt.title('Timespace fundamental diagram')
plt.xlabel(r'$t \,\,\mathrm{[s]}$')
plt.ylabel(r'$x \,\,\, \mathrm{[m]}$')
#plt.xlim(0, 10)
#plt.ylim(0, 120)
plt.legend()
plt.show()

# Aerial plot
plt.figure()
plt.plot(ped_data.x[0], ped_data.y[0], 'r-o', label = 'ped 1')
plt.plot(ped_data.x[1], ped_data.y[1], 'g-o', label = 'ped 2')
plt.plot(ped_data.x[2], ped_data.y[2], 'b-o', label = 'ped 3')
plt.plot(ped_data.x[3], ped_data.y[3], 'k-o', label = 'ped 4')
plt.plot(ped_data.x[4], ped_data.y[4], 'm-o', label = 'ped 5')
plt.title('Aerial plot')
plt.xlabel(r'$x \,\,\mathrm{[m]}$')
plt.ylabel(r'$y \,\,\, \mathrm{[m]}$')
#plt.xlim(0, 10)
#plt.ylim(0, 120)
plt.legend()
plt.show()


