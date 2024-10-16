import pandas as pd
import numpy as np
import math as math
from matplotlib import pyplot as plt 

pd.options.mode.chained_assignment = None  # default='warn'

# https://pedpy.readthedocs.io/en/stable/

def init_ped_data(t,x,y,vx,vy,ex):
    # initialize ped data container
    # INPUT: vector of init time, init position and init velocity
    
    # init dataframe with the first ped
    ped_data = pd.DataFrame({'ped_id': 0,
                             't': [[t[0]]],                         # to be list of recorded time
                             'x': [[x[0]]],                         # to be list of recorded x position
                             'y': [[y[0]]],                         # to be list of recorded y position
                             'vx': [[vx[0]]],                       # to be list of recorded x velocity
                             'vy': [[vy[0]]],                       # to be list of recorded y velocity
                             'time_left': ex[0]                     # time when ped left - or NaN    
                             }, index = [0])

    # add peds one by one
    rep = range(len(x)-1)
    for i in rep:
        ped_data_n = pd.DataFrame({'ped_id': i+1,
                                't': [[t[i+1]]],                         
                                'x': [[x[i+1]]],   
                                'y': [[y[i+1]]],      
                                'vx': [[vx[i+1]]],
                                'vy': [[vy[i+1]]],
                                'time_left': ex[i+1] 
                                }, index = [i+1])
        ped_data = pd.concat([ped_data,ped_data_n])
    
    return ped_data


def one_ped_step(ped_data, ped_idx, act_t, act_vx, act_vy, const):
    # performs one step of one ped with respect to the new velocity
    
    ped_data.t[ped_idx] = ped_data.t[ped_idx] + [act_t]             # '+' works as append   
    
    new_x = ped_data.x[ped_idx][-1] + act_vx*const['dt']             # [-1] reffers to the last value            
    new_y = ped_data.y[ped_idx][-1] + act_vy*const['dt']                         
    
    # hard core zero-range wall repulsion
    if new_x < const['wx1']:
        new_x = const['wx1']
        act_vx = 0
    if new_x > const['wx2']:
        new_x = const['wx2']
        act_vx = 0    
    if new_y < const['wy1']:
        new_y = const['wy1']
        act_vy = 0
    if new_y > const['wy2']:
        new_y = const['wy2']
        act_vy = 0        
        
    
    ped_data.vx[ped_idx] = ped_data.vx[ped_idx] + [act_vx]  
    ped_data.vy[ped_idx] = ped_data.vy[ped_idx] + [act_vy] 
    
    ped_data.x[ped_idx] = ped_data.x[ped_idx] + [new_x]
    ped_data.y[ped_idx] = ped_data.y[ped_idx] + [new_y]
    
    
    return ped_data


def f_motivation(ped_data, ped_idx, const):
    
    sx =  (const['attractor_x'] - ped_data.x[ped_idx][-1]) / math.sqrt(
            (ped_data.x[ped_idx][-1] - const['attractor_x'])**2 
             + (ped_data.y[ped_idx][-1] - const['attractor_y'])**2)
    sy =  (const['attractor_y'] - ped_data.y[ped_idx][-1]) / math.sqrt(
             (ped_data.x[ped_idx][-1] - const['attractor_x'])**2 
             + (ped_data.y[ped_idx][-1] - const['attractor_y'])**2)
                
    fx = (const['v_opt']*sx - ped_data.vx[ped_idx][-1])/const['tau']
    fy = (const['v_opt']*sy - ped_data.vy[ped_idx][-1])/const['tau']

    return fx, fy


def f_ped_rep(ped_data, ped_idx, other_idx, const):
    
    if ped_idx == other_idx:
        fx = 0
        fy = 0
    else:
    
        d = math.sqrt((ped_data.x[ped_idx][-1] - ped_data.x[other_idx][-1])**2
                          + (ped_data.y[ped_idx][-1] - ped_data.y[other_idx][-1])**2)
        U = const['U0']/const['distance_scale'] * math.exp(-1*d/const['distance_scale'])
        
        sx =  (ped_data.x[other_idx][-1] - ped_data.x[ped_idx][-1]) / math.sqrt(
                (ped_data.x[ped_idx][-1] - ped_data.x[other_idx][-1])**2 
                + (ped_data.y[ped_idx][-1] - ped_data.y[other_idx][-1])**2)
        
        sy =  (ped_data.y[other_idx][-1] - ped_data.y[ped_idx][-1]) / math.sqrt(
                (ped_data.x[ped_idx][-1] - ped_data.x[other_idx][-1])**2 
                + (ped_data.y[ped_idx][-1] - ped_data.y[other_idx][-1])**2)
        
        fx = U*sx 
        fy = U*sy 

    return fx, fy


def f_wall(ped_data, ped_idx, const):
    
    if ped_data.x[ped_idx][-1] > const['wx2']/2:
        Ux = const['U0_w']/const['distance_scale_w'] * math.exp(-1*(const['wx2'] - ped_data.x[ped_idx][-1])/const['distance_scale_w'])
        fx = -1*Ux
    else:
        Ux = const['U0_w']/const['distance_scale_w'] * math.exp(-1*ped_data.x[ped_idx][-1]/const['distance_scale_w'])
        fx = Ux
    
    if ped_data.y[ped_idx][-1] > const['wy2']/2:
        Uy = const['U0_w']/const['distance_scale_w'] * math.exp(-1*(const['wy2'] - ped_data.y[ped_idx][-1])/const['distance_scale_w'])
        fy = -1*Uy
    else:
        Uy = const['U0_w']/const['distance_scale_w'] * math.exp(-1*ped_data.y[ped_idx][-1]/const['distance_scale_w'])
        fy = Uy

    return fx, fy


def update_v(ped_data, ped_idx, model_name, const):
    # With respect to the model and situation, new velocity is calculated here
    
    if model_name == 'no_change':
        new_vx = ped_data.vx[ped_idx][-1]
        new_vy = ped_data.vy[ped_idx][-1]
        
    elif model_name == 'motivation_only':
        fx, fy = f_motivation(ped_data, ped_idx, const)
        new_vx = ped_data.vx[ped_idx][-1] + fx*const['dt']
        new_vy = ped_data.vy[ped_idx][-1] + fy*const['dt']
        
    elif model_name == 'motivation_interaction':    
        
        fxa, fya = f_motivation(ped_data, ped_idx, const)
        
        fxp, fyp = calc_ped_interaction(ped_data, ped_idx, const)
  
        fx = fxa + fxp
        fy = fya + fyp
  
        new_vx = ped_data.vx[ped_idx][-1] + fx*const['dt']
        new_vy = ped_data.vy[ped_idx][-1] + fy*const['dt']
        
    elif model_name == 'motivation_interaction_wall':    
         
         fxa, fya = f_motivation(ped_data, ped_idx, const)
         
         fxp, fyp = calc_ped_interaction(ped_data, ped_idx, const)
   
         fxw, fyw = f_wall(ped_data, ped_idx, const) 
   
         fx = fxa + fxp + fxw
         fy = fya + fyp + fyw
   
         new_vx = ped_data.vx[ped_idx][-1] + fx*const['dt']
         new_vy = ped_data.vy[ped_idx][-1] + fy*const['dt']       
        
    else:
        new_vx = np.nan
        new_vy = np.nan
           
    return new_vx, new_vy
    

def calc_ped_interaction(ped_data, ped_idx, const):
    
    if np.isnan(ped_data.time_left[0]):
        fx0, fy0 = f_ped_rep(ped_data, ped_idx, 0, const)
    else:
        fx0 = 0
        fy0 = 0 
    if np.isnan(ped_data.time_left[1]):    
        fx1, fy1 = f_ped_rep(ped_data, ped_idx, 1, const)
    else:
        fx1 = 0
        fy1 = 0         
    if np.isnan(ped_data.time_left[2]):
        fx2, fy2 = f_ped_rep(ped_data, ped_idx, 2, const)
    else:
        fx2 = 0
        fy2 = 0         
    if np.isnan(ped_data.time_left[3]):
        fx3, fy3 = f_ped_rep(ped_data, ped_idx, 3, const)
    else:
        fx3 = 0
        fy3 = 0         
    if np.isnan(ped_data.time_left[4]):
        fx4, fy4 = f_ped_rep(ped_data, ped_idx, 4, const)
    else:
        fx4 = 0
        fy4 = 0         
        
    fx = -fx0 - fx1 - fx2 - fx3 - fx4
    fy = -fy0 - fy1 - fy2 - fy3 - fy4
    
    return fx, fy


def ped_exit(ped_data, j, act_t, const):
    # A pedestrian leaves the simulation if the distance 
    # to the exit is smaller than a constant 'exit_range'
    
    exit_dist = math.sqrt((ped_data.x[j][-1] - const['attractor_x'])**2 
                          + (ped_data.y[j][-1] - const['attractor_y'])**2)
    
    if exit_dist <= const['exit_range']:
        print('Ped ' + str(j) + ' left')
        ped_data.time_left[j] = act_t 
    
    return


#============================================#
#              SCRIPT STARTS HERE            #
#============================================#

#======================#
#     PRELIMINARIES    #
#======================#

# Constants - dictionary
const = {'N_ped': 5,                # numer of peds in the system
         'N_step': 100,             # number of steps
         'dt': 0.1,                 # diffrential step length [s]
         'v_opt': 3,                # optimal velocity (scalar) [m/s]
         'tau': 4,                  # motivation force parameter [s]
         'U0': 1,                   # ped intearction potential parameter [J.m]
         'U0_w': 0.1,               # ped intearction potential parameter [J.m]
         'distance_scale': 1,       # ped interaction distance parameter [m]
         'distance_scale_w': 0.5,   # wall interaction distance parameter [m]
         'attractor_x': 10,         # x position of attractor [m]
         'attractor_y': 5,          # y position of attractor [m]
         'exit_range': 0.2,         # distance where ped falls into the exit [m] 
         'wx1': 0,                  # wall coordinates [m] - rentangle expected by code, [0,0] expected corner
         'wx2': 10,
         'wy1': 0,
         'wy2': 7   
        }

# Init time, positions and velocities
t =  [  0,   0,   0,   0,    0]
x =  [1.1, 0.9,   1,   3, 3.05]
y =  [  1, 2.5,   4, 1.1, 2.60]
vx = [1.5, 1.5, 1.5, 1.5, 1.50]
vy = [  0,   0,   0,   0,    0]
ex = [np.nan, np.nan, np.nan, np.nan, np.nan]

# Init data containers
ped_data = init_ped_data(t, x, y, vx, vy, ex)



#======================#
#         MODEL        #
#======================#

# Model loop over time
rep = range(const['N_step'])
for i in rep: 

    act_t = (i+1)*const['dt']                         # i+1 is current itteration as i = 0  was defined in init step 
    
    # model loop over other peds 
    rep2 = range(const['N_ped'])
    for j in rep2: 
    
        if np.isnan(ped_data.time_left[j]):    
            act_vx, act_vy = update_v(ped_data, j, 'motivation_interaction_wall', const)
            ped_data = one_ped_step(ped_data, j, act_t, act_vx, act_vy, const)
    
            ped_exit(ped_data, j, act_t, const)

#======================#
#     POSTPROCESSING   #
#======================#


# Timespace fundamental diagram x
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
plt.plot(const['attractor_x'], const['attractor_y'], 'r*', label = 'ped 1')
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

# Timespace fundamental diagram y
plt.figure()
plt.plot(ped_data.t[0], ped_data.y[0], 'r-', label = 'ped 1')
plt.plot(ped_data.t[1], ped_data.y[1], 'g-', label = 'ped 2')
plt.plot(ped_data.t[2], ped_data.y[2], 'b-', label = 'ped 3')
plt.plot(ped_data.t[3], ped_data.y[3], 'k-', label = 'ped 4')
plt.plot(ped_data.t[4], ped_data.y[4], 'm-', label = 'ped 5')
plt.title('Timespace fundamental diagram')
plt.xlabel(r'$t \,\,\mathrm{[s]}$')
plt.ylabel(r'$y \,\,\, \mathrm{[m]}$')
#plt.xlim(0, 10)
#plt.ylim(0, 120)
plt.legend()
plt.show()

