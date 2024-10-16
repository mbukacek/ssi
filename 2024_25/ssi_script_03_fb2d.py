import pandas as pd
import numpy as np
import math as math
from matplotlib import pyplot as plt 

pd.options.mode.chained_assignment = None  # default='warn'

# https://pedpy.readthedocs.io/en/stable/

def init_peds(N_ped, const):
    # set up initial pedestrian positions and create data container 
    
        
    const['N_ped'] = N_ped
    
    t =  np.random.rand(const['N_ped'])*0
    x =  np.random.rand(const['N_ped'])*const['wx2']
    y =  np.random.rand(const['N_ped'])*const['wy2']
    vx = np.random.rand(const['N_ped'])*0
    vy = np.random.rand(const['N_ped'])*0
    ex = np.random.rand(const['N_ped'])*np.nan
    
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
    
    return ped_data, const


def one_ped_step(ped_data, ped_idx, act_t, act_vx, act_vy, const):
    # performs one step of one ped with respect to the new velocity
    
    ped_data.t[ped_idx] = ped_data.t[ped_idx] + [act_t]             # '+' works as append   
    
    new_x = ped_data.x[ped_idx][-1] + act_vx*const['dt']            # [-1] reffers to the last value            
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


def calc_ped_interaction(ped_data, ped_idx, const):
    # aggregates contribution from all active pedestrians
    fx = 0
    fy = 0
    
    rep = range(const['N_ped'])
    for k in rep:
        if np.isnan(ped_data.time_left[0]):
            fx_one, fy_one = f_ped_rep(ped_data, ped_idx, 0, const)
            fx = fx - fx_one
            fy = fy - fy_one
    
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
        fx = 0
        fy = 0
        
    elif model_name == 'motivation_only':
        fx, fy = f_motivation(ped_data, ped_idx, const)
        
    elif model_name == 'motivation_interaction':    
        fxa, fya = f_motivation(ped_data, ped_idx, const)
        fxp, fyp = calc_ped_interaction(ped_data, ped_idx, const)
        fx = fxa + fxp
        fy = fya + fyp
        
    elif model_name == 'motivation_interaction_wall':    
         fxa, fya = f_motivation(ped_data, ped_idx, const)
         fxp, fyp = calc_ped_interaction(ped_data, ped_idx, const)
         fxw, fyw = f_wall(ped_data, ped_idx, const) 
         fx = fxa + fxp + fxw
         fy = fya + fyp + fyw      
        
    else:
        fx = np.nan
        fy = np.nan
        
    new_vx = ped_data.vx[ped_idx][-1] + fx*const['dt']
    new_vy = ped_data.vy[ped_idx][-1] + fy*const['dt']     
        
    return new_vx, new_vy
    

def ped_exit(ped_data, j, act_t, const):
    # A pedestrian leaves the simulation if the distance 
    # to the exit is smaller than a constant 'exit_range'
    
    exit_dist = math.sqrt((ped_data.x[j][-1] - const['attractor_x'])**2 
                          + (ped_data.y[j][-1] - const['attractor_y'])**2)
    
    if exit_dist <= const['exit_range']:
#       print('Ped ' + str(j) + ' left')
        ped_data.time_left[j] = act_t 
        exit_indication = True
    else:
        exit_indication = False
    
    return exit_indication


def one_model_run(ped_data, const):
    # executes one model loop from initial position to the exit of the last ped 
    
    evacuation_time = np.inf
    
    i = 0                           # iteration counter
    active_num = const['N_ped']     # active ped counter
    
    while ((i < const['N_step']) & (active_num > 0)):   # anyone active and 
    
        act_t = (i+1)*const['dt']                       # i+1 is current itteration as i = 0  was defined in init step 
        
        rep2 = range(const['N_ped'])                    # model loop over other peds 
        for j in rep2: 
        
            if np.isnan(ped_data.time_left[j]):    
                act_vx, act_vy = update_v(ped_data, j, 'motivation_interaction_wall', const)
                ped_data = one_ped_step(ped_data, j, act_t, act_vx, act_vy, const)
                exit_indication = ped_exit(ped_data, j, act_t, const)
                
                if exit_indication:                 # adjust 
                    active_num = active_num - 1
        i = i+1            
        
    if active_num == 0:                  # in case all peds made it before max iteration count
        evacuation_time = act_t
                
    return evacuation_time, ped_data

#============================================#
#              SCRIPT STARTS HERE            #
#============================================#

#======================#
#     PRELIMINARIES    #
#======================#

# Constants - dictionary
const = {'N_ped': np.nan,           # numer of peds in the system - to be set in init function
         'N_step': 500,             # maximum number of model steps
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

# Simulation set up .. ~ 32 min 

max_occupancy = 40
occupancy_range = range(max_occupancy)                  # range of occupancies of model
iteration_count = 10                                    # number of model iterations for each set up

evacuation_time = np.random.rand(iteration_count,max_occupancy)*np.nan    # container for model output

#======================#
#      SIMULATIONS     #
#======================#

for n in occupancy_range: 
    
    it_range = range(iteration_count)
    for k in it_range:
    
        print('Model with occupancy ' + str(n+1) + ' iteration ' + str(k+1) + ' started')
    
        ped_data, const = init_peds(n+1, const)                     # CREATES MODEL DATA CONTAINER
    
        time_last_left, ped_data = one_model_run(ped_data, const)   # RUNS MODEL
        
        print('    evacuation time: ' + str(time_last_left))
        
        evacuation_time[k][n] = time_last_left
    



#======================#
#     POSTPROCESSING   #
#======================#

plt.figure()
ax = plt.boxplot(evacuation_time)


# # Timespace fundamental diagram x
# plt.figure()
# plt.plot(ped_data.t[0], ped_data.x[0], 'r-', label = 'ped 1')
# plt.plot(ped_data.t[1], ped_data.x[1], 'g-', label = 'ped 2')
# plt.plot(ped_data.t[2], ped_data.x[2], 'b-', label = 'ped 3')
# plt.plot(ped_data.t[3], ped_data.x[3], 'k-', label = 'ped 4')
# plt.plot(ped_data.t[4], ped_data.x[4], 'm-', label = 'ped 5')
# plt.title('Timespace fundamental diagram')
# plt.xlabel(r'$t \,\,\mathrm{[s]}$')
# plt.ylabel(r'$x \,\,\, \mathrm{[m]}$')
# #plt.xlim(0, 10)
# #plt.ylim(0, 120)
# plt.legend()
# plt.show()

# # Aerial plot
# plt.figure()
# plt.plot(const['attractor_x'], const['attractor_y'], 'r*', label = 'ped 1')
# plt.plot(ped_data.x[0], ped_data.y[0], 'r-o', label = 'ped 1')
# plt.plot(ped_data.x[1], ped_data.y[1], 'g-o', label = 'ped 2')
# plt.plot(ped_data.x[2], ped_data.y[2], 'b-o', label = 'ped 3')
# plt.plot(ped_data.x[3], ped_data.y[3], 'k-o', label = 'ped 4')
# plt.plot(ped_data.x[4], ped_data.y[4], 'm-o', label = 'ped 5')
# plt.title('Aerial plot')
# plt.xlabel(r'$x \,\,\mathrm{[m]}$')
# plt.ylabel(r'$y \,\,\, \mathrm{[m]}$')
# #plt.xlim(0, 10)
# #plt.ylim(0, 120)
# plt.legend()
# plt.show()

# # Timespace fundamental diagram y
# plt.figure()
# plt.plot(ped_data.t[0], ped_data.y[0], 'r-', label = 'ped 1')
# plt.plot(ped_data.t[1], ped_data.y[1], 'g-', label = 'ped 2')
# plt.plot(ped_data.t[2], ped_data.y[2], 'b-', label = 'ped 3')
# plt.plot(ped_data.t[3], ped_data.y[3], 'k-', label = 'ped 4')
# plt.plot(ped_data.t[4], ped_data.y[4], 'm-', label = 'ped 5')
# plt.title('Timespace fundamental diagram')
# plt.xlabel(r'$t \,\,\mathrm{[s]}$')
# plt.ylabel(r'$y \,\,\, \mathrm{[m]}$')
# #plt.xlim(0, 10)
# #plt.ylim(0, 120)
# plt.legend()
# plt.show()

