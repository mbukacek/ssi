import pandas as pd
import numpy as np
import math as math
import random as rn
from matplotlib import pyplot as plt 

pd.options.mode.chained_assignment = None  # default='warn'

def init_car_data(t,x,vx):
    # initialize car data container
    # INPUT: vector of init time, init position and init velocity
    
    # init dataframe with the first ped
    car_data = pd.DataFrame({'ped_id': 0,
                             't': [[t[0]]],                         # to be list of recorded time
                             'x': [[x[0]]],                         # to be list of recorded x position
                             'vx': [[vx[0]]],                       # to be list of recorded x velocity
                             }, index = [0])

    # add peds one by one
    rep = range(len(x)-1)
    for i in rep:
        car_data_n = pd.DataFrame({'ped_id': i+1,
                                't': [[t[i+1]]],                         
                                'x': [[x[i+1]]],         
                                'vx': [[vx[i+1]]],
                                }, index = [i+1])
        car_data = pd.concat([car_data,car_data_n])
    
    return car_data


def one_car_step(car_data, ped_idx, act_t, act_vx, const):
    # performs one step of one ped with respect to the new velocity
    
    car_data.t[ped_idx] = car_data.t[ped_idx] + [act_t]             # '+' works as append   
    
    car_data.vx[ped_idx] = car_data.vx[ped_idx] + [act_vx]       
    
    new_x = car_data.x[ped_idx][-1] + act_vx*const['dt']             # [-1] reffers to the last value            
    car_data.x[ped_idx] = car_data.x[ped_idx] + [new_x]
    
    return car_data


def update_v(car_data, ped_idx, model_name, const):
    # With respect to the model and situation, new velocity is calculated here
    
    if model_name == '1D_NASH': 
    
        # acceleration
        new_vx = car_data.vx[ped_idx][-1] + 1
        # predecessor avoidance     
        if ped_idx == 0:
            x_space = 100
        else:
            x_space = car_data.x[ped_idx-1][-1] - car_data.x[ped_idx][-1]    # previous ped has been updated, i.e. made a space for our step 
        if x_space <= new_vx:
            new_vx = x_space - 1
        # velocity limits  
        if const['v_opt'] < new_vx:
            new_vx = const['v_opt']
        
        # random deceleration 
        r = rn.random()
        if (new_vx > 0) & (r > 0.5):
            new_vx = new_vx - 1
            
        if ped_idx == 0:    # první vozidlo slouží jako překážka
             new_vx = 0    
            
    else:
        new_vx = np.nan
    
    return new_vx


#============================================#
#              SCRIPT STARTS HERE            #
#============================================#


#======================#
#     PRELIMINARIES    #
#======================#

# Constants - dictionary
const = {'N_ped': 5,                # numer of peds in the system
         'N_step': 30,              # number of steps
         'dt': 1,                   # time step length [s]
         'v_opt': 7,                # optimal velocity (scalar) [cell/time_step]
         'attractor_x': 10,         # x position of attractor [cell]
        }

# Init time, positions and velocities
t =  [0,  0, 0, 0, 0]
x =  [50, 8, 6, 5, 1]
vx = [0,  0, 0, 0, 0]

rn.seed(42)

# Init data containers
car_data = init_car_data(t,x,vx)



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
    
        act_vx = update_v(car_data, j, '1D_NASH', const)
        car_data = one_car_step(car_data, j, act_t, act_vx, const)
    


#======================#
#     POSTPROCESSING   #
#======================#


# Analyze deceleration

acceleration = pd.Series()

rep = range(const['N_ped'])
for j in rep:
    acceleration = pd.concat([acceleration, pd.Series(car_data.vx[j]).diff(periods=1)])
    
acceleration = acceleration[acceleration.isna() == False]    
    
plt.figure()
plt.hist(acceleration, bins=[-8.5, -7.5, -6.5, -5.5, -4.5, -3.5, -2.5, -1.5, -0.5, 0.5, 1.5])
plt.title('max deceleration: ' + str(min(acceleration)))
plt.show()



# Timespace fundamental diagram
plt.figure()
plt.plot(car_data.t[0], car_data.x[0], 'r-o', label = 'ped 1')
plt.plot(car_data.t[1], car_data.x[1], 'g-o', label = 'ped 2')
plt.plot(car_data.t[2], car_data.x[2], 'b-o', label = 'ped 3')
plt.plot(car_data.t[3], car_data.x[3], 'k-o', label = 'ped 4')
plt.plot(car_data.t[4], car_data.x[4], 'm-o', label = 'ped 5')
plt.title('Timespace fundamental diagram')
plt.xlabel(r'$t \,\,\mathrm{[s]}$')
plt.ylabel(r'$x \,\,\, \mathrm{[m]}$')
#plt.xlim(0, 10)
#plt.ylim(0, 120)
plt.legend()
plt.show()


# velocity
plt.figure()
plt.plot(car_data.t[0], car_data.vx[0], 'r-o', label = 'ped 1')
plt.plot(car_data.t[1], car_data.vx[1], 'g-o', label = 'ped 2')
plt.plot(car_data.t[2], car_data.vx[2], 'b-o', label = 'ped 3')
plt.plot(car_data.t[3], car_data.vx[3], 'k-o', label = 'ped 4')
plt.plot(car_data.t[4], car_data.vx[4], 'm-o', label = 'ped 5')
plt.title('Velocity in time')
plt.xlabel(r'$t \,\,\mathrm{[s]}$')
plt.ylabel(r'$v \,\,\, \mathrm{[m/s]}$')
#plt.xlim(0, 10)
#plt.ylim(0, 120)
plt.legend()
plt.show()

