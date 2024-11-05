import pandas as pd
import numpy as np
import math as math
from matplotlib import pyplot as plt 

def init_car_data(t,x,y,vx,vy):
    # initialize car data container
    # INPUT: vector of init time, init position and init velocity
    
    # init dataframe with the first ped
    car_data = pd.DataFrame({'ped_id': 0,
                             't': [[t[0]]],                         # to be list of recorded time
                             'x': [[x[0]]],                         # to be list of recorded x position
                             'y': [[y[0]]],                         # to be list of recorded y position
                             'vx': [[vx[0]]],                       # to be list of recorded x velocity
                             'vy': [[vy[0]]]                        # to be list of recorded y velocity
                             }, index = [0])

    # add peds one by one
    rep = range(len(x)-1)
    for i in rep:
        car_data_n = pd.DataFrame({'ped_id': i+1,
                                't': [[t[i+1]]],                         
                                'x': [[x[i+1]]],   
                                'y': [[y[i+1]]],      
                                'vx': [[vx[i+1]]],
                                'vy': [[vy[i+1]]]  
                                }, index = [i+1])
        car_data = pd.concat([car_data,car_data_n])
    
    return car_data


def one_car_step(car_data, ped_idx, act_t, act_vx, act_vy, const):
    # performs one step of one ped with respect to the new velocity
    
    car_data.t[ped_idx] = car_data.t[ped_idx] + [act_t]             # '+' works as append   
    
    car_data.vx[ped_idx] = car_data.vx[ped_idx] + [act_vx]  
    car_data.vy[ped_idx] = car_data.vy[ped_idx] + [act_vy]       
    
    new_x = car_data.x[ped_idx][-1] + act_vx*const['dt']             # [-1] reffers to the last value            
    car_data.x[ped_idx] = car_data.x[ped_idx] + [new_x]
    
    new_y = car_data.y[ped_idx][-1] + act_vy*const['dt']                         
    car_data.y[ped_idx] = car_data.y[ped_idx] + [new_y]
    
    return car_data


def update_v(car_data, ped_idx, model_name, const):
    # With respect to the model and situation, new velocity is calculated here
    
    # TO improve ..
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
         'dt': 1,                   # time step length [s]
         'v_opt': 3,                # optimal velocity (scalar) [cell/time_step]
         'attractor_x': 10,         # x position of attractor [cell]
         'attractor_y': 5           # y position of attractor [cell]
        }

# Init time, positions and velocities
t =  [0,  0, 0, 0, 0]
x =  [12, 8, 6, 5, 1]
y =  [5,  5, 5, 5, 5]
vx = [0,  0, 0, 0, 0]
vy = [0,  0, 0, 0, 0]

# Init data containers
car_data = init_car_data(t,x,y,vx,vy)



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
    
        act_vx, act_vy = update_v(car_data, j, '1D_NASH', const)
        car_data = one_car_step(car_data, j, act_t, act_vx, act_vy, const)
    


#======================#
#     POSTPROCESSING   #
#======================#


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

# Aerial plot
plt.figure()
plt.plot(const['attractor_x'], const['attractor_y'], 'r*', label = 'ped 1')
plt.plot(car_data.x[0], car_data.y[0], 'r-o', label = 'ped 1')
plt.plot(car_data.x[1], car_data.y[1], 'g-o', label = 'ped 2')
plt.plot(car_data.x[2], car_data.y[2], 'b-o', label = 'ped 3')
plt.plot(car_data.x[3], car_data.y[3], 'k-o', label = 'ped 4')
plt.plot(car_data.x[4], car_data.y[4], 'm-o', label = 'ped 5')
plt.title('Aerial plot')
plt.xlabel(r'$x \,\,\mathrm{[m]}$')
plt.ylabel(r'$y \,\,\, \mathrm{[m]}$')
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

