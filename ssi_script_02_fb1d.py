import pandas as pd
import numpy as np
import math as math
from matplotlib import pyplot as plt 


def init_car_data(t,x,v):
    # initialize car data container
    # INPUT: vector of init time, init position and init velocity
    
    # init dataframe with the first car
    car_data = pd.DataFrame({'car_id': 0,
                             't': [[t[0]]],                         # to be list of recorded time
                             'x': [[x[0]]],                         # to be list of recorded position
                             'v': [[v[0]]]                          # to be list of recorded velocity
                             }, index = [0])

    # add cars one by one
    rep = range(len(x)-1)
    for i in rep:
        car_data_n = pd.DataFrame({'car_id': i+1,
                                't': [[t[i+1]]],                         
                                'x': [[x[i+1]]],                         
                                'v': [[v[i+1]]]                          
                                }, index = [i+1])
        car_data = car_data.append(car_data_n)
    
    return car_data


def one_car_step(car_data, car_idx, act_t, act_v, const):
    # performs one step of one car with respect to the new velocity
    
    car_data.t[car_idx] = car_data.t[car_idx] + [act_t]             # '+' works as append   
    
    car_data.v[car_idx] = car_data.v[car_idx] + [act_v]      
    
    new_x = car_data.x[car_idx][-1] + act_v*const['dt']             # [-1] reffers to the last value            
    car_data.x[car_idx] = car_data.x[car_idx] + [new_x]
    
    return car_data


def update_v(car_data, car_idx, model_name, const):
    # With respect to the model and situation, new velocity is calculated here
    
    # TO DO: Implement FLM, OVM, IDM 
    
    if model_name == 'Trivial':
        new_v = 1
        
    elif model_name == 'FLM':
        f = (car_data.v[car_idx-1][-2] - car_data.v[car_idx][-1])/const['t_safe']   # previous car has been updated, thus we need second last value [-2]
        new_v = car_data.v[car_idx][-1] + f*const['dt']
        
    elif model_name == 'OVM_hyp':
        dx = car_data.x[car_idx-1][-2] - car_data.x[car_idx][-1]
        v_opt_loc = 0.5*const['v_opt']*(math.tanh(dx-const['d_safe']) + math.tanh(const['d_safe']))
        f = (v_opt_loc - car_data.v[car_idx][-1])/const['t_safe']
        new_v = car_data.v[car_idx][-1] + f*const['dt']
        
    elif model_name == 'IDM':
        dx = car_data.x[car_idx-1][-2] - car_data.x[car_idx][-1]
        dv = car_data.v[car_idx-1][-2] - car_data.v[car_idx][-1]
        
        d_star = max(0, const['d_safe'] 
                         + car_data.v[car_idx][-1]*const['t_safe'] 
                         - car_data.v[car_idx][-1]*dv / (2*math.sqrt(const['idm_a']*const['idm_b'])) 
        )
        
        f = const['idm_a']*(1 - math.pow(car_data.v[car_idx][-1]/const['v_opt'], const['idm_delta']) - math.pow(d_star/dx,2))
        
        new_v = car_data.v[car_idx][-1] + f*const['dt']
        
    else:
        new_v = np.nan
    
    return new_v


#============================================#
#              SCRIPT STARTS HERE            #
#============================================#

#======================#
#     PRELIMINARIES    #
#======================#

# Constants - dictionary
const = {'N_car': 5,                # numer of cars in the system
         'N_step': 500,             # number of steps
         'v_opt': 20,               # optimal velocity [m/s] - static, the same for any car
         'dt': 0.1,                 # diffrential step length [s]
         'd_safe': 4,              # minimal safe spacial distance to previous car [m]
         't_safe': 3,               # minimal safe time distance to previous car [s]
         'idm_a': 0.8,              # IDM acceleration [m/s2]
         'idm_b': 1.5,              # IDM deceleration [m/s2]
         'idm_delta': 4             # IDM delta [1]
        }

# Init time, positions and velocities
t = [0, 0, 0, 0, 0]
x = [100, 80, 40, 35, 10]
v = [10, 0, 0, 0, 0]

# Init data containers
car_data = init_car_data(t,x,v)



#======================#
#         MODEL        #
#======================#

# Model loop over time
rep = range(const['N_step'])
for i in rep: 

    act_t = (i+1)*const['dt']                                       # i+1 is current itteration as i = 0  was defined in init step 
    
    # first car - trivial model      
    act_v = 10 + 10* np.sin(i*np.pi/8)               # harmonicaly adjust 
    car_data = one_car_step(car_data, 0, act_t, act_v, const)

    # model loop over other cars 
    rep2 = range(const['N_car'] - 1)
    for j in rep2: 
    
        act_v = update_v(car_data, j+1, 'IDM', const)
        car_data = one_car_step(car_data, j+1, act_t, act_v, const)
    


#======================#
#     POSTPROCESSING   #
#======================#


# Velocity of the first car
plt.figure()
plt.plot(car_data.t[0], car_data.v[0], 'r-', label = 'car 1')
plt.plot(car_data.t[1], car_data.v[1], 'g-', label = 'car 2')
plt.plot(car_data.t[2], car_data.v[2], 'b-', label = 'car 3')
plt.plot(car_data.t[3], car_data.v[3], 'k-', label = 'car 4')
plt.plot(car_data.t[4], car_data.v[4], 'm-', label = 'car 5')
plt.title('Velocity in time')
plt.show()

# Timespace fundamental diagram
plt.figure()
plt.plot(car_data.t[0], car_data.x[0], 'r-', label = 'car 1')
plt.plot(car_data.t[1], car_data.x[1], 'g-', label = 'car 2')
plt.plot(car_data.t[2], car_data.x[2], 'b-', label = 'car 3')
plt.plot(car_data.t[3], car_data.x[3], 'k-', label = 'car 4')
plt.plot(car_data.t[4], car_data.x[4], 'm-', label = 'car 5')
plt.title('Timespace fundamental diagram')
plt.xlabel(r'$t \,\,\mathrm{[s]}$')
plt.ylabel(r'$x \,\,\, \mathrm{[m]}$')
#plt.xlim(0, 10)
#plt.ylim(0, 120)
plt.legend()
plt.show()



