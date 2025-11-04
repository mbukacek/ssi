import pandas as pd
import numpy as np 
import math as math
import random as rn
from matplotlib import pyplot as plt 



#======================#
#         FUNKCE       #
#======================#

def init_car_data(t, x, v, pocet_vozidel):
# Funkce vytvoří dataframe car_data ...   
    
    car_data = pd.DataFrame({'car_id': 0,
                             't': [[t[0]]],                         
                             'x': [[x[0]]],                         
                             'v': [[v[0]]],
                                 }
                            , index = [0]
                            )
    #prvni zavorka jake auto, druha zavorka jaky cas
    rep = range(len(x)-1)
    for i in rep:
            car_data_n = pd.DataFrame({'car_id': i+1,
                                    't': [[t[i+1]]],                         
                                    'x': [[x[i+1]]],                         
                                    'v': [[v[i+1]]],
                                    }, index = [i+1])
            car_data = pd.concat([car_data,car_data_n])
           
    return car_data        
    
    
def update_positions(car_data, car_idx, v_new, t_new, delta_t):    
# provede update pozice a rychlosti jednoho vozidla na základě minulé pozice a aktualizované rychlosti
# uloží napočítané hodnoty do polí v car_data
    
    x_new = car_data.x[car_idx][-1] + v_new
        
    car_data.at[car_idx,'x'] = car_data.x[car_idx]+[x_new]
    car_data.at[car_idx,'v'] = car_data.v[car_idx]+[v_new]
    car_data.at[car_idx,'t'] = car_data.t[car_idx]+[t_new]

    return car_data


def update_velocity(car_data, car_idx, model, const):
# funkce zaktualizuje rychlost vozidlo dle zadaného modelu    
    
    if model == 'NASH':

        # actual velocity
        v_new = car_data.v[car_idx][-1]
        
        # velocity limits & acceleration
        if v_new < const['v_max']:
            v_new = v_new + 1
        
        # predecessor avoidance  
        if car_idx == 0:
            x_space = 1000
        else:
            x_space = car_data.x[car_idx-1][-1] - car_data.x[car_idx][-1] - 1    # previous ped has been updated, i.e. made a space for our step 
            
        if x_space <= v_new:
            v_new = x_space
        
        # random deceleration 
        r = rn.random()
        if (v_new > 0) & (r < const['q_rand']):
           v_new = v_new - 1
                          
    else: 
        v_new = np.nan
    
    return v_new


#======================#
#      PARAMETRY       #
#======================#


t = [0,0,0,0,0]
x = [10,8,4,3,1]        # index buňky, buňka ~ 10m
v = [0,0,0,0,0]         # pocet buněk za krok


const = {'v_max': 6,              # 
         'q_rand': 0.7            # 
        }

delta_t = 1             # čas v krocích, krok ~ 2s (-> rychlost 1 buňka / krok ~ 18 km/h)
pocet_vozidel = 5
doba = 50

#======================#
#         MODEL        #
#======================#


car_data = init_car_data(t, x, v, pocet_vozidel)


for k in range(1,doba):
    
    t_new = k*delta_t    
    
    
    for car_idx in range(pocet_vozidel):

        v_new = update_velocity(car_data, car_idx, 'NASH', const)       
        
        car_data = update_positions(car_data, car_idx, v_new, t_new, delta_t)





# Velocity of cars
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



















