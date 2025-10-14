import pandas as pd
import numpy as np 
import math as math
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
    
    
def update_positions(car_data, car_idx, F, t_new, delta_t):    
# provede update pozice a rychlosti jednoho vozidla na základě minulé pozice, rychlosti a síly
# uloží napočítané hodnoty do polí v car_data
    
    x_new = car_data.x[car_idx][-1] + delta_t*car_data.v[car_idx][-1]
    v_new = car_data.v[car_idx][-1] + delta_t*F
        
    car_data.at[car_idx,'x'] = car_data.x[car_idx]+[x_new]
    car_data.at[car_idx,'v'] = car_data.v[car_idx]+[v_new]
    car_data.at[car_idx,'t'] = car_data.t[car_idx]+[t_new]

    return car_data


def calculate_force(car_data, car_idx, model, t_safe, d_safe, v_opt, const):
# funkce vypočítá sílu působící na zadané vozidlo dle zadaného modelu    
    
    if model == 'zero_force':
        F = 0
        
    elif model == 'FLM':
        F = (car_data.v[car_idx-1][-2] - car_data.v[car_idx][-1])/t_safe
        
    elif model == 'OVM_hyp':
        dx = car_data.x[car_idx-1][-2] - car_data.x[car_idx][-1]
        v_opt_loc = 0.5*v_opt*(math.tanh(dx-d_safe) + math.tanh(d_safe))
        F = (v_opt_loc - car_data.v[car_idx][-1])/t_safe       

    elif model == 'IDM':
        dx = car_data.x[car_idx-1][-2] - car_data.x[car_idx][-1]
        dv = car_data.v[car_idx-1][-2] - car_data.v[car_idx][-1]
        
        d_star = max(0, d_safe 
                         + car_data.v[car_idx][-1]*t_safe
                         - car_data.v[car_idx][-1]*dv / (2*math.sqrt(const['idm_a']*const['idm_b'])) 
        )
        
        F = const['idm_a']*(1 - math.pow(car_data.v[car_idx][-1]/v_opt, const['idm_delta']) - math.pow(d_star/dx,2))
               
    else: 
        F = np.nan
    
    return F


#======================#
#      PARAMETRY       #
#======================#


t = [0,0,0,0,0]
x = [100,80,40,35,10]
v = [10,0,0,0,0]
d = [0]

#parametry simulace
delta_t = 0.02
t_max = 20
doba = int(t_max/delta_t)
pocet_vozidel = len(x)

t_safe = 2
d_safe = 2
v_opt = 20

const = {'idm_a': 0.8,              # IDM acceleration [m/s2]
         'idm_b': 1.5,              # IDM deceleration [m/s2]
         'idm_delta': 4             # IDM delta [1]
        }




#======================#
#         MODEL        #
#======================#


car_data = init_car_data(t, x, v, pocet_vozidel)


for k in range(1,doba):
    
    t_new = k*delta_t    
    
    F = 0
    car_data = update_positions(car_data, 0, F, t_new, delta_t)  
    
    
    for car_idx in range(pocet_vozidel - 1):

        F = calculate_force(car_data, car_idx+1, 'IDM', t_safe, d_safe, v_opt, const)        # IDM, OVM_hyp, FLM, zero_force
        
        car_data = update_positions(car_data, car_idx+1, F, t_new, delta_t)





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



















