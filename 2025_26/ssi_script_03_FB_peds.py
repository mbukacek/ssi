import pandas as pd
import numpy as np 
import math as math
from matplotlib import pyplot as plt 



#======================#
#         FUNKCE       #
#======================#

def init_ped_data(t, x, y, vx, vy, pocet_chodcu):
# Funkce vytvoří dataframe ped_data ...   
    
    ped_data = pd.DataFrame({'ped_id': 0,
                             't': [[t[0]]],                         
                             'x': [[x[0]]],                         
                             'y': [[y[0]]],
                             'vx': [[vx[0]]],                         
                             'vy': [[vy[0]]],
                                 }
                            , index = [0]
                            )
    #prvni zavorka jake auto, druha zavorka jaky cas
    rep = range(len(x)-1)
    for i in rep:
            ped_data_n = pd.DataFrame({'ped_id': i+1,
                                    't': [[t[i+1]]],                         
                                    'x': [[x[i+1]]],                         
                                    'y': [[y[i+1]]],
                                    'vx': [[vx[i+1]]],                         
                                    'vy': [[vy[i+1]]],
                                    }, index = [i+1])
            ped_data = pd.concat([ped_data,ped_data_n])
           
    return ped_data        
    
    
def update_positions(ped_data, ped_idx, Fx, Fy, t_new, delta_t):    
# provede update pozice a rychlosti jednoho chodce na základě minulé pozice, rychlosti a síly
# uloží napočítané hodnoty do polí v ped_data
    
    x_new = ped_data.x[ped_idx][-1] + delta_t*ped_data.vx[ped_idx][-1]
    y_new = ped_data.y[ped_idx][-1] + delta_t*ped_data.vy[ped_idx][-1]
    vx_new = ped_data.vx[ped_idx][-1] + delta_t*Fx
    vy_new = ped_data.vy[ped_idx][-1] + delta_t*Fy
        
    ped_data.at[ped_idx,'x'] = ped_data.x[ped_idx]+[x_new]
    ped_data.at[ped_idx,'y'] = ped_data.y[ped_idx]+[y_new]
    ped_data.at[ped_idx,'vx'] = ped_data.vx[ped_idx]+[vx_new]
    ped_data.at[ped_idx,'vy'] = ped_data.vy[ped_idx]+[vy_new]
    ped_data.at[ped_idx,'t'] = ped_data.t[ped_idx]+[t_new]

    return ped_data


def calculate_force(ped_data, ped_idx, model, const):
# funkce vypočítá sílu působící na zadané vozidlo dle zadaného modelu    
    
    if model == 'zero_force':
        Fx = 0
        Fy = 0
        
  
               
    else: 
        Fx = np.nan
        Fy = np.nan
    
    return Fx, Fy


#======================#
#      PARAMETRY       #
#======================#


t = [0,0,0,0,0]
x = [100,80,40,35,10]
y = [0,0,0,0,0]
vx = [10,0,0,0,0]
vy = [0,0,0,0,0]
d = [0]

#parametry simulace
delta_t = 0.02
t_max = 20
doba = int(t_max/delta_t)
pocet_chodcu = len(x)



const = {'a': 1,              # tmp
        }




#======================#
#         MODEL        #
#======================#


ped_data = init_ped_data(t, x, y, vx, vy, pocet_chodcu)


for k in range(1, doba):
    
    t_new = k*delta_t    
        
    for ped_idx in range(pocet_chodcu):

        Fx, Fy = calculate_force(ped_data, ped_idx, 'zero_force', const)        #  zero_force
        
        ped_data = update_positions(ped_data, ped_idx, Fx, Fy, t_new, delta_t)





# Velocity of peds
plt.figure()
plt.plot(ped_data.t[0], ped_data.vx[0], 'r-', label = 'car 1')
plt.plot(ped_data.t[1], ped_data.vx[1], 'g-', label = 'car 2')
plt.plot(ped_data.t[2], ped_data.vx[2], 'b-', label = 'car 3')
plt.plot(ped_data.t[3], ped_data.vx[3], 'k-', label = 'car 4')
plt.plot(ped_data.t[4], ped_data.vx[4], 'm-', label = 'car 5')
plt.title('Velocity in time')
plt.show()

# Timespace fundamental diagram
plt.figure()
plt.plot(ped_data.t[0], ped_data.x[0], 'r-', label = 'car 1')
plt.plot(ped_data.t[1], ped_data.x[1], 'g-', label = 'car 2')
plt.plot(ped_data.t[2], ped_data.x[2], 'b-', label = 'car 3')
plt.plot(ped_data.t[3], ped_data.x[3], 'k-', label = 'car 4')
plt.plot(ped_data.t[4], ped_data.x[4], 'm-', label = 'car 5')
plt.title('Timespace fundamental diagram')
plt.xlabel(r'$t \,\,\mathrm{[s]}$')
plt.ylabel(r'$x \,\,\, \mathrm{[m]}$')
#plt.xlim(0, 10)
#plt.ylim(0, 120)
plt.legend()
plt.show()



















