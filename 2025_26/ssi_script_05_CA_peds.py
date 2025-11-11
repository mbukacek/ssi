import pandas as pd
import numpy as np 
import math as math
import random as rn

from matplotlib import pyplot as plt 



#======================#
#         FUNKCE       #
#======================#

def init_ped_data(t, x, y, pocet_chodcu):
# Funkce vytvoří dataframe ped_data ...   
    
    ped_data = pd.DataFrame({'ped_id': 0,
                             't': [[t[0]]],                         
                             'x': [[x[0]]],   
                             'y': [[y[0]]], 
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
                                    }, index = [i+1])
            ped_data = pd.concat([ped_data, ped_data_n])
           
    return ped_data        
    

def cell_guest(ped_data, x, y):
    
    ped_id = np.nan

    rep_k = range(const['N_ped'])                                    
    for k in rep_k:
        
        if (ped_data.x[k][-1] == x) & (ped_data.y[k][-1] == y):
            ped_id = ped_data.ped_id[k]
    
    return ped_id

    
def update_positions(ped_data, distance_grid, ped_idx, dec_x, dec_y, t_new, delta_t):    
# provede update pozice a základě minulé pozice a rozhodnutí, pokud je volno. Jinak uloží stávající pozici
# uloží napočítané hodnoty do polí v car_data
 
    if (pd.isna(cell_guest(ped_data, dec_x, dec_y)) & (distance_grid[dec_x][dec_y] < np.inf)):       # empty and not wall    
        ped_data.at[ped_idx,'x'] = ped_data.x[ped_idx]+[dec_x]
        ped_data.at[ped_idx,'y'] = ped_data.y[ped_idx]+[dec_y]
    else:
        ped_data.at[ped_idx,'x'] = ped_data.x[ped_idx]+[ped_data.x[ped_idx][-1]]
        ped_data.at[ped_idx,'y'] = ped_data.y[ped_idx]+[ped_data.y[ped_idx][-1]]
        
    ped_data.at[ped_idx,'t'] = ped_data.t[ped_idx]+[t_new]
        

    return ped_data


def ped_make_decision(ped_data, distance_grid, ped_idx, model, const):
# funkce vybere novou cílovou pozici chodce   

    x = ped_data.x[ped_idx][-1]         # present position
    y = ped_data.y[ped_idx][-1]
    
    if model == 'Random':               # random pick from Moor neighborhood
        dec_x = x + rn.randint(-1,1)        # includes both boundaries
        dec_y = y + rn.randint(-1,1)     
    
    elif model == 'Atractor':           # pick based on distance grid from von Neuman neighborhood
        
        d_xy = distance_grid[x][y]
        d_xu = distance_grid[x][y+1]
        d_xd = distance_grid[x][y-1]
        d_ly = distance_grid[x-1][y]
        d_ry = distance_grid[x+1][y]
        
        norm = math.exp(-1*const['k_s']*d_xy) + math.exp(-1*const['k_s']*d_xu) + math.exp(-1*const['k_s']*d_xd) \
                + math.exp(-1*const['k_s']*d_ly) + math.exp(-1*const['k_s']*d_ry)
                
        p_xy =  math.exp(-1*const['k_s']*d_xy)/norm
        p_xu =  math.exp(-1*const['k_s']*d_xu)/norm
        p_xd =  math.exp(-1*const['k_s']*d_xd)/norm
        p_ly =  math.exp(-1*const['k_s']*d_ly)/norm
        p_ry =  math.exp(-1*const['k_s']*d_ry)/norm
                
        r = rn.random()
        if r < p_xy:
            dec_x = x
            dec_y = y
        elif r < (p_xy + p_xu):
            dec_x = x
            dec_y = y+1
        elif r < (p_xy + p_xu + p_xd):
            dec_x = x
            dec_y = y-1
        elif r < (p_xy + p_xu + p_xd + p_ly):
            dec_x = x-1
            dec_y = y
        else:
            dec_x = x+1
            dec_y = y

    else: 
        dec_x = np.nan
        dec_y = np.nan
    
    return dec_x, dec_y


#======================#
#      PARAMETRY       #
#======================#


t = [0,0,0,0,0]
x = [3,3,1,1,2]        # index buňky, buňka ~ 0.5 m x 0.5 m
y = [1,5,3,4,5]


const = {'N_ped': 5,              #
         'k_s': 1,                # sensitivity to static fiels
         'grid_size_x': 4,        # x: number of rows
         'grid_size_y': 5,        # y: number of columns    
         'attractor_x': 4,        # x position of attractor [cell]
         'attractor_y': 2         # y position of attractor [cell]
        }

distance_grid = [[np.inf,  np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
                 [np.inf,       4,      3,      4,      5,      6, np.inf], 
                 [np.inf,       3,      2,      3,      4,      5, np.inf], 
                 [np.inf,       2,      1,      2,      3,      4, np.inf], 
                 [np.inf,       1,      0,      1,      2,      3, np.inf],
                 [np.inf,  np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]]


delta_t = 1             # čas v krocích, krok ~ 2s (-> rychlost 1 buňka / krok ~ 18 km/h)
pocet_chodcu = 5
doba = 20

plt.ion()       # turn on interactive mode of plots to see them in real time

#======================#
#         MODEL        #
#======================#


ped_data = init_ped_data(t, x, y, pocet_chodcu)

plt.figure()
plt.xlim(-2, 10)
plt.ylim(-2, 10)

for k in range(1,doba):
    
    t_new = k*delta_t    
    
    for ped_idx in range(pocet_chodcu):
        
        dec_x, dec_y = ped_make_decision(ped_data, distance_grid, ped_idx, 'Atractor', const)
        
        ped_data = update_positions(ped_data, distance_grid, ped_idx, dec_x, dec_y, t_new, delta_t)


    plt.plot(ped_data.x[0], ped_data.y[0], 'r-o', label = 'ped 1')
    plt.plot(ped_data.x[1], ped_data.y[1], 'g-o', label = 'ped 2')
    plt.plot(ped_data.x[2], ped_data.y[2], 'b-o', label = 'ped 3')
    plt.plot(ped_data.x[3], ped_data.y[3], 'k-o', label = 'ped 4')
    plt.plot(ped_data.x[4], ped_data.y[4], 'm-o', label = 'ped 5')

    plt.draw()          # překreslí aktuální figure
    plt.pause(0.5)


plt.ioff()
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
plt.xlim(-2, 10)
plt.ylim(-2, 10)
plt.legend()
plt.show()


# Velocity of cars
plt.figure()
plt.plot(ped_data.t[0], ped_data.v[0], 'r-', label = 'car 1')
plt.plot(ped_data.t[1], ped_data.v[1], 'g-', label = 'car 2')
plt.plot(ped_data.t[2], ped_data.v[2], 'b-', label = 'car 3')
plt.plot(ped_data.t[3], ped_data.v[3], 'k-', label = 'car 4')
plt.plot(ped_data.t[4], ped_data.v[4], 'm-', label = 'car 5')
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



















