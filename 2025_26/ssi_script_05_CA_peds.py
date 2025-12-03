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
                             'dec_x': np.nan,
                             'dec_y': np.nan
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
                                    'dec_x': np.nan,
                                    'dec_y': np.nan
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

    
def update_positions(ped_data, distance_grid, ped_idx, t_new, delta_t):    
# provede update pozice a základě minulé pozice a rozhodnutí, pokud je volno. Jinak uloží stávající pozici
# uloží napočítané hodnoty do polí v car_data
 
    dec_x = int(ped_data.dec_x[ped_idx])
    dec_y = int(ped_data.dec_y[ped_idx])

    if (pd.isna(cell_guest(ped_data, dec_x, dec_y)) & (distance_grid[dec_x][dec_y] < np.inf)):       # empty and not wall    
        ped_data.at[ped_idx,'x'] = ped_data.x[ped_idx]+[dec_x]
        ped_data.at[ped_idx,'y'] = ped_data.y[ped_idx]+[dec_y]
    else:
        ped_data.at[ped_idx,'x'] = ped_data.x[ped_idx]+[ped_data.x[ped_idx][-1]]
        ped_data.at[ped_idx,'y'] = ped_data.y[ped_idx]+[ped_data.y[ped_idx][-1]]
        
    ped_data.at[ped_idx,'t'] = ped_data.t[ped_idx]+[t_new]
    
    ped_data.dec_x[ped_idx] = np.nan
    ped_data.dec_y[ped_idx] = np.nan   

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
        
    ped_data.dec_x[ped_idx] = dec_x
    ped_data.dec_y[ped_idx] = dec_y
    
    return ped_data


def resolve_conflicts(ped_data, const):
    
    print('   Conflict resolution started')
    
    rep_x = range(const['grid_size_x'])                                         # For all cells
    for i in rep_x: 
        rep_y = range(const['grid_size_y'])
        for j in rep_y: 
    
            ped_conf = []                                                       # Initiate empty "waiting room"
            
            rep_k = range(const['N_ped']-1)                                     # For all peds
            for k in rep_k:
                
                if (ped_data.dec_x[k] == i) & (ped_data.dec_y[k] == j):         # Check whether they want to enther this cell 
                    ped_conf = ped_conf + [k]                                   # If so, they are written to waiting list    
            
            if len(ped_conf) > 1:                                               # If waiting room is occupied by more than 2 peds
                r = rn.randint(0,len(ped_conf)-1)                               # Pick one randomly to keep his decision
                       
                rep_id = range(len(ped_conf))   
                for p in rep_id:                                                # Others will change they mind to stay at their positions
                    if p != r:
                        ped_data.dec_x[p] = ped_data.x[p][-1]
                        ped_data.dec_y[p] = ped_data.y[p][-1]
                        print('     Ped ' + str(ped_conf[p]) + ' blocked by conflict')
       
    return ped_data  


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
    print('  Time: ' + str(t_new))
    
    # Decision:
    for ped_idx in range(pocet_chodcu):
        ped_data = ped_make_decision(ped_data, distance_grid, ped_idx, 'Atractor', const)
       
    # Conflicts:
    ped_data = resolve_conflicts(ped_data, const)
        
        
    # Movement:
    peds_to_move = ped_data.ped_id[~ped_data.dec_x.isna()]                  # Initialy, chance to move is defined as True if at least one ped has decision
    chance_to_move = len(peds_to_move) > 0
        
    while chance_to_move:                                                   # We may need more loops in case of complex blocking situation
                                                                                # The loop will repeated if there was at least one move in previous one
        chance_to_move = False          
        peds_to_move = ped_data.ped_id[~ped_data.dec_x.isna()]   
        peds_to_move.reset_index(inplace=True, drop=True)
        
        rep_k = range(len(peds_to_move))                                    # For all peds that may move
        for k in rep_k:
            blocking_ped = cell_guest(ped_data, ped_data.dec_x[peds_to_move[k]], ped_data.dec_y[peds_to_move[k]])   # Who is in his desired cell
            
            if blocking_ped == ped_data.ped_id[peds_to_move[k]]:                            # Ped can't block himself
                blocking_ped = np.nan
            
            if pd.isna(blocking_ped):                                                       # Noone is blocking -> move
                ped_data = update_positions(ped_data, distance_grid, peds_to_move[k], t_new, delta_t)
                chance_to_move = True
                print('     Ped ' + str(peds_to_move[k]) + ' moved')
               
            elif pd.isna(ped_data.dec_x[blocking_ped]):                                     # Blocking ped that will not move this timestep -> present ped will not move either
                ped_data.dec_x[k] = ped_data.x[k][-1]
                ped_data.dec_y[k] = ped_data.y[k][-1]
                ped_data = update_positions(ped_data, distance_grid, peds_to_move[k], t_new, delta_t)
                print('     Ped ' + str(peds_to_move[k]) + ' blocked in queue')
                    
            else:                                                                           # Blocking ped that may move this timestep -> waiting
                #chance_to_move = True                                                      # Mutual blok is possible, thus chance to move is not triggered here
                print('     Ped ' + str(peds_to_move[k]) + ' blocker may move')
        
        


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

# Timespace fundamental diagram
plt.figure()
plt.plot(ped_data.t[0], ped_data.x[0], 'r-', label = 'ped 1')
plt.plot(ped_data.t[1], ped_data.x[1], 'g-', label = 'ped 2')
plt.plot(ped_data.t[2], ped_data.x[2], 'b-', label = 'ped 3')
plt.plot(ped_data.t[3], ped_data.x[3], 'k-', label = 'ped 4')
plt.plot(ped_data.t[4], ped_data.x[4], 'm-', label = 'ped  5')
plt.title('Timespace fundamental diagram')
plt.xlabel(r'$t \,\,\mathrm{[s]}$')
plt.ylabel(r'$x \,\,\, \mathrm{[m]}$')
#plt.xlim(0, 10)
#plt.ylim(0, 120)
plt.legend()
plt.show()



















