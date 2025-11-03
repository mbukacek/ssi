import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

wall = {'w1x':0,
        'w1y':0,
        'w2y':1.5,
        'xi':0.5,
        'U_0':0.1
        }

attractors = {'A1x':10,
              'A1y':0.5,
              'A2x':10,
              'A2y':1}

const = {'dt':0.1,
         't_max':50,
         'I_in':1, #I_in...průměrný počet nově příchozích agentů za sekundu
         'v_opt':3, #optimální rychlost lyžařů
         'tau':0.5,
         'entry_dist':0.1,
         'reach_dist':1,
         'U_0':1,
         'xi':1,
         'R':0.3,
         'lambda':0.1,
         'N_ped_init':4, #počáteční počet čekajících
         'kapacita':6, #kapacita lanovky
         'interval':12} #časový interval příjezdu lanovky

def add_ped(ped_data,t):
    idx = len(ped_data)
    x_initial = np.random.rand() + max(max(i[-1] for i in ped_data.x),attractors['A1x'])
    y_initial = np.random.rand() * wall['w2y']
    ped_data_m = pd.DataFrame({'ped_id': idx,
                               't': [[t]],                         
                               'x': [[x_initial]],
                               'y': [[y_initial]], 
                               'vx': [[0]],
                               'vy': [[0]],
                               't_in': np.nan,
                               't_out': np.nan,
                               'waiting': False,
                               'active': True
                               }, index = [idx])
    ped_data = pd.concat([ped_data,ped_data_m])
    return ped_data

def init_ped_data(t, x, y, vx, vy, const):   
    ped_data = pd.DataFrame({'ped_id': 0,
                             't': [[t[0]]],                         
                             'x': [[x[0]]],
                             'y': [[y[0]]],
                             'vx': [[vx[0]]],
                             'vy': [[vy[0]]],
                             't_in': np.nan,
                             't_out': np.nan,
                             'waiting': False,
                             'active': True
                             }, index = [0])
    rep = range(len(x)-1)
    for i in rep:
            ped_data_n = pd.DataFrame({'ped_id': i+1,
                                    't': [[t[i+1]]],                         
                                    'x': [[x[i+1]]],
                                    'y': [[y[i+1]]], 
                                    'vx': [[vx[i+1]]],
                                    'vy': [[vy[i+1]]],
                                    't_in': np.nan,
                                    't_out': np.nan,
                                    'waiting': False,
                                    'active': True
                                    }, index = [0])
            ped_data = pd.concat([ped_data,ped_data_n])     
    return ped_data        

def update_position_and_speed(ped_data, ped_idx, Fx, Fy, t_new, delta_t):
    
    if ped_data.x[ped_idx][-1] < const['reach_dist']:
        vx_new = 0
        vy_new = 0
        x_new = ped_data.x[ped_idx][-1]
        y_new = ped_data.y[ped_idx][-1]
        ped_data.waiting[ped_idx] = True
        
    else:
          
        x_new = ped_data.x[ped_idx][-1] + delta_t*ped_data.vx[ped_idx][-1]
        y_new = ped_data.y[ped_idx][-1] + delta_t*ped_data.vy[ped_idx][-1]
    
        vx_new = ped_data.vx[ped_idx][-1] + delta_t*Fx
        vy_new = ped_data.vy[ped_idx][-1] + delta_t*Fy
    
        if x_new < wall['w1x']:
            x_new = wall['w1x']
            vx_new = 0 
        if y_new < wall['w1y']:
            y_new = wall['w1y']
            vy_new = 0
        if y_new > wall['w2y']:
            y_new = wall['w2y']
            vy_new = 0
        if (x_new < attractors['A1x'] and np.isnan(ped_data.t_in[ped_idx])):
            x_new = attractors['A1x']
            vx_new = 0
    
        A1 = np.sqrt((attractors['A1x']-x_new)**2 + (attractors['A1y']-y_new)**2)
        A2 = np.sqrt((attractors['A2x']-x_new)**2 + (attractors['A2y']-y_new)**2)

        if np.isnan(ped_data.t_in[ped_idx]):
                    
            if (A1 < const['entry_dist'] or A2 < const['entry_dist']):
                ped_data.at[ped_idx,'t_in'] = t_new
                vx_new = 0
                vy_new = 0
    
    ped_data.at[ped_idx,'x'] = ped_data.x[ped_idx]+[x_new]
    ped_data.at[ped_idx,'y'] = ped_data.y[ped_idx]+[y_new]
    ped_data.at[ped_idx,'vx'] = ped_data.vx[ped_idx]+[vx_new]
    ped_data.at[ped_idx,'vy'] = ped_data.vy[ped_idx]+[vy_new]
    ped_data.at[ped_idx,'t'] = ped_data.t[ped_idx]+[t_new]
    
    return ped_data

def motivation_force(ped_data, ped_idx, const):
    
    if np.isnan(ped_data.t_in[ped_idx]):
        A1 = np.sqrt((attractors['A1x']-ped_data.x[ped_idx][-1])**2+(attractors['A1y']-ped_data.y[ped_idx][-1])**2)
        A2 = np.sqrt((attractors['A2x']-ped_data.x[ped_idx][-1])**2+(attractors['A2y']-ped_data.y[ped_idx][-1])**2)
    
        if A1 < A2:
        
            sx = (attractors['A1x']-ped_data.x[ped_idx][-1])/A1
            sy = (attractors['A1y']-ped_data.y[ped_idx][-1])/A1
    
        else:
        
            sx = (attractors['A2x']-ped_data.x[ped_idx][-1])/A2
            sy = (attractors['A2y']-ped_data.y[ped_idx][-1])/A2
            
        F_Mx = (sx*const['v_opt']-ped_data.vx[ped_idx][-1])/const['tau']
        F_My = (sy*const['v_opt']-ped_data.vy[ped_idx][-1])/const['tau']
        
    else:
        
        F_Mx = -(const['v_opt']-ped_data.vx[ped_idx][-1])/const['tau']
        F_My = 0
    
    return F_Mx,F_My
      
def agent_interaction_force(ped_data, ped_idx, const):

    F_Ix = [0]
    F_Iy = [0]
        
    for l in ped_data[ped_data.active==True].ped_id:
            
        if ped_idx != l:
                
            d = np.sqrt((ped_data.x[ped_idx][-1]-ped_data.x[l][-1])**2+(ped_data.y[ped_idx][-1]-ped_data.y[l][-1])**2)-(2*const['R'])
            v = np.sqrt((ped_data.vx[ped_idx][-1])**2+(ped_data.vy[ped_idx][-1])**2)
            
            cosfi = -(ped_data.vx[ped_idx][-1]*(ped_data.x[ped_idx][-1]-ped_data.x[l][-1])+(ped_data.vy[ped_idx][-1]*(ped_data.y[ped_idx][-1]-ped_data.y[l][-1])))/((d+(2*const['R']))*(max(0.001,v)))
            lamb = const['lambda']+(1-const['lambda'])*((1+cosfi)/2)
            
            F_Ix = F_Ix + [lamb*((const['U_0']/const['xi'])*np.exp(-(d/const['xi']))*((ped_data.x[ped_idx][-1]-ped_data.x[l][-1])/d))]
            F_Iy = F_Iy + [lamb*((const['U_0']/const['xi'])*np.exp(-(d/const['xi']))*((ped_data.y[ped_idx][-1]-ped_data.y[l][-1])/d))]
            
    F_Ix = np.sum(F_Ix)
    F_Iy = np.sum(F_Iy)

    return F_Ix,F_Iy

def wall_repulsion(ped_data, ped_idx, wall, attractors):
    
    if np.isnan(ped_data.t_in[ped_idx]):
        F_Ex = wall['U_0']/wall['xi'] * np.exp(-(ped_data.x[ped_idx][-1]-attractors['A1x'])/wall['xi'])
        
    else:
        F_Ex = 0
        
    if ped_data.y[ped_idx][-1] > wall['w2y']/2:
        F_Ey = -wall['U_0']/wall['xi'] * np.exp(-(wall['w2y']-ped_data.y[ped_idx][-1])/wall['xi'])
        
    else:
        F_Ey = wall['U_0']/wall['xi'] * np.exp(-ped_data.y[ped_idx][-1]/wall['xi'])

    return F_Ex, F_Ey

def lanovka(ped_data, t_new):
    
    if len(ped_data[ped_data.waiting == True]) <= const['kapacita']:
        for h in ped_data[ped_data.waiting == True].ped_id:
            ped_data.t_out[h] = t_new
            ped_data.waiting[h] = False
            ped_data.active[h] = False
            
    else:
        
        while len(ped_data[ped_data.t_out == t_new])<const['kapacita']:
            h = min(ped_data[ped_data.waiting == True].ped_id)
            ped_data.t_out[h] = t_new
            ped_data.waiting[h] = False
            ped_data.active[h] = False
    
    return ped_data

# t = [0]
# x = [11]
# y = [1]
# vx = [0]
# vy = [0]
# t_in = [np.nan]
# t_out = [np.nan]

t =  np.random.rand(const['N_ped_init'])*0
x =  np.random.rand(const['N_ped_init'])+attractors['A1x']
y =  np.random.rand(const['N_ped_init'])*wall['w2y']
vx = np.random.rand(const['N_ped_init'])*0
vy = np.random.rand(const['N_ped_init'])*0

ped_data = init_ped_data(t,x,y,vx,vy,const)

for k in range(1,int(const['t_max']/const['dt'])):
    
    t_new = k*const['dt']
    N_new = np.random.poisson(const['I_in']*const['dt'])
    
    for s in ped_data[ped_data.active == True].ped_id:
        F_Mx,F_My = motivation_force(ped_data, s, const)
        F_Ix,F_Iy = agent_interaction_force(ped_data, s, const)
        F_Ex,F_Ey = wall_repulsion(ped_data, s, wall, attractors)
        Fx = F_Mx + F_Ix + F_Ex
        Fy = F_My + F_Iy + F_Ey
        ped_data = update_position_and_speed(ped_data, s, Fx, Fy, t_new, const['dt'])
    
    for j in range(N_new):
        ped_data = add_ped(ped_data,t_new)
    
    if round(t_new,2) % const['interval'] == 0:
        ped_data = lanovka(ped_data, t_new)
        
        
# Aerial plot
plt.figure()
plt.plot(attractors['A1x'], attractors['A1y'], 'r*', label = 'attractor 1')
plt.plot(attractors['A2x'], attractors['A2y'], 'r*', label = 'attractor 2')
for j in range(len(ped_data)):
    plt.plot(ped_data.x[j], ped_data.y[j])
    # plt.plot(ped_data.x[0], ped_data.y[0], 'y-o', label = 'ped 1')
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
    
