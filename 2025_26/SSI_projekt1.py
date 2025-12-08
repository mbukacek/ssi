import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'

#=================================#
#             FUNKCE              #
#=================================#   


def add_ped(ped_data,t):  # přidání chodce (lyžaře) do systému

    idx = len(ped_data)
    
    # generování náhodných pozic chodců
    
    x_initial = const['R'] + np.random.rand() + max(max(i[-1] for i in ped_data.x),attractors['A1x'])
    y_initial = const['R'] + (np.random.rand() * (wall['w2y']-(2*const['R'])))
    
    
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

def init_ped_data(t, x, y, vx, vy, const):   # vytvoření dataframu
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
                                    }, index = [i+1])
            ped_data = pd.concat([ped_data,ped_data_n])     
    return ped_data        

def update_position_and_speed(ped_data, ped_idx, Fx, Fy, t_new, delta_t, T_turniket): # update polohy a rychlosti chodce
    
    dist = np.sqrt((attractors['Ax']-ped_data.x[ped_idx][-1])**2+(attractors['Ay']-ped_data.y[ped_idx][-1])**2)
    
    if dist < const['reach_dist']: # ověření, zda je chodec blízko lanovky
        #vx_new = 0
        #vy_new = 0
        #x_new = ped_data.x[ped_idx][-1]
        #y_new = ped_data.y[ped_idx][-1]
        #ped_data.waiting[ped_idx] = True
        
        if ped_data.at[ped_idx,'waiting'] == False: # změna statusu lyžaře na čekajícího
            
            ped_data.at[ped_idx,'waiting'] = True
        
    else:
        ped_data.at[ped_idx,'waiting'] = False
          
    x_new = ped_data.x[ped_idx][-1] + delta_t*ped_data.vx[ped_idx][-1]
    y_new = ped_data.y[ped_idx][-1] + delta_t*ped_data.vy[ped_idx][-1]
    
    vx_new = ped_data.vx[ped_idx][-1] + delta_t*Fx
    vy_new = ped_data.vy[ped_idx][-1] + delta_t*Fy
    
    #----------------------hard-core repulsion wall---------------------------#
    
    if x_new < wall['w1x']:
        x_new = wall['w1x']
        vx_new = 0 
    if y_new < wall['w1y']:
        y_new = wall['w1y']
        vy_new = 0
    if y_new > wall['w2y']:
        y_new = wall['w2y']
        vy_new = 0
    if (x_new < attractors['A1x'] and np.isnan(ped_data.at[ped_idx, 't_in'])):
        x_new = attractors['A1x']
        vx_new = 0
    
    
    
    #-------------------------------------------------------------------------#
    
    A1 = np.sqrt((attractors['A1x']-x_new)**2 + (attractors['A1y']-y_new)**2)
    A2 = np.sqrt((attractors['A2x']-x_new)**2 + (attractors['A2y']-y_new)**2)

    if np.isnan(ped_data.at[ped_idx,'t_in']):
                    
        if (A1 < const['entry_dist'] or A2 < const['entry_dist']): # ověření, zda je chodec v dosahové vzdálenosti turniketů
            ped_data.at[ped_idx,'t_in'] = t_new
            T_turniket = T_turniket + [ped_data.at[ped_idx,'t_in']-ped_data.t[ped_idx][0]]
            vx_new = 0
            vy_new = 0
    
    ped_data.at[ped_idx,'x'] = ped_data.x[ped_idx]+[x_new]
    ped_data.at[ped_idx,'y'] = ped_data.y[ped_idx]+[y_new]
    ped_data.at[ped_idx,'vx'] = ped_data.vx[ped_idx]+[vx_new]
    ped_data.at[ped_idx,'vy'] = ped_data.vy[ped_idx]+[vy_new]
    ped_data.at[ped_idx,'t'] = ped_data.t[ped_idx]+[t_new]
    
    return ped_data, T_turniket

def motivation_force(ped_data, ped_idx, const): # výpočet přitažlivé síly lyžaře k lanovce
    
    if np.isnan(ped_data.at[ped_idx, 't_in']): # je-li lyžař před turnikety
    #if np.isnan(ped_data.t_in[ped_idx]):
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
        
    else: # je-li lyžař v prostoru mezi turnikety a lanovkou
        
        A = np.sqrt((attractors['Ax']-ped_data.x[ped_idx][-1])**2+(attractors['Ay']-ped_data.y[ped_idx][-1])**2)
        
        sx = (attractors['Ax']-ped_data.x[ped_idx][-1])/A
        sy = (attractors['Ay']-ped_data.y[ped_idx][-1])/A
        
        F_Mx = (sx*const['v_opt']-ped_data.vx[ped_idx][-1])/const['tau']
        F_My = (sy*const['v_opt']-ped_data.vy[ped_idx][-1])/const['tau']
    
    return F_Mx,F_My
      
def agent_interaction_force(ped_data, ped_idx, const): # výpočet vzájemné interakce dvou agentů

    F_Ix = [0]
    F_Iy = [0]
        
    for l in ped_data[ped_data.active==True].ped_id:
            
        if ped_idx != l:
                
            d = np.sqrt((ped_data.x[ped_idx][-1]-ped_data.x[l][-1])**2+(ped_data.y[ped_idx][-1]-ped_data.y[l][-1])**2)
            
            v = np.sqrt((ped_data.vx[ped_idx][-1])**2+(ped_data.vy[ped_idx][-1])**2)
            
            #if d < 2:
            
            cosfi = -(ped_data.vx[ped_idx][-1]*(ped_data.x[ped_idx][-1]-ped_data.x[l][-1])+(ped_data.vy[ped_idx][-1]*(ped_data.y[ped_idx][-1]-ped_data.y[l][-1])))/(d*(max(0.00001,v)))
            lamb = (const['lambda']+(1-const['lambda'])*((1+cosfi)/2))
            
            F_Ix = F_Ix + [lamb*((const['U_0']/const['xi'])*np.exp(((2*const['R'])-d)/const['xi'])*((ped_data.x[ped_idx][-1]-ped_data.x[l][-1])/d))]
            F_Iy = F_Iy + [lamb*((const['U_0']/const['xi'])*np.exp(((2*const['R'])-d)/const['xi'])*((ped_data.y[ped_idx][-1]-ped_data.y[l][-1])/d))]
            
    F_Ix = np.sum(F_Ix)
    F_Iy = np.sum(F_Iy)

    return F_Ix,F_Iy

def wall_repulsion(ped_data, ped_idx, wall, const): # odpudivá síla od zábran koridoru
    
    if np.isnan(ped_data.at[ped_idx,'t_in']):
        F_Ex = wall['U_0']/wall['xi'] * np.exp((const['R']-(ped_data.x[ped_idx][-1]-wall['w2x']))/wall['xi'])
        
    else:
        F_Ex = wall['U_0']/wall['xi'] * np.exp((const['R']-(ped_data.x[ped_idx][-1]-wall['w1x']))/wall['xi'])
        
    #if ped_data.y[ped_idx][-1] > wall['w2y']/2:
    F_Ey1 = -wall['U_0']/wall['xi'] * np.exp((const['R']-(wall['w2y']-ped_data.y[ped_idx][-1]))/wall['xi'])
        
    #else:
    F_Ey = F_Ey1 + (wall['U_0']/wall['xi'] * np.exp((const['R']-ped_data.y[ped_idx][-1])/wall['xi']))

    return F_Ex, F_Ey

def lanovka(ped_data, t_new, T_lanovka, T_celkovy): # jak jezdí lanovka
    
    if len(ped_data[ped_data.waiting == True]) <= const['kapacita']: 
        for h in ped_data[ped_data.waiting == True].ped_id:
            ped_data.at[h,'t_out'] = t_new
            ped_data.at[h,'waiting'] = False
            ped_data.at[h,'active'] = False
            
            T_lanovka = T_lanovka + [ped_data.at[h, 't_out']-ped_data.at[h, 't_in']] 
            T_celkovy = T_celkovy + [ped_data.at[h, 't_out']-ped_data.t[h][0]]
            
    else:
        
        while len(ped_data[ped_data.t_out == t_new]) < const['kapacita']: #nabírá lyžaře v blízkosti lanovky, dokud není naplněna kapacita
            h = min(ped_data[ped_data.waiting == True].ped_id)
            
            ped_data.at[h,'t_out'] = t_new
            ped_data.at[h,'waiting'] = False
            ped_data.at[h,'active'] = False
            
            T_lanovka = T_lanovka + [ped_data.at[h, 't_out']-ped_data.at[h, 't_in']]
            T_celkovy = T_celkovy + [ped_data.at[h, 't_out']-ped_data.t[h][0]]
    
    return ped_data, T_lanovka, T_celkovy
    

#=============================#
#          PARAMETRY          #
#=============================#

#geometrie systému (koridor)
wall = {'w1x':0,
        'w1y':0,
        'w2x':10,
        'w2y':1.5,
        'xi':0.1,
        'U_0':0.25,
        }

#atraktory (turnikety a lanovka)
attractors = {'A1x':10,
              'A1y':0.5,
              'A2x':10,
              'A2y':1,
              'Ax':0,
              'Ay':0.75
              }

#parametry prvního lyžaře
t = [0]
x = [attractors['A1x']+0.01]
y = [attractors['A1y']+0.01]
vx = [0]
vy = [0]

T_lanovka = []
T_turniket = []
T_celkovy = []

#=====================================================#
#                      SCRIPT                         #
#=====================================================#    

for l in range(1): #kolikrát provádíme simulaci
    const = {'dt':0.05, #časový krok simulace
             't_max':30, #délka simulace [s]
             'I_in':0.5, #I_in...průměrný počet nově příchozích agentů za sekundu
             'v_opt':3, #optimální rychlost lyžařů
             'tau':0.25, #škálovací parametr motivační síly
             'entry_dist':0.1, #dosah čtečky u turniketu
             'reach_dist':0.75, #dosah lanovky
             'U_0':0.5, #škálovací parametr interakční síly mezi agenty
             'xi':0.2, #dosah interakční síly
             'R':0.3, #poloměr agenta
             'lambda':0.1, #anizotropní faktor
             'N_ped_init':4, #počáteční počet čekajících
             'kapacita':6, #kapacita lanovky
             'interval':12} #časový interval příjezdu lanovky

    ped_data = init_ped_data(t,x,y,vx,vy,const)
    
    for i in range(const['N_ped_init']-1):
        
        ped_data = add_ped(ped_data,t[0])
    
    for k in range(1,int(const['t_max']/const['dt'])):
    
        t_new = k*const['dt']
        N_new = np.random.poisson(const['I_in']*const['dt']) #vygenerujeme nově příchozí chodce
    
        for s in ped_data[ped_data.active == True].ped_id: # u aktivních lyžařů vypočítáme působící síly a aktualizujeme jeho polohu a rychlost
            F_Mx,F_My = motivation_force(ped_data, s, const)
            F_Ix,F_Iy = agent_interaction_force(ped_data, s, const)
            F_Ex,F_Ey = wall_repulsion(ped_data, s, wall, const)
            Fx = F_Mx + F_Ix + F_Ex
            Fy = F_My + F_Iy + F_Ey
            ped_data, T_turniket = update_position_and_speed(ped_data, s, Fx, Fy, t_new, const['dt'], T_turniket)
    
        for j in range(N_new):
            ped_data = add_ped(ped_data,t_new)
    
        if round(t_new,2) % const['interval'] == 0:
            ped_data, T_lanovka, T_celkovy = lanovka(ped_data, t_new, T_lanovka, T_celkovy)
            
        # if round(t_new,2) % 0.25 == 0:
        #     plt.figure()
        #     active = ped_data[ped_data.active==True].reset_index()
        #     for j in range(len(active)):
        #         plt.scatter(active.x[j][-1], active.y[j][-1])
        #         plt.xlim(0,12)
        #         plt.ylim(0,1.5)
            
#=============================================================================#
#                               Vizualizace                                   #
#=============================================================================#
plt.figure()
plt.plot(attractors['A1x'], attractors['A1y'], 'r*', label = 'turniket 1')
plt.plot(attractors['A2x'], attractors['A2y'], 'r*', label = 'turniket 2')
plt.plot(attractors['Ax'], attractors['Ay'], 'r*', label = 'lanovka')
for j in range(len(ped_data[ped_data.active==True])):
    active = ped_data[ped_data.active==True].reset_index()
    plt.scatter(active.x[j][-1], active.y[j][-1], s=3000)
    plt.ylim(0.00,1.50)
    plt.xlim(0.00,1.50)

plt.figure()
plt.scatter(range(len(T_turniket)),T_turniket)
plt.show()

plt.figure()
plt.scatter(range(len(T_lanovka)),T_lanovka)
plt.show()

plt.figure()
plt.scatter(range(len(T_celkovy)),T_celkovy)
plt.show()