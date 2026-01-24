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
    
    x_initial = 2*const['R'] + np.random.rand() + max(max(i[-1] for i in ped_data.x),attractors['A1x'])
    y_initial = const['R'] + (np.random.rand() * (wall['w2y']-(2*const['R'])))
    vx_initial = -0.01
    
    ped_data_m = pd.DataFrame({'ped_id': idx,
                               't': [[t]],                         
                               'x': [[x_initial]],
                               'y': [[y_initial]], 
                               'vx': [[vx_initial]],
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

def update_position_and_speed(ped_data, ped_idx, Fx, Fy, t_new, delta_t, T_turniket, wall, barrier): # update polohy a rychlosti chodce
    
    dist = np.sqrt((attractors['Ax']-ped_data.x[ped_idx][-1])**2+(attractors['Ay']-ped_data.y[ped_idx][-1])**2)
    
    if dist < const['reach_dist']: # ověření, zda je chodec blízko lanovky
        
        if ped_data.at[ped_idx,'waiting'] == False: # změna statusu lyžaře na čekajícího
            
            ped_data.at[ped_idx,'waiting'] = True
        
    else:
        ped_data.at[ped_idx,'waiting'] = False
          
    x_new = ped_data.x[ped_idx][-1] + (delta_t*ped_data.vx[ped_idx][-1])
    y_new = ped_data.y[ped_idx][-1] + (delta_t*ped_data.vy[ped_idx][-1])
    
    vx_new = ped_data.vx[ped_idx][-1] + (delta_t*Fx)
    vy_new = ped_data.vy[ped_idx][-1] + (delta_t*Fy)
    
    #----------------------hard-core repulsion wall---------------------------#
    
    if x_new < wall['w1x']:
        x_new = wall['w1x']+0.0001
        vx_new = 0 
    if y_new < wall['w1y']:
        y_new = wall['w1y']+0.0001
        vy_new = 0
    if y_new > wall['w2y']:
        y_new = wall['w2y']-0.0001
        vy_new = 0
    if (x_new < wall['w2x'] and np.isnan(ped_data.at[ped_idx, 't_in'])):
        x_new = wall['w2x']+0.0001
        vx_new = 0
    
    if barrier == True and np.isnan(ped_data.at[ped_idx, 't_in']):
        
        if y_new < wall['w2y']/2 and ped_data.y[ped_idx][-1] > wall['w2y']/2:
            y_new = (wall['w2y']/2)+0.0001
            vy_new = 0
            
        if y_new > wall['w2y']/2 and ped_data.y[ped_idx][-1] < wall['w2y']/2:
            y_new = (wall['w2y']/2)-0.0001
            vy_new = 0
    
    #-------------------------------------------------------------------------#
    
    A1 = np.sqrt((attractors['A1x']-x_new)**2 + (attractors['A1y']-y_new)**2)
    A2 = np.sqrt((attractors['A2x']-x_new)**2 + (attractors['A2y']-y_new)**2)

    if np.isnan(ped_data.at[ped_idx,'t_in']):
                    
        if (A1 < const['entry_dist'] or A2 < const['entry_dist']): # ověření, zda je chodec v dosahové vzdálenosti turniketů
            ped_data.at[ped_idx,'t_in'] = t_new
            T_turniket = T_turniket + [ped_data.at[ped_idx,'t_in']-ped_data.t[ped_idx][0]]
    
    ped_data.at[ped_idx,'x'] = ped_data.x[ped_idx]+[x_new]
    ped_data.at[ped_idx,'y'] = ped_data.y[ped_idx]+[y_new]
    ped_data.at[ped_idx,'vx'] = ped_data.vx[ped_idx]+[vx_new]
    ped_data.at[ped_idx,'vy'] = ped_data.vy[ped_idx]+[vy_new]
    ped_data.at[ped_idx,'t'] = ped_data.t[ped_idx]+[t_new]
    
    return ped_data, T_turniket

def motivation_force(ped_data, ped_idx, const): # výpočet přitažlivé síly lyžaře k lanovce
    
    if np.isnan(ped_data.at[ped_idx, 't_in']): # je-li lyžař před turnikety

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
      
def agent_interaction_force(ped_data, ped_idx, const, d_min): # výpočet vzájemné interakce dvou agentů

    F_Ix = [0]
    F_Iy = [0]
    
    if barrier == True:
        
        if ped_data.y[ped_idx][-1] < wall['w2y']/2:
        
            for l in ped_data.ped_id:
                
                if (ped_data.t_in[l] != np.nan and ped_data.active[l]==True) or (ped_data.t_in[l] == np.nan and ped_data.y[l] < wall['w2y']/2):
                
                    if ped_idx != l:
                    
                        d = np.sqrt((ped_data.x[ped_idx][-1]-ped_data.x[l][-1])**2+(ped_data.y[ped_idx][-1]-ped_data.y[l][-1])**2)
                    
                        v = np.sqrt((ped_data.vx[ped_idx][-1])**2+(ped_data.vy[ped_idx][-1])**2)
                
                        cosfi = -(ped_data.vx[ped_idx][-1]*(ped_data.x[ped_idx][-1]-ped_data.x[l][-1])+(ped_data.vy[ped_idx][-1]*(ped_data.y[ped_idx][-1]-ped_data.y[l][-1])))/(d*(max(0.00001,v)))
                        lamb = (const['lambda']+((1-const['lambda'])*((1+cosfi)/2)))
                
                        F_Ix = F_Ix + [lamb*((const['U_0']/const['xi'])*np.exp(((2*const['R'])-d)/const['xi'])*((ped_data.x[ped_idx][-1]-ped_data.x[l][-1])/d))]
                        F_Iy = F_Iy + [lamb*((const['U_0']/const['xi'])*np.exp(((2*const['R'])-d)/const['xi'])*((ped_data.y[ped_idx][-1]-ped_data.y[l][-1])/d))]
                
                    F_Ix = np.sum(F_Ix)
                    F_Iy = np.sum(F_Iy)
                
        if ped_data.y[ped_idx][-1] > wall['w2y']/2:
            
            for l in ped_data.ped_id:
                
                if (ped_data.t_in[l] != np.nan and ped_data.active[l]==True) or (ped_data.t_in[l] == np.nan and ped_data.y[l] < wall['w2y']/2):
                
                    if ped_idx != l:
                    
                        d = np.sqrt((ped_data.x[ped_idx][-1]-ped_data.x[l][-1])**2+(ped_data.y[ped_idx][-1]-ped_data.y[l][-1])**2)
                
                        v = np.sqrt((ped_data.vx[ped_idx][-1])**2+(ped_data.vy[ped_idx][-1])**2)
                
                        cosfi = -(ped_data.vx[ped_idx][-1]*(ped_data.x[ped_idx][-1]-ped_data.x[l][-1])+(ped_data.vy[ped_idx][-1]*(ped_data.y[ped_idx][-1]-ped_data.y[l][-1])))/(d*(max(0.00001,v)))
                        lamb = (const['lambda']+((1-const['lambda'])*((1+cosfi)/2)))
                
                        F_Ix = F_Ix + [lamb*((const['U_0']/const['xi'])*np.exp(((2*const['R'])-d)/const['xi'])*((ped_data.x[ped_idx][-1]-ped_data.x[l][-1])/d))]
                        F_Iy = F_Iy + [lamb*((const['U_0']/const['xi'])*np.exp(((2*const['R'])-d)/const['xi'])*((ped_data.y[ped_idx][-1]-ped_data.y[l][-1])/d))]
                
                    F_Ix = np.sum(F_Ix)
                    F_Iy = np.sum(F_Iy)
        
    else:
        
        for l in ped_data[ped_data.active==True].ped_id:
            
            if ped_idx != l:
                
                d = np.sqrt((ped_data.x[ped_idx][-1]-ped_data.x[l][-1])**2+(ped_data.y[ped_idx][-1]-ped_data.y[l][-1])**2)
            
                if d < d_min:
                    d_min = d
            
                v = np.sqrt((ped_data.vx[ped_idx][-1])**2+(ped_data.vy[ped_idx][-1])**2)
            
                cosfi = -(ped_data.vx[ped_idx][-1]*(ped_data.x[ped_idx][-1]-ped_data.x[l][-1])+(ped_data.vy[ped_idx][-1]*(ped_data.y[ped_idx][-1]-ped_data.y[l][-1])))/(d*(max(0.00001,v)))
                lamb = (const['lambda']+((1-const['lambda'])*((1+cosfi)/2)))
            
                F_Ix = F_Ix + [lamb*((const['U_0']/const['xi'])*np.exp(((2*const['R'])-d)/const['xi'])*((ped_data.x[ped_idx][-1]-ped_data.x[l][-1])/d))]
                F_Iy = F_Iy + [lamb*((const['U_0']/const['xi'])*np.exp(((2*const['R'])-d)/const['xi'])*((ped_data.y[ped_idx][-1]-ped_data.y[l][-1])/d))]
            
            F_Ix = np.sum(F_Ix)
            F_Iy = np.sum(F_Iy)

    return F_Ix,F_Iy, d_min

def wall_repulsion(ped_data, ped_idx, wall, const, barrier): # odpudivá síla od zábran koridoru
    
    if np.isnan(ped_data.at[ped_idx,'t_in']): #je-li lyžař v prostoru před turnikety
            
        F_Ex = wall['U_0x_turn']/wall['xi_x_turn'] * np.exp((const['R']-(ped_data.x[ped_idx][-1]-wall['w2x']))/wall['xi_x_turn'])

        if barrier == True:
        
            if ped_data.y[ped_idx][-1] > wall['w2y']/2:
            
                F_Ey1 = -wall['U_0y']/wall['xi_y'] * np.exp((const['R']-(wall['w2y']-ped_data.y[ped_idx][-1]))/wall['xi_y'])
            
                F_Ey = F_Ey1 + (wall['U_0y']/wall['xi_y'] * np.exp((const['R']-(ped_data.y[ped_idx][-1]-wall['w2y']/2))/wall['xi_y']))
            
            else:
                
                F_Ey1 = -wall['U_0y']/wall['xi_y'] * np.exp((const['R']-((wall['w2y']/2)-ped_data.y[ped_idx][-1]))/wall['xi_y'])
            
                F_Ey = F_Ey1 + (wall['U_0y']/wall['xi_y'] * np.exp((const['R']-(ped_data.y[ped_idx][-1]))/wall['xi_y']))
        else:
            
            F_Ey1 = -wall['U_0y']/wall['xi_y'] * np.exp((const['R']-(wall['w2y']-ped_data.y[ped_idx][-1]))/wall['xi_y'])
        
            F_Ey = F_Ey1 + (wall['U_0y']/wall['xi_y'] * np.exp((const['R']-ped_data.y[ped_idx][-1])/wall['xi_y']))
            
    else:
        
        F_Ex = wall['U_0x_lan']/wall['xi_x_lan'] * np.exp((const['R']-(ped_data.x[ped_idx][-1]-wall['w1x']))/wall['xi_x_lan'])  
        
        F_Ey1 = -wall['U_0y']/wall['xi_y'] * np.exp((const['R']-(wall['w2y']-ped_data.y[ped_idx][-1]))/wall['xi_y'])
        
        F_Ey = F_Ey1 + (wall['U_0y']/wall['xi_y'] * np.exp((const['R']-ped_data.y[ped_idx][-1])/wall['xi_y']))

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

#atraktory (turnikety a lanovka)
attractors = {'A1x':10,
              'A1y':0.5,
              'A2x':10,
              'A2y':1.5,
              'Ax':0,
              'Ay':1
              }

#parametry prvního lyžaře
t = [0]
x = [attractors['A1x']+2.00]
y = [attractors['A1y']+0.0]
vx = [-0.01]
vy = [0]

barrier = False

T_mean = []
y_max = []
y_min = []
dmin = []
delka_avg = []
interval_set = [6]
N_ped_end = np.full((50, 3), np.nan)
N_ped_lan = np.full((50, 3), np.nan)
N_ped_turn = np.full((50, 3), np.nan)
T_total = []

#=====================================================#
#                      SCRIPT                         #
#=====================================================#    

for m in range(1):
    
    const = {'dt':0.1, #časový krok simulace
             't_max':50, #doba simulace [s]
             'I_in':0.5*(2**m), #I_in...průměrný počet nově příchozích agentů za sekundu
             'v_opt':3, #optimální rychlost lyžařů
             'tau':0.2, #škálovací parametr motivační síly
             'entry_dist':0.30, #dosah čtečky u turniketu
             'reach_dist':2.0, #dosah lanovky
             'U_0':15*(1-(np.exp(-(0.5/2.4))))*2.4, #škálovací parametr interakční síly mezi agenty
             'xi':2.4, #dosah interakční síly
             'R':0.25, #poloměr agenta
             'lambda':0.00, #anizotropní faktor
             'N_ped_init':1, #počáteční počet čekajících
             'kapacita':4, #kapacita lanovky
             'interval':6} #časový interval příjezdu lanovky
    
    delka = 0
    
    y_in_avg = []

    for l in range(1): #kolikrát provádíme simulaci

        T_lanovka = []
        T_turniket = []
        T_celkovy = []
        d_min = np.inf
        y_in1 = []
        y_in2 = []
        
        wall = {'w1x':0,
                'w1y':0,
                'w2x':10,
                'w2y':2.0,
                'xi_x_lan':2.4,
                'xi_x_turn':2.4,
                'xi_y':0.1,
                'U_0x_turn':36,
                'U_0x_lan':36,
                #'U_0y':(0.1*15*(1-(np.exp(-(0.5/2.4))))*2.4)*np.exp(-0.5/2.4)/2/2.4/(np.exp(-(0.25/0.1))-np.exp(-(1.25/0.1)))
                'U_0y':1.7
                }

        print('param set ' + str(m+1) + ', round ' + str(l+1) + ' started')
        ped_data = init_ped_data(t,x,y,vx,vy,const)
    
        for i in range(const['N_ped_init']-1):
        
            ped_data = add_ped(ped_data,t[0])
    
        for k in range(1,int(const['t_max']/const['dt'])+1):
    
            t_new = k*const['dt']
            N_new = np.random.poisson(const['I_in']*const['dt']) #vygenerujeme nově příchozí chodce
    
            for s in ped_data[ped_data.active == True].ped_id: # u aktivních lyžařů vypočítáme působící síly a aktualizujeme jeho polohu a rychlost
                F_Ix,F_Iy, d_min = agent_interaction_force(ped_data, s, const, d_min)
                F_Ex,F_Ey = wall_repulsion(ped_data, s, wall, const, barrier)
                F_Mx,F_My = motivation_force(ped_data, s, const)
                Fx = F_Mx + F_Ix + F_Ex
                Fy = F_My + F_Iy + F_Ey
                ped_data, T_turniket = update_position_and_speed(ped_data, s, Fx, Fy, t_new, const['dt'], T_turniket, wall, barrier)
    
            for j in range(N_new):
                ped_data = add_ped(ped_data,t_new)
    
            if round(t_new,3) % const['interval'] == 0:
                ped_data, T_lanovka, T_celkovy = lanovka(ped_data, t_new, T_lanovka, T_celkovy)
            
        N_ped_end[l][m] = sum(ped_data.active)
        N_ped_turn[l][m] = sum((ped_data.active[i] and np.isnan(ped_data.t_in[i])) for i in ped_data.index)
        N_ped_lan[l][m] = sum((ped_data.active[i] and not np.isnan(ped_data.t_in[i])) for i in ped_data.index)
        T_total = T_total + T_celkovy
        dmin = dmin + [d_min]
        y_max = y_max + [max(max(i) for i in ped_data.y)]
        y_min = y_min + [min(min(i) for i in ped_data.y)]
        for q in ped_data.index:
            if np.isnan(ped_data.t_in[q]) == False:
                y_in = ped_data.y[q][int(round(ped_data.t_in[q]-ped_data.t[q][0],2)*10)]
                if np.abs(y_in-attractors['A1y']) < np.abs(y_in-attractors['A2y']):
                    y_in1 = y_in1 + [y_in-attractors['A1y']]
                else:
                    y_in2 = y_in2 + [attractors['A2y']-y_in]
                
        y_in_avg = y_in_avg + [np.mean(y_in1 +y_in2)]     
            
    inactive = ped_data[ped_data.active==False]
    for u in inactive.index:
        delka = delka + (np.sum(np.sqrt(((np.diff(ped_data.x[u]))**2)+((np.diff(ped_data.y[u]))**2))))/len(inactive)
        
    delka_avg = delka_avg + [delka]
    
    T_mean = T_mean+[np.mean(T_total)]

#=============================================================================#
#                               Vizualizace                                   #
#=============================================================================#

for j in ped_data.index:
    plt.plot(ped_data.x[j], ped_data.y[j])
    plt.plot(attractors['A1x'], attractors['A1y'], 'r*', label = 'turniket 1')
    plt.plot(attractors['A2x'], attractors['A2y'], 'r*', label = 'turniket 2')
    plt.plot(attractors['Ax'], attractors['Ay'], 'r*', label = 'lanovka')
    plt.ylim(0.00,2.0)
    plt.xlim(10.00,25.00)


plt.figure()
plt.boxplot(N_ped_end)
plt.xticks(
    ticks = range(1, 3 + 1),
    labels = [0.5, 1, 2])
plt.xlabel('Průměrný počet přijíždějících lyžařů za sekundu [1]')
plt.ylabel('Počet aktivnich osob na konci [1]')
plt.show()

plt.figure()
plt.boxplot(N_ped_lan)
plt.xticks(
    ticks = range(1, 3 + 1),
    labels = [0.5, 1, 2])
plt.xlabel('Průměrný počet přijíždějících lyžařů za sekundu [1]')
plt.ylabel('Počet čekajících osob před lanovkou [1]')
plt.show()

plt.figure()
plt.boxplot(N_ped_turn)
plt.xticks(
    ticks = range(1, 3 + 1),
    labels = [0.5, 1, 2])
plt.xlabel('Průměrný počet přijíždějících lyžařů za sekundu [1]')
plt.ylabel('Počet čekajících osob před turnikety [1]')
plt.show()

plt.figure()
plt.scatter(np.linspace(1.5,2.5,11),y_in_avg)
plt.xlabel('U_0y [Nm]')
plt.ylabel('rozdíl y-souřadnice lyžaře a turniketu [m]')
plt.show()
