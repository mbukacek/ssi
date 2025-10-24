import pandas as pd
import numpy as np 
from matplotlib import pyplot as plt 


#======================#
#         FUNKCE       #
#======================#

def init_ped_data(t, x, y, vx, vy, fin, pocet_chodcu):
# Funkce vytvoří dataframe ped_data ...   
    
    ped_data = pd.DataFrame({'ped_id': 0,
                             't': [[t[0]]],                         
                             'x': [[x[0]]],
                             'y': [[y[0]]],
                             'vx': [[vx[0]]],
                             'vy': [[vy[0]]],
                             'finish_time': fin[0]
                                 }
                            , index = [0]
                            )
    #prvni zavorka jaky chodec, druha zavorka jaky cas
    rep = range(len(x)-1)
    for i in rep:
            ped_data_n = pd.DataFrame({'ped_id': i+1,
                                    't': [[t[i+1]]],                         
                                    'x': [[x[i+1]]],
                                    'y': [[y[i+1]]], 
                                    'vx': [[vx[i+1]]],
                                    'vy': [[vy[i+1]]],
                                    'finish_time': fin[i+1]
                                    }, index = [i+1])
            ped_data = pd.concat([ped_data,ped_data_n])
           
    return ped_data        

def add_ped(ped_data, t, x, y):
    m = len(ped_data)
    ped_data_m = pd.DataFrame({'ped_id': m,
                            't': [[t]],                         
                            'x': [[x]],
                            'y': [[y]], 
                            'vx': [[0]],
                            'vy': [[0]],
                            'finish_time': np.nan
                            }, index = [m])
    ped_data = pd.concat([ped_data,ped_data_m])
    return ped_data
    
def update_position_and_speed(ped_data, ped_idx, Fx, Fy, t_new, delta_t):    
# provede update pozice a rychlosti jednoho vozidla na základě minulé pozice, rychlosti a síly
# uloží napočítané hodnoty do polí v car_data
    
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
# funkce vypočítá sílu působící na zadaného chodce dle zadaného modelu    
    
    if model == 'zero_force':
        Fx = 0
        Fy = 0
    
    elif model == 'motivation':
        sx = (const['Ax']-ped_data.x[ped_idx][-1])/np.sqrt((const['Ax']-ped_data.x[ped_idx][-1])**2+(const['Ay']-ped_data.y[ped_idx][-1])**2)
        sy = (const['Ay']-ped_data.y[ped_idx][-1])/np.sqrt((const['Ax']-ped_data.x[ped_idx][-1])**2+(const['Ay']-ped_data.y[ped_idx][-1])**2)
        
        F_Mx = (sx*const['v_opt']-ped_data.vx[ped_idx][-1])/const['tau']
        F_My = (sy*const['v_opt']-ped_data.vy[ped_idx][-1])/const['tau']
    
        b = np.sqrt((ped_data.x[ped_idx][-1]-barriers['B1_x'])**2 + (ped_data.y[ped_idx][-1]-barriers['B1_y'])**2)
            
        F_Bx = (const['U_0']/const['xi'])*np.exp(-(b/const['xi']))
        F_By = (const['U_0']/const['xi'])*np.exp(-(b/const['xi']))
        
        Fx = F_Mx - F_Bx
        Fy = F_My - F_By
    
    elif model == 'motivation_&_interaction':
        
        sx = (const['Ax']-ped_data.x[ped_idx][-1])/np.sqrt((const['Ax']-ped_data.x[ped_idx][-1])**2+(const['Ay']-ped_data.y[ped_idx][-1])**2)
        sy = (const['Ay']-ped_data.y[ped_idx][-1])/np.sqrt((const['Ax']-ped_data.x[ped_idx][-1])**2+(const['Ay']-ped_data.y[ped_idx][-1])**2)
        
        F_Mx = (sx*const['v_opt']-ped_data.vx[ped_idx][-1])/const['tau']
        F_My = (sy*const['v_opt']-ped_data.vy[ped_idx][-1])/const['tau']
        
        F_Ix = []
        F_Iy = []
            
        b = np.sqrt((ped_data.x[ped_idx][-1]-barriers['B1_x'])**2 + (ped_data.y[ped_idx][-1]-barriers['B1_y'])**2)
            
        F_Bx = (const['U_0']/const['xi'])*np.exp(-(b/const['xi']))
        F_By = (const['U_0']/const['xi'])*np.exp(-(b/const['xi']))
        
        for j in range(pocet_chodcu):
            
            if np.isnan(ped_data.finish_time[l]):
            
                if ped_idx == j:
                    F_Ix = F_Ix + [0]
                    F_Iy = F_Iy + [0]
            
                else:
                
                    d = np.sqrt((ped_data.x[ped_idx][-1]-ped_data.x[j][-1])**2+(ped_data.y[ped_idx][-1]-ped_data.y[j][-1])**2)
            
                    #d_x = (ped_data.x[j][-1]-ped_data.x[ped_idx][-1])/d
                    #d_y = (ped_data.y[j][-1]-ped_data.y[ped_idx][-1])/d
            
                    F_Ix = F_Ix + [(const['U_0']/const['xi'])*np.exp(-(d/const['xi']))]
                    F_Iy = F_Iy + [(const['U_0']/const['xi'])*np.exp(-(d/const['xi']))]
                
                Fx = F_Mx - np.sum(F_Ix) - F_Bx
                Fy = F_My - np.sum(F_Iy) - F_By
    else: 
        Fx = np.nan
        Fy = np.nan
    
    return Fx,Fy

#======================#
#      PARAMETRY       #
#======================#

const = {'dt':0.1,
         'tau':0.5,
         'v_opt':1,
         'U_0':160,
         'xi':0.08,
         'eps':0.5,
         'Ax':20,
         'Ay':20}

t = [0,0,0,0,0]
x = [100,80,40,35,10]
y = [15,74,45,70,25]
vx = [-10,0,0,0,0]
vy = [-20,0,0,0,0]
fin = [np.nan,np.nan,np.nan,np.nan,np.nan]

barriers = {'B1_x':40, 'B1_y':40}

#parametry simulace
delta_t = 0.1
t_max = 100
doba = int(t_max/delta_t)
pocet_chodcu = len(x)

# t_safe = 2
# d_safe = 2
# v_opt = 20

#======================#
#         MODEL        #
#======================#

ped_data = init_ped_data(t, x, y, vx, vy, fin, pocet_chodcu)

for k in range(1,doba):
    
    t_new = k*delta_t
    
    Fx = 0
    Fy = 0
    
    for l in range(len(ped_data)):
        
        if np.isnan(ped_data.finish_time[l]):

            Fx,Fy = calculate_force(ped_data, l, 'motivation_&_interaction', const)
        
            ped_data = update_position_and_speed(ped_data, l, Fx, Fy, t_new, delta_t)
        
            d_target = np.sqrt((ped_data.x[l][-1]-const['Ax'])**2+(ped_data.y[l][-1]-const['Ay'])**2)
        
            if d_target < const['eps']:
                
                ped_data.finish_time[l] = t_new
                
                #ped_data = add_ped(ped_data, t_new, np.random.rand(1)*100, np.random.rand(1)*100)

# Velocity of peds
plt.figure()
plt.plot(ped_data.t[0], ped_data.vx[0], 'r-', label = 'ped 1')
plt.plot(ped_data.t[1], ped_data.vx[1], 'g-', label = 'ped 2')
plt.plot(ped_data.t[2], ped_data.vx[2], 'b-', label = 'ped 3')
plt.plot(ped_data.t[3], ped_data.vx[3], 'k-', label = 'ped 4')
plt.plot(ped_data.t[4], ped_data.vx[4], 'm-', label = 'ped 5')
plt.title('Velocity in time')
plt.show()

# Timespace fundamental diagram
plt.figure()
plt.plot(ped_data.t[0], ped_data.x[0], 'r-', label = 'ped 1')
plt.plot(ped_data.t[1], ped_data.x[1], 'g-', label = 'ped 2')
plt.plot(ped_data.t[2], ped_data.x[2], 'b-', label = 'ped 3')
plt.plot(ped_data.t[3], ped_data.x[3], 'k-', label = 'ped 4')
plt.plot(ped_data.t[4],ped_data.x[4], 'm-', label = 'ped 5')
plt.title('Timespace fundamental diagram')
plt.xlabel(r'$t \,\,\mathrm{[s]}$')
plt.ylabel(r'$x \,\,\, \mathrm{[m]}$')
#plt.xlim(0, 10)
#plt.ylim(0, 120)
plt.legend()
plt.show()

# Aerial plot
plt.figure()
plt.plot(const['Ax'], const['Ay'], 'r*', label = 'finish')
plt.plot(ped_data.x[0], ped_data.y[0], 'r-o', label = 'ped 1')
plt.plot(ped_data.x[1], ped_data.y[1], 'g-o', label = 'ped 2')
plt.plot(ped_data.x[2], ped_data.y[2], 'b-o', label = 'ped 3')
plt.plot(ped_data.x[3], ped_data.y[3], 'k-o', label = 'ped 4')
plt.plot(ped_data.x[4], ped_data.y[4], 'm-o', label = 'ped 5')
plt.title('Aerial plot')
plt.xlabel(r'$x \,\,\mathrm{[m]}$')
plt.ylabel(r'$y \,\,\, \mathrm{[m]}$')
#plt.xlim(0, 10)
#plt.ylim(0, 120)
plt.legend()
plt.show()
