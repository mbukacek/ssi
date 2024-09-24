import numpy as np
import math as math
import random as rn
from matplotlib import pyplot as plt 


def calculate_energy(positions):
    
    energy = 0
    positions_tmp = positions + [positions[0]+1]
    
    for i in range(len(positions)):
        r = positions_tmp[i+1] - positions_tmp[i]
        energy = energy + 1/r
    
    return energy


def get_new_pozition(positions):
    
    positions_new = positions.copy()
    
    # select an agent to update
    r = rn.randint(0,len(positions_new)-1)  
    step = np.random.normal(loc=0.0, scale=.2, size=None)
    
    x = positions_new[r]
    x = x + step
    x = x % 1                           # modulo 1 to stay in unit circle
    
    positions_new[r] = x 
    
    positions_new.sort(key = float)     # keep them sorted after overtaking
    
    return positions_new


def make_decision(positions, positions_new, beta):
    
    energy = calculate_energy(positions)
    energy_new = calculate_energy(positions_new)
    
    if energy_new < energy:
        accepted = True
    else:
        accepted = math.exp(-1*beta*(energy_new - energy)) > rn.random()
        
    return accepted


#============================================#
#              SCRIPT STARTS HERE            #
#============================================#

iter_cnt = 1000

beta = 3   # to accept only lower energy 10000

positions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

energy_record = [calculate_energy(positions)]
energy_accepted = [True] 

for i in range(iter_cnt):
    positions_new = get_new_pozition(positions)
    
    energy_new = calculate_energy(positions_new)
    energy_record = energy_record + [energy_new]
    
    accepted = make_decision(positions, positions_new, beta)
    
    if accepted:
        positions = positions_new
        energy_accepted = energy_accepted + [True] 
    else:
        energy_accepted = energy_accepted + [False]  


plt.scatter(list(range(len(energy_record))), energy_record, s=100, c=energy_accepted)
plt.grid()
plt.show()

