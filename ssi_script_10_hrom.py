import numpy as np
from matplotlib import pyplot as plt 

def one_node_simulation(j_in, j_out, t_max):

    N = [0]
    T = [0]
    t_act = 0
    N_out = 0
    N_in = 0

    next_in = t_act + np.random.exponential(1/j_in)
    next_out = np.inf
    t_next = min(next_in, next_out)

    while t_next < t_max:
        t_act = t_next   
        T = T + [t_act]
    
        if t_act == next_in:
            N_in = N_in + 1
            N = N + [N[-1] + 1]
            next_in = t_act + np.random.exponential(1/j_in)
            if next_out == np.inf:
                next_out = t_act + np.random.exponential(1/j_out)
                
        elif t_act == next_out:      
            N_out = N_out + 1
            N = N + [N[-1] - 1]
            if N[-1] <= 0:
                next_out = np.inf
            elif N[-1] <= 10:    
                next_out = t_act + np.random.exponential(1/j_out)
            elif N[-1] <= 20: 
                next_out = t_act + np.random.exponential(1/(j_out*(1+0.2*N[-1]/20)))
            else:
                next_out = t_act + np.random.exponential(1/(j_out*(1+0.2)))
        
        else: print('Unknown event')
    
        t_next = min(next_in, next_out)

    return N_out, N_in, T, N

#============================================#
#              SCRIPT STARTS HERE            #
#============================================#

J_out = [0]
J_in = [0]

j_out = 2
t_max = 500

for j in np.arange(0.1, 5.1, 0.1):
    print(j)
    
    for i in range(10):
    
        N_out_loc, N_in_loc, T_loc, N_loc = one_node_simulation(j, j_out, t_max)
    
        J_out = J_out + [N_out_loc/t_max]
        J_in = J_in + [j]

plt.plot(J_in, J_out, '.')


#  To analzye one round in detail
#N_out_loc, N_in_loc, T_loc, N_loc = one_node_simulation(2.2, 2, 1000)
#plt.plot(T_loc,N_loc)


