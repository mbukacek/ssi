import numpy as np
from matplotlib import pyplot as plt

T_max = 100
J_out = 2
tok_in = []
tok_out = []

def obsluha(N_in,N_out,T,entry,leave,J_in):

    if entry < leave:
        t = T[-1] + entry
        N_in = N_in + [N_in[-1]+1]
        N_out = N_out + [N_out[-1]]
        T = T + [t]
        leave = leave - entry
        entry = np.random.exponential(1/J_in)

    elif entry >= leave and N_in[-1] != 0:
        t = T[-1] + leave
        N_in = N_in + [N_in[-1]-1]
        N_out = N_out + [N_out[-1]+1]
        T = T + [t]
        entry = entry - leave
        leave = np.random.exponential(1/J_out)
        
    else:
        t = T[-1] + leave
        N_in = N_in + [N_in[-1]+1]
        N_out = N_out + [N_out[-1]]
        T = T + [t]
        entry = np.random.exponential(1/J_in)
        leave = np.random.exponential(1/J_out)
    
    return N_in,N_out,T,entry,leave
        
for j in range(0,51,1):
    
    for k in range(0,6,1):
        
        T = [0]
        
        leave = np.random.exponential(1/J_out)

        entry = np.random.exponential(1/(0.1+(j*0.1)))
        
        N_in = [0]
        N_out = [0]

        while T[-1] < T_max:
        
            N_in,N_out,T,entry,leave = obsluha(N_in,N_out,T,entry,leave,0.1+(j*0.1))
    
        tok_in = tok_in + [0.1+(j*0.1)]
        tok_out = tok_out + [N_out[-1]/T_max]
        plt.plot(tok_in, tok_out, '.')
    
#plt.plot(T,N_in)
    
    