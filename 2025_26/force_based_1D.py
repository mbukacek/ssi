import pandas as pd
import numpy as np 

t = [0,0,0,0,0]
x = [100,80,40,35,10]
v = [10,0,0,0,0]
d = [0]

#parametry simulace
delta_t = 1
t_max = 20
doba = int(t_max/delta_t)
pocet_vozidel = len(x)

car_data = pd.DataFrame({'car_id': 0,
                         't': [[t[0]]],                         
                         'x': [[x[0]]],                         
                         'v': [[v[0]]],
                             }
                        , index = [0]
                        )
#prvni zavorka jake auto, druha zavorka jaky cas
rep = range(len(x)-1)
for i in rep:
        car_data_n = pd.DataFrame({'car_id': i+1,
                                't': [[t[i+1]]],                         
                                'x': [[x[i+1]]],                         
                                'v': [[v[i+1]]],
                                }, index = [i+1])
        car_data = pd.concat([car_data,car_data_n])

#car_data.v[car_idx][-1] #-1 poslední hodnota

#testovaci model 1 - vynulovani
for j in range(pocet_vozidel):
    for k in range(1,doba):
        car_data.x[j] = car_data.x[j]+[0]
        car_data.v[j] = car_data.v[j]+[0]
        car_data.t[j] = car_data.t[j]+[k*delta_t]

# #testovaci model 2 - kopirovani predchoziho
# for m in range(1,doba):
#     for n in range(pocet_vozidel):
#         car_data.x[n][m] = car_data.x[n][m-1]
#         car_data.v[n][m] = car_data.v[n][m-1]
#         car_data.t[n][m] = car_data.t[n][m-1]+delta_t
