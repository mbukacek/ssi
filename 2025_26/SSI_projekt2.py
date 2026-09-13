import pandas as pd
import random as rn
from matplotlib import pyplot as plt 


#=============================================================#
#                        FUNKCE                               #
#=============================================================#

def init_vehicles(category, y, v, const):
# Funkce vytvoří dataframe vehicle_data ...   
    
    vehicle_data = pd.DataFrame({'vehicle_id': 0,                               # index vozidla
                                 'typ': category[0],                            # typ vozidla osobák x kamion
                                 'len': const['len_car'] if category[0] == 'car' else const['len_trc'], # délka vozidla
                                 'pruh': [[pruh[0]]],                           # jízdní pruh, ve kterém se vozidlo nachází
                                 'y': [[y[0]]],                                 # pozice
                                 'v': [[v[0]]]                                  # rychlost
                                 }, 
                                index = [0])
    #prvni zavorka jake auto, druha zavorka jaky cas
    rep = range(len(y)-1)
    for i in rep:
            vehicle_data_n = pd.DataFrame({'vehicle_id': i+1,
                                           'typ': category[i+1],
                                           'len': const['len_car'] if category[i+1] == 'car' else const['len_trc'],
                                           'pruh': [[pruh[i+1]]],
                                           # 'x': [[x[i+1]]],
                                           'y': [[y[i+1]]],
                                           'v': [[v[i+1]]]  
                                    }, index = [i+1])
            vehicle_data = pd.concat([vehicle_data,vehicle_data_n])
           
    return vehicle_data        

    
def sort_vehicles(vehicle_data, const):
# aplikace periodických okrajových podmínek a seřazení vozidel
    
    for l in range(const['pocet_vozidel']):
        
        vehicle_data.y[l][0] = ((vehicle_data.y[l][0]-1) % const['delka_useku']) + 1
        
    vehicle_data = vehicle_data.sort_values(by=["y"], ascending=False, ignore_index=True)
        
    return vehicle_data

    
def update_position_and_velocity(vehicle_data, vehicle_idx, v_new):    
# provede update pozice a rychlosti jednoho vozidla na základě minulé pozice a aktualizované rychlosti
# uloží napočítané hodnoty do polí v vehicle_data
            
    y_new = vehicle_data.y[vehicle_idx][0] + v_new
     
    vehicle_data.at[vehicle_idx,'y'] = [y_new] + vehicle_data.y[vehicle_idx]
        
    vehicle_data.at[vehicle_idx,'v'] = [v_new] + vehicle_data.v[vehicle_idx]

    return vehicle_data


def gap_to_preceding_vehicle(vehicle_data, vehicle_idx, const):
# napočítá prostorové světlosti (mezeru) od předchozích vozidel v obou pruzích
    
    y_current = vehicle_data.y[vehicle_idx][0]  #aktuální poloha
    
    preceding_gap = const['delka_useku']        #prostorová světlost od předchozího vozidla v mém pruhu
    
    preceding_gap_other = const['delka_useku']  #prostorová světlost od předchozího vozidla v druhém pruhu
    
    empty_lane = True                           # True, pokud funkce dosud nenašla vozidlo v mém pruhu
    
    empty_other_lane = True                     # False, pokud funkce našla vozidlo ve druhém pruhu
    
    for i in range(1, const['pocet_vozidel']):
        
        candidate_idx = (vehicle_idx - i) % const['pocet_vozidel']
            
        if empty_lane:
                    
            lane = vehicle_data.pruh[vehicle_idx][0]
                    
            if vehicle_data.pruh[candidate_idx][0] == lane:
                        
                if candidate_idx < vehicle_idx:
                            
                    preceding_gap = vehicle_data.y[candidate_idx][1] - y_current - vehicle_data.len[vehicle_idx]
                            
                else:
                            
                    preceding_gap = vehicle_data.y[candidate_idx][0] + const['delka_useku'] - y_current - vehicle_data.len[vehicle_idx] 
                    
                empty_lane = False
                    
        if empty_other_lane:
                    
            lane = 1 - vehicle_data.pruh[vehicle_idx][0]
                    
            if vehicle_data.pruh[candidate_idx][0] == lane:
                        
                if candidate_idx < vehicle_idx:
                            
                    preceding_gap_other = vehicle_data.y[candidate_idx][1] - y_current - vehicle_data.len[vehicle_idx]
                            
                else:
                            
                    preceding_gap_other = vehicle_data.y[candidate_idx][0] + const['delka_useku'] - y_current - vehicle_data.len[vehicle_idx]
                                                   
                empty_other_lane = False
            
        if not empty_lane and not empty_other_lane:
            
            break
                
    return preceding_gap, preceding_gap_other


def gap_to_following_vehicle(vehicle_data, vehicle_idx, const):
# napočítá vzdálenost od nejbližšího pronásledovatele ve druhém pruhu

    lane = vehicle_data.pruh[vehicle_idx][0]        #muj jizdni pruh
    
    other_lane = 1 - lane                           #druhy jizdni pruh
    
    y_current = vehicle_data.y[vehicle_idx][0]
    
    successive_gap = const['delka_useku']           #prostorová světlost od následujícího vozidla ve druhém pruhu
    
    for j in range(1, const['pocet_vozidel']):
        
        candidate_idx = (vehicle_idx + j) % const['pocet_vozidel']
        
        if vehicle_data.pruh[candidate_idx][0] == other_lane:
                
            if candidate_idx > vehicle_idx:
                    
                successive_gap = y_current - vehicle_data.y[candidate_idx][0] - vehicle_data.v[candidate_idx][0] - vehicle_data.len[candidate_idx]
                    
            else:
                    
                successive_gap = y_current + const['delka_useku'] - vehicle_data.y[candidate_idx][0] - vehicle_data.len[candidate_idx]
                
            break
                
    return successive_gap


def calculate_velocity(vehicle_data, vehicle_idx, const):
# funkce zaktualizuje rychlost vozidla dle dvouproudého heterogenního NaSchova modelu    

    v_new = vehicle_data.v[vehicle_idx][0]          #aktuální rychlost
        
    if vehicle_data.typ[vehicle_idx] == 'car':
        
        # velocity limits & acceleration
        if v_new < const['v_max_car']:
            v_new = v_new + 1
    
    else:
        
        if v_new < const['v_max_trc']:
            v_new = v_new + 1
            
    my_space, other_space_in_front = gap_to_preceding_vehicle(vehicle_data, vehicle_idx, const) #napočítá světlosti od předchozích vozidel v obou pruzích
            
    if (my_space < v_new) or (vehicle_data.pruh[vehicle_idx][0]==0): #pokud se osobní auto nachází v předjížděcím pruhu
                                                                     #nebo je nuceno zpomalit
        if vehicle_data.typ[vehicle_idx]=='car':
            
            other_space_behind = gap_to_following_vehicle(vehicle_data, vehicle_idx, const) #mezera od následujícího vozidla ve druhém pruhu
            
            if ((other_space_in_front > my_space) or ((vehicle_data.pruh[vehicle_idx][0]==0) & (other_space_in_front >= v_new))) & (other_space_behind >=0):#podmínky změny pruhu
            
                if other_space_in_front < v_new:
                
                    v_new = other_space_in_front
            
                vehicle_data.at[vehicle_idx,'pruh'] = [1-vehicle_data.pruh[vehicle_idx][0]] + vehicle_data.pruh[vehicle_idx] #update jizdniho pruhu
            
            else:
            
                if my_space < v_new:
            
                    v_new = my_space
            
                vehicle_data.at[vehicle_idx,'pruh'] = [vehicle_data.pruh[vehicle_idx][0]] + vehicle_data.pruh[vehicle_idx]
        
        else:
            
            if my_space < v_new:
                
                v_new = my_space
                
            vehicle_data.at[vehicle_idx,'pruh'] = [vehicle_data.pruh[vehicle_idx][0]] + vehicle_data.pruh[vehicle_idx]
    
    else:
        
        vehicle_data.at[vehicle_idx,'pruh'] = [vehicle_data.pruh[vehicle_idx][0]] + vehicle_data.pruh[vehicle_idx]
    
    # random deceleration 
    r = rn.random()
    if (v_new > 0) & (r < const['q_rand']):
        v_new = v_new - 1
            
    return v_new


#==================================#
#            PARAMETRY             #
#==================================#

const = {'cell_size':3.75,        # velikost buňky
         'v_max_car': 7,          # v_max pro osobáky 7 buněk/krok
         'v_max_trc': 6,          # v_max pro kamiony 6 buněk/krok
         'len_car': 2,            # osobní auta - 2 buňky
         'len_trc': 5,            # náklaďáky - 5 buněk
         'q_rand': 1/3,           # pravděpodobnost zpomalení
         'delta_t': 1,            # krok v sekundách, krok ~ 1s (-> rychlost 1 buňka / krok ~ 13,5 km/h)
         'pocet_vozidel': 200,    # počet vozidel
         'doba': 3600,            # doba simulace v krocích (celkem tedy 1h)
         'delka_useku': 1000,     # délka úseku v buňkách, tzn. 3750 m
         '%_trc': 18               # procentuální zastoupení kamionů ve vzorku
         }

#==================================================#
#                     MODEL                        #
#==================================================#

car_speed = []                          #rychlosti automobilů

truck_speed = []                        #rychlosti kamionů

skalovane_svetlosti = []                #ukládá naměřené škálované světlosti

skalovane_svetlosti_car = []            #škálované světlosti automobilů

skalovane_svetlosti_truck = []          #škálované světlosti kamionů

svetlosti = []

for rep in range(10):            # počet simulací

    y = [const['delka_useku'] - int(i*const['delka_useku']/const['pocet_vozidel']) for i in range(0, const['pocet_vozidel'])]    # index buňky, počáteční rozdělení - rovnoměrné
        
    v = [const['v_max_trc']-1]*const['pocet_vozidel']                      # pocet buněk za krok
    
    pruh = [1]*const['pocet_vozidel']                                      # všechna vozidla začínají v pravém pruhu
    
    category = ['car'] * int((1-(const['%_trc']/100)) * const['pocet_vozidel']) + ['trc'] * (const['pocet_vozidel']-(int((1-(const['%_trc']/100)) * const['pocet_vozidel'])))
                
    rn.shuffle(category)                    #náhodně rozdělená příslušnost vozidel k daným typům

    vehicle_data = init_vehicles(category, y, v, const)     # Funkce vytvoří dataframe vehicle_data ...  

    for k in range(const['doba']):    
    
        for vehicle_idx in range(const['pocet_vozidel']):

            v_new = calculate_velocity(vehicle_data, vehicle_idx, const)    # funkce zaktualizuje rychlost vozidla dle dvouproudého heterogenního NaSchova modelu 
     
            vehicle_data = update_position_and_velocity(vehicle_data, vehicle_idx, v_new) # provede update pozice a rychlosti jednoho vozidla na základě minulé pozice a aktualizované rychlosti

        vehicle_data = sort_vehicles(vehicle_data, const)    # aplikace periodických okrajových podmínek a seřazení vozidel
        
#==================================================#
#                  POSTPROCESSING                  #
#==================================================#

    for veh_idx in range(len(vehicle_data)):

        vehicle_data.at[veh_idx,'finalni pruh'] = vehicle_data.pruh[veh_idx][0]

    data = vehicle_data[vehicle_data['finalni pruh']==1]            #budeme analyzovat světlosti pouze z pravého pruhu
    data = data.reset_index(drop=True)

    inhomo = len(data[data['typ']=='trc'])/len(data)*100            # procentuální zastoupení kamionů v rámci pravého pruhu
    
    density = len(data)/(const['delka_useku']*const['cell_size']/1000)      #výpočet hustoty v pravém pruhu
    
    r = [data.y[0][0]]                                              #pozice zadních nárazníků vozidel
    
    f = [data.y[0][0] + data.len[0] - 1]                            #pozice předních nárazníků vozidel
    
    data.at[0,'distance'] = data.y[len(data)-1][0] + const['delka_useku'] - f[0] - 1    #ukládá světlosti mezi vozidly
    
    data.at[0,'rychlost'] = data.v[0][0]                            #ukládá finální rychlosti vozidel (použito v rámci kalibrace)

    for i in range(1,len(data)):
    
        r = r + [data.y[i][0]]
        
        f = f + [r[-1] + data.len[i] - 1]

        data.at[i,'distance'] = r[-2] - f[-1] - 1
        
        data.at[i,'rychlost'] = data.v[i][0]
        
    svetlosti = svetlosti + (data['distance']+(3.5/3.75)).tolist()

    skalovane_svetlosti = skalovane_svetlosti + ((data['distance']+(3.5/3.75)) / data['distance'].mean()).tolist() #skalovani
                                                                                #3.5/3.75 udává neobsazenou část celku buňky
    #skalovane_svetlosti_car = skalovane_svetlosti_car + ((data[data['typ']=='car']['distance']+(3.5/3.75))/ data['distance'].mean()).tolist()
    
    #skalovane_svetlosti_truck = skalovane_svetlosti_truck + ((data[data['typ']=='trc']['distance']+(3.5/3.75))/ data['distance'].mean()).tolist()
    
    #car_speed = car_speed + [data[data['typ']=='car']['rychlost'].mean()]

    #truck_speed = truck_speed + [data[data['typ']=='trc']['rychlost'].mean()]                                                                    

#plt.figure()
#plt.boxplot((homogenni_svetlosti, skoro_homogenni_svetlosti, nehomogenni_svetlosti))
#plt.xticks(ticks = range(1, 3 + 1), labels = ['0%', '9%', '36%'])
#plt.xlabel('Procentuální zastoupení nákladních vozidel ve vzorku')
#plt.ylabel('Prostorová světlost sousedních vozidel [buňka]')
#plt.show()

