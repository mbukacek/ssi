import numpy as np
from matplotlib import pyplot as plt

"""
This script implements the Nagel-Schreckenberg model for traffic flow.
"""

# ------functions------

def init_road(lane_len, T, p_init):
    """
    Initializes road.
    :param lane_len:  Length of the lane (cells)
    :param T:      Number of time steps
    :param p_init:  Probability of a car being present at a cell
    :return:    Road at time 0
    """

    road = np.zeros((T, lane_len))
    road[0, :] = np.random.rand(lane_len) < p_init

    return road

def update_car(v, vmax, d, p):
    """
    Updates velocity of a car.
    :param v:   Current velocity
    :param vmax:    Maximal velocity
    :param d:   Desired distance to the car in front
    :param p:   Probability of slowing down
    :return:    Updated velocity
    """
    if v < vmax:
        v += 1
    if v > d:
        v = d
    if v > 0 and np.random.rand() < p:
        v -= 1
    return v


def move_car(v, j, lane_len):
    """
    Moves a car.
    :param v:   Current velocity
    :param j:   Current position on the lane
    :param lane_len:    Length of the lane
    :return:    New position on the lane
    """
    if v > 0:
        j_new = (j + v) % lane_len
        j_new = int(j_new)
    else:
        j_new = j
    return j_new

def nagel_schreckenberg(road, vmax, d, p):
    """
    Implements the Nagel-Schreckenberg model for traffic flow.
    :param road:    Road at time t
    :param vmax:    Maximal velocity
    :param d:   Desired distance to the car in front
    :param p:   Probability of slowing down
    :return:    Road at time t+1
    """
    for j in range(len(road[0])):
        if road[t, j] == 1:
            v = update_car(
                v=road[t, j],
                vmax=vmax,
                d=d,
                p=p
            )
            j_new = move_car(v=v, j=j, lane_len=len(road[0]))

            if t < len(road[0]) - 1:
                road[t + 1, j] = 0
                road[t + 1, j_new] = 1
            else:
                road[t, j] = 0
                road[t, j_new] = 1
    return road


def get_avg_velocity(road):
    """
    Computes average velocity of all cars at a given time step.
    :param road:    Road at a given time step
    :return:    Average velocity of all cars
    """
    avg_velocity = np.sum(road) / np.count_nonzero(road[0])
    return avg_velocity

# ------setup-------

constants = {
    'T': 100,
    'lane_len': 100,
    'vmax': 80,
    'p': 0.4,
    'p_init': 0.3,
    'd': 30
}

rd = init_road(
    lane_len=constants['lane_len'],
    T=constants['T'],
    p_init=constants['p_init']
)

# initialize densities and average velocities for visualization
avg_velocities = []
densities = np.linspace(0.0, 1.0, 100)

# ------model-------

for t in range(constants['T']):
    rd = nagel_schreckenberg(
        road=rd,
        vmax=constants['vmax'],
        d=constants['d'],
        p=constants['p']
    )
#--------------------
for density in densities:

    road = init_road(
        lane_len=constants['lane_len'],
        T=constants['T'],
        p_init=density
    )

    for t in range(constants['T']):
        road = nagel_schreckenberg(
            road=road,
            vmax=constants['vmax'],
            d=constants['d'],
            p=constants['p']
        )
    avg_velocities.append(get_avg_velocity(road))


# ------visualization--------

# plot road at each time step
plt.figure(figsize=(8, 6), dpi=100)
plt.imshow(rd, interpolation='nearest', aspect='auto', cmap='viridis_r', origin='upper')
plt.xlabel('Position')
plt.ylabel('Time')
plt.tight_layout()
plt.show()

# plot the average velocity vs density
plt.figure(figsize=(8, 6), dpi=100)
plt.plot(densities, avg_velocities, marker='o', color='#CBD500')
plt.xlabel('Density')
plt.ylabel('Average Velocity')
plt.grid(True)
plt.tight_layout()
plt.show()
