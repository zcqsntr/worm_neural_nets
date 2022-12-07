
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib as mpl
mpl.use('tkagg')
import matplotlib.pyplot as plt
import os
import time
from load_data import load_data

theta = np.random.random() * 2 * np.pi
last_sample = 0


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


def concentration_func(x,y,t):

    origin = np.array([4.5, 0.])

    delx = origin[0] - x
    dely = origin[1] - y

    dist = np.sqrt(delx**2 + dely**2)

    std = 4

    return gaussian(dist, mu=0, sig=std)*100*0


def xdot(t, X, p):
    global theta
    global last_sample
    global go_forward
    plate_r = 3

    AWC_v, AWC_f, AWC_s, AIB_v, AIA_v, AIY_v, x, y = X

    AWC_f_a, AWC_f_b, AWC_s_gamma, tm, AWC_v0, AWC_gain, AIB_v0, AIA_v0, AIY_v0, speed, w_1, w_2, w_3, w_4, w_5, w_6, w_7, sampling_time = p

    conc = concentration_func(x, y, t)

    dAWC_f = AWC_f_a*conc - AWC_f_b*AWC_f
    dAWC_s = AWC_s_gamma*(AWC_f - AWC_s)
    AWC_i = AWC_f-AWC_s
    dAWC_v = 1/tm *(-AWC_v + AWC_v0 + np.tanh(-AWC_gain*AWC_i)) # -ve in tanh because downstep in conc activates AWC

    AIB_i = w_1*AWC_v + w_4*AIA_v
    dAIB_v = 1/tm *(-AIB_v + AIB_v0 + np.tanh(AIB_i)) # removed gains as redundant with the weights


    AIA_i = w_2 * AWC_v + w_5*AIB_v + w_6*AIY_v
    dAIA_v = 1 / tm * (-AIA_v + AIA_v0 + np.tanh(AIA_i))

    AIY_i = w_3 * AWC_v + w_7 * AIA_v
    dAIY_v = 1 / tm * (-AIY_v + AIY_v0 + np.tanh(AIY_i))

    #go_forward = (np.random.random()*2 - 1) < AIY_v

    turn = False

    if t - last_sample >= sampling_time:
        last_sample = t

        #turn = (np.random.random() * 2 - 1) < AIB_v
        #go_forward = (np.random.random() * 2 - 1) < AIY_v

        turn = (np.random.random() * 4 - 2) < (AIB_v - AIY_v)

    if turn:
        theta = np.random.random() * 2 * np.pi

    dx = speed * np.cos(theta)
    dy = speed * np.sin(theta)

    # stop worms going off the plate
    x, y = X[6], X[7]
    if abs((x+dx)**2 + (y+dy)**2)**0.5 > plate_r:
        dx = dy = 0



    return dAWC_v, dAWC_f, dAWC_s, dAIB_v, dAIA_v, dAIY_v, dx, dy

def plot_sol(solution, save_path = None):



    #plot neuron voltages
    plt.figure()
    plt.plot(solution[0, :], label = 'AWC')
    plt.plot(solution[3, :], label = 'AIB')
    plt.plot(solution[4, :], label = 'AIA')
    plt.plot(solution[5, :], label = 'AIY')
    plt.legend()

    if save_path is not None:
        plt.savefig(os.path.join(save_path, 'voltages.pdf'))

    #plot sensory neuron components

    plt.figure()
    plt.plot(solution[1,:], label = 'AWC fast')
    plt.plot(solution[2,:], label = 'AWC slow')
    plt.legend()

    if save_path is not None:
        plt.savefig(os.path.join(save_path, 'sensory.pdf'))


    # plot worm position
    fig, ax = plt.subplots(figsize = [6.4,6.4])

    # plate outline
    circle = plt.Circle([0,0], plate_r, fill=False)
    ax.add_patch(circle)

    # scoring sectors
    pos = [-origin, origin]
    rad = [2.5, 3.5]

    for p in pos:
        for r in rad:
            circle = plt.Circle(p, r, fill=False, color='gray')
            ax.add_patch(circle)

    ax.vlines(0, domain[0], domain[1], color='grey')

    ax.plot(solution[6,:], solution[7,:])
    ax.scatter(solution[6,0], solution[7,0], label = 'start')
    ax.scatter(solution[6,-1], solution[7,-1], label = 'end')


    plt.xlim(domain[0], domain[1])
    plt.ylim(domain[0], domain[1])

    plt.legend(loc = 'lower left')

    if save_path is not None:
        plt.savefig(os.path.join(save_path, 'worm.pdf'))


def plot_conc(domain):
    plt.figure()
    x = np.arange(domain[0], domain[1], 0.001)
    y = np.arange(domain[0], domain[1], 0.001)
    X, Y = np.meshgrid(x, y)

    img = concentration_func(X,Y,0)

    plt.imshow(img, extent=[domain[0], domain[1], domain[1], domain[0]])
    plt.colorbar()


def score_worm(solution):
    trajectory = solution[6:8, :]
    x = trajectory[0, :]
    y = trajectory[1, :]
    delx = origin[0] - x
    dely = origin[1] - y

    origin_dist = np.sqrt(delx ** 2 + dely ** 2)

    mirror_origin = -origin
    delx = mirror_origin[0] - x
    dely = mirror_origin[1] - y

    mirror_origin_dist = np.sqrt(delx ** 2 + dely ** 2)

    sectors = []

    if np.any(mirror_origin_dist < 2.5):
        sectors.append(-3)
    if np.any(mirror_origin_dist < 3.5):
        sectors.append(-2)

    if np.any(origin_dist < 2.5):
        sectors.append(3)

    if np.any(origin_dist < 3.5):
        sectors.append(2)

    if np.any(x < 0):
        sectors.append(-1)
    if np.any(x > 0):
        sectors.append(1)

    return sectors


def run_experiment(params, n_worms):
    global theta
    global last_sample

    sectors = []



    for i in range(n_worms):
        theta = np.random.random() * 2 * np.pi
        last_sample = 0

        sol = forward_euler(y0, params, dt, t_span[-1])

        sector = score_worm(sol)

        sectors.append(sector)

    return sectors



def get_fitnesses(population):

    # values form the no cond no odor data
    ms = np.mean(list(map(sum, no_cond_no_odour)))
    ss = np.std(list(map(sum, no_cond_no_odour)))
    mr = np.mean(list(map(lambda x: max(x) - min(x), no_cond_no_odour)))
    sr = np.std(list(map(lambda x: max(x) - min(x), no_cond_no_odour)))

    fitnesses = []

    parameters = [AWC_f_a, AWC_f_b, AWC_s_gamma, tm, AWC_v0, AWC_gain, AIB_v0, AIA_v0, AIY_v0,
         speed, w_1, w_2, w_3, w_4, w_5, w_6, w_7]
    all_sectors = []

    for p in population:

        params = parameters + list(p)

        sectors= run_experiment(params, n_worms)

        all_sectors.append(sectors)
        mean_score = np.mean(list(map(sum, sectors)))
        std_score = np.std(list(map(sum, sectors)))
        mean_range = np.mean(list(map(lambda x: max(x) - min(x), sectors)))
        std_range = np.std(list(map(lambda x: max(x) - min(x), sectors)))

        fitness = - (abs(mean_score-ms) + abs(mean_range-mr) + abs(std_score-ss) + abs(std_range-sr))

        fitnesses.append(fitness)
        print(p, fitness)

    return fitnesses, all_sectors


def evolve():

    population = np.random.random(size = (pop_size, 1))*10 + 0.05 # population of sampling times


    for i in range(n_gens):
        fitnesses = get_fitnesses(population)

        order = np.argsort(fitnesses)[::-1]

        fitnesses = np.array(fitnesses)[order]
        print(fitnesses)
        population = population[order]

        sol = solve_ivp(xdot, t_span, y0, t_eval=np.arange(t_span[-1]), args=(parameters[:-1] + list(population[0]),))
        save_path = os.path.join('working_dir', 'gen'+str(i))
        os.makedirs(save_path, exist_ok=True)
        plot_sol(sol.y, save_path=save_path)

        population[int(pop_size*0.4): int(pop_size*0.8)] += np.random.random(size = (int(pop_size*0.8)- int(pop_size*0.4), 1))*2 - 1.

        population[int(pop_size*0.8):] = np.random.random(size = (pop_size-int(pop_size*0.8), 1))*10 + 0.05

        population[population < 0.05] = 0.05

        print('max: ', np.max(fitnesses), population[0])
        print('mean: ', np.mean(fitnesses))

def param_scan(start, stop, step, save_path = './working_dir/param_scan', plot=True):
    os.makedirs(save_path, exist_ok = True)
    population = np.arange(start, stop, step).reshape(-1, 1)

    fitnesses, all_sectors  = get_fitnesses(population)



    if plot:
        plt.plot(population, fitnesses)


    np.save(save_path + '/params.npy',population)
    np.save(save_path + '/fitnesses.npy',fitnesses)
    np.save(save_path + '/sectors.npy', all_sectors)

    if plot:
        plt.show()


def forward_euler(y0, params, dt, tmax):

    y = y0
    all_ys = [y0]

    for t in np.arange(0, tmax+dt, dt):
        y = y + np.array(xdot(t, y, params))*dt
        all_ys.append(y)

    return np.array(all_ys).T




no_cond_no_odour, no_cond_odour, aversive_odour, sex_odour = load_data('./data/behaviourdatabysector_NT.csv')



n_gens = 100
pop_size = 100
n_worms = 1000 # number of worms in each experiment

origin = np.array([4.5, 0.])
# starting params from gosh et al
tm = 0.5 #s
AIB_v0 = AIA_v0 = AIY_v0 = AWC_v0 = 0
AWC_gain = 2
AWC_f_a = 4 #1/s
AWC_f_b = 15 #1/s
AWC_s_gamma = 2 #1/s
speed = 0.11 #mm/s

domain = [-3,3]
plate_r = 3
w_2 = w_3 = w_4 = w_5 = -2 # -ve weights
w_1 = w_6 = w_7 = 2 # +ve weights

sample_time = 0.01 #s
parameters = [AWC_f_a, AWC_f_b, AWC_s_gamma, tm, AWC_v0, AWC_gain, AIB_v0,  AIA_v0, AIY_v0,
         speed, w_1, w_2, w_3, w_4, w_5, w_6, w_7, sample_time]



t_span = [0, 1200] #s

y0 = [0, 0, 0, 0, 0, 0, 0, 0]
#sol = solve_ivp(xdot, t_span, y0, t_eval = np.arange(t_span[-1]), args = (p,)).y
dt = 0.1




plt.violinplot(list(map(sum, no_cond_no_odour)))
plt.ylim(bottom=-6.1, top=6.1)
plt.figure()
plt.violinplot(list(map(lambda x: max(x) - min(x), no_cond_no_odour)))

print(len(no_cond_no_odour))
print(no_cond_no_odour)
print('score', np.mean(list(map(sum, no_cond_no_odour))), 'score std', np.std(list(map(sum, no_cond_no_odour))), 'range', np.mean(list(map(lambda x: max(x) - min(x), no_cond_no_odour))), 'range std', np.std(list(map(lambda x: max(x) - min(x), no_cond_no_odour))))


no_cond_no_odour = run_experiment(parameters, 35)
print(no_cond_no_odour)
plt.figure()
plt.violinplot(list(map(sum, no_cond_no_odour)))
plt.ylim(bottom=-6.1, top=6.1)
plt.figure()
plt.violinplot(list(map(lambda x: max(x) - min(x), no_cond_no_odour)))
print(len(no_cond_no_odour))
print('score', np.mean(list(map(sum, no_cond_no_odour))), 'score std', np.std(list(map(sum, no_cond_no_odour))), 'range', np.mean(list(map(lambda x: max(x) - min(x), no_cond_no_odour))), 'range std', np.std(list(map(lambda x: max(x) - min(x), no_cond_no_odour))))

#plt.show()
plt.close('all')

param_scan(0.06, 0.121, 0.01)