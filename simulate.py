from xdot import xdot, concentration_func
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib as mpl
mpl.use('tkagg')
import matplotlib.pyplot as plt
import os
import time

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

    score = 0
    score += -3*np.any(mirror_origin_dist < 2.5)
    score += -2*np.any(mirror_origin_dist < 3.5)

    score += 3 * np.any(origin_dist < 2.5)
    score += 2 * np.any(origin_dist < 3.5)

    score += -1 * np.any(x < 0)
    score += 1 * np.any(x > 0)

    return score




def get_scores(population):

    scores = []

    parameters = [AWC_f_a, AWC_f_b, AWC_s_gamma, tm, AWC_v0, AWC_gain, AIB_v0, AIA_v0, AIY_v0,
         speed, w_1, w_2, w_3, w_4, w_5, w_6, w_7]

    for p in population:

        params = parameters[:-7] + list(p)

        sol = solve_ivp(xdot, t_span, y0, t_eval=np.arange(t_span[-1]), args=(params,))
        score = score_worm(sol.y)
        scores.append(score)
        print(p, score)

    return scores





def evolve():

    population = np.random.random(size = (pop_size, 7))*20 - 10


    for i in range(n_gens):
        scores = get_scores(population)

        order = np.argsort(scores)[::-1]

        population = population[order]

        sol = solve_ivp(xdot, t_span, y0, t_eval=np.arange(t_span[-1]), args=(parameters[:-7] + list(population[0]),))
        save_path = os.path.join('working_dir', 'gen'+str(i))
        os.makedirs(save_path, exist_ok=True)
        plot_sol(sol.y, save_path=save_path)

        population[int(pop_size*0.4): int(pop_size*0.8)] += np.random.random(size = (int(pop_size*0.8)- int(pop_size*0.4), 7))*0.2 - 0.1

        population[int(pop_size*0.8):] = np.random.random(size = (pop_size-int(pop_size*0.8), 7))*20 - 10

        print('max: ', np.max(scores), population[0])
        print('mean: ', np.mean(scores))


def forward_euler(y0, params, dt, tmax):

    y = y0
    all_ys = [y0]
    for t in np.arange(0, tmax+dt, dt):

        y = y + np.array(xdot(t, y, params))*dt
        all_ys.append(y)
    return np.array(all_ys).T

n_gens = 100
pop_size = 100
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

sample_time = 0.000001
parameters = [AWC_f_a, AWC_f_b, AWC_s_gamma, tm, AWC_v0, AWC_gain, AIB_v0,  AIA_v0, AIY_v0,
         speed, w_1, w_2, w_3, w_4, w_5, w_6, w_7, sample_time]



t_span = [0, 1200] #s

y0 = [0, 0, 0, 0, 0, 0, 0, 0]
#sol = solve_ivp(xdot, t_span, y0, t_eval = np.arange(t_span[-1]), args = (p,)).y
dt = 0.1 #from paper
t = time.time()
sol = forward_euler(y0, parameters, dt, t_span[-1])
print(time.time()-t)
#evolve()

print(score_worm(sol))

plot_sol(sol)
plot_conc(domain)
plt.show()