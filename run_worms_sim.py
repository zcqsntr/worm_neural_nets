import math
import os
import numpy as np
import sys
import time
from wormSimulator import WormSimulator
import copy
import argparse
import matplotlib.pyplot as plt


def load_params(param_file):

    if param_file[-3:] == 'csv':
        params = np.loadtxt(open(param_file, "rb"), delimiter=",")
    elif param_file[-3:] == 'npy':
        params = np.load(param_file)
    else:
        raise ValueError('parameter file not recognised')

    return params



dir_path = os.path.dirname(os.path.realpath(__file__))

simulator = WormSimulator(dt=0.005)



worm_trapped = False
conc_interval = None


parser = argparse.ArgumentParser(description='run worm simulations')
parser.add_argument('--weights_file',  type=str, help='input parameter file, can be csv or saved numpy array')
parser.add_argument('--weights',  type=str, help='can be used to quickly simulate a set of parameters, either this or --in_file must be specified, if both specified --in_file will be used')
parser.add_argument('--out_dir', type=str, help='directory to save results in, default is ./working_dir')
parser.add_argument('--plot', type=str, help='1 to plot 0 to not, default is 0')
parser.add_argument('--opt', type=str, help='A to run behaviour assay, C to run calcium plot, B to run both, default is B')
parser.add_argument('--n_worms', type=str, help='number of worms to simulate in each experiment for the violin plots, default is 100. If --calcium=1 this argument is ignored as only one worm is required')


if __name__ == '__main__':

    args = parser.parse_args()

    if args.opt == 'A':
        opt = 'A'
    elif args.opt == 'C':
        opt = 'C'
    else:
        opt = 'B'

    plot = args.plot == '1'
    out_dir = 'working_dir' if args.out_dir is None else args.out_dir
    n_worms = int(args.n_worms) if args.n_worms is not None else 100

    if args.weights_file is not None:
        weights = load_params(args.weights_file)

    elif args.weights is not None:
        weights = np.array([list(map(float, args.weights.split(',')))])
    else:
        raise ValueError('One of --param_file or --parameters must be given')

    if weights.shape[1] != 9:
        raise ValueError('weights must have shape (-1, 9) current weights shape = {}'.format(weights.shape))

    if opt in ['A', 'B']:
        all_sectors, all_sols = simulator.run_experiment_par(weights, n_worms)
        data_dir = os.path.join(out_dir, 'data')
        os.makedirs(data_dir, exist_ok=True)
        np.save(os.path.join(data_dir, 'all_sectors.npy'), all_sectors)
        np.save(os.path.join(data_dir, 'all_sols.npy'), all_sols)


        if plot:
            ncols = 10
            nrows = math.ceil(len(all_sectors) / ncols)
            fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 7.5))

            for i, sectors in enumerate(all_sectors):
                print(sectors)
                ax = axs[i]
                ax.set_ylim(bottom=-6.1, top=6.1)
                ax.violinplot(list(map(sum, sectors)))
            plot_dir = os.path.join(out_dir, 'plots')
            os.makedirs(plot_dir, exist_ok=True)
            plt.savefig(os.path.join(plot_dir, 'violin_plots.pdf'))

            for weights,sols in enumerate(all_sols):
                weights_dir = os.path.join(plot_dir, 'sols', 'weights'+str(weights))
                os.makedirs(weights_dir, exist_ok=True)
                for worm,sol in enumerate(sols):
                    simulator.plot_sol(sol, save_path=os.path.join(weights_dir, 'worm' + str(worm)))



    if opt in ['C', 'B']:
        worm_trapped = True
        conc_interval = [10, 40]
        max_t = 70
        params = [AWC_f_a, AWC_f_b, AWC_s_gamma, tm, AWC_v0, AWC_gain, AIB_v0, AIA_v0, AIY_v0,
                  speed, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, worm_trapped, conc_interval]

        #population = np.load(path + 'gen99/population.npy')

        population = np.load(path + 'weights_population.npy')
        ncols = 10
        fig, axs = plt.subplots(nrows=10, ncols=ncols, figsize=(15, 7.5))
        calcium_sims = []

        for i in range(len(population)):
            weights = population[i]

            # positive weights
            params[10] = weights[0]
            params[15] = weights[1]
            params[16] = weights[2]
            params[17] = weights[3]

            # negative weights
            params[11] = weights[4]
            params[12] = weights[5]
            params[13] = weights[6]
            params[14] = weights[7]
            params[18] = weights[8]

            simulator.t_span[-1] = max_t
            solution = simulator.forward_euler(simulator.y0, params)
            calcium_sims.append(solution)

            # plot neuron voltages

            ax = axs[i // ncols, i  % ncols]
            ax.set_ylim(bottom=-1., top = 1.)
            ax.plot(np.arange(0, max_t, simulator.dt), solution[0, 1:-1], label='AWC')
            ax.plot(np.arange(0, max_t, simulator.dt), solution[3, 1:-1], label='AIB')
            ax.plot(np.arange(0, max_t, simulator.dt), solution[5, 1:-1], label='AIY')
            ax.legend()
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Neuron voltages')


            #simulator.plot_conc()



        #np.save('/results/final_results/230111_mock/fitting_data/final_calcium_sims.npy', calcium_sims)
