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


def run_behaviour_assay(simulator, weights):
    simulator.set_mode('B')
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
            ax = axs[i]
            ax.set_ylim(bottom=-6.1, top=6.1)
            ax.violinplot(list(map(sum, sectors)))
        plot_dir = os.path.join(out_dir, 'plots')
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir, 'violin_plots.pdf'))

        for weights, sols in enumerate(all_sols):
            weights_dir = os.path.join(plot_dir, 'sols', 'weights' + str(weights))
            os.makedirs(weights_dir, exist_ok=True)
            for worm, sol in enumerate(sols):
                simulator.plot_sol(sol, save_path=os.path.join(weights_dir, 'worm' + str(worm)))


def run_calcium_experiments(simulator, weights):
    simulator.set_mode('C')

    _, calcium_sims = simulator.run_experiment_par(weights, n_worms=1)

    data_dir = os.path.join(out_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    np.save(os.path.join(data_dir, 'calcium_sims.npy'), calcium_sims)


    if plot:
        ncols = 10
        fig, axs = plt.subplots(nrows=10, ncols=ncols, figsize=(15, 7.5))

        plot_dir = os.path.join(out_dir, 'plots')
        os.makedirs(plot_dir, exist_ok=True)

        for i, c_sim in enumerate(calcium_sims):
            c_sim = c_sim[0] #just one worm
            # add to the summary plot
            r = i // ncols
            c = i % ncols
            ax = axs[r, c]
            alpha = 0.5
            lw = 1
            max_t = simulator.t_span[-1]
            dt = simulator.dt

            ax.plot(np.arange(0, max_t, dt), c_sim[0, 1:-1], label='AWC', alpha=alpha, lw=lw)
            ax.plot(np.arange(0, max_t, dt), c_sim[3, 1:-1], label='AIB', alpha=alpha, lw=lw)
            ax.plot(np.arange(0, max_t, dt), c_sim[5, 1:-1], label='AIY', alpha=alpha, lw=lw)
            # ax.plot(np.arange(0, max_t, dt), c_sim[4, 1:-1], label='AIA', alpha = alpha, lw = lw) #uncomment this line to plot AIA
            ax.set_title('worm ' + str(i + 1), fontsize=10)
            if r == 0 and c == 0:
                ax.legend(loc=2, bbox_to_anchor=(0, 2.2))

            if c != 0:
                ax.set_yticklabels([])

            if r != 9:
                ax.set_xticklabels([])

            # create the detailed plot
            d_fig, d_axs = plt.subplots(2, 2, figsize=(12.0, 8.0))
            p_c = plt.rcParams['axes.prop_cycle']
            colours = p_c.by_key()['color']

            inds = [0, 3, 5, 4]
            labels = ['AWC', 'AIB', 'AIY', 'AIA']
            for j in range(4):
                d_axs[j // 2, j % 2].plot(np.arange(0, max_t, dt), c_sim[inds[j], 1:-1], label=labels[j], c=colours[j])
                d_axs[j // 2, j % 2].legend()
                d_axs[j // 2, j % 2].set_xlabel('Time (s)')
                d_axs[j // 2, j % 2].set_ylabel('Neuron voltage')



            d_fig.suptitle('worm ' + str(i + 1))
            os.makedirs(os.path.join(plot_dir, 'detailed_calcium'), exist_ok=True)
            d_fig.savefig(os.path.join(plot_dir, 'detailed_calcium', 'weights' + str(i + 1) + '.pdf'))


        fig.savefig(os.path.join(plot_dir, 'calcium_plots.pdf'))



dir_path = os.path.dirname(os.path.realpath(__file__))

simulator = WormSimulator(dt=0.005)



worm_trapped = False
conc_interval = None


parser = argparse.ArgumentParser(description='run worm simulations')
parser.add_argument('--weights_file',  type=str, help='input parameter file, can be csv or saved numpy array')
parser.add_argument('--weights',  type=str, help='can be used to quickly simulate a set of parameters, either this or --in_file must be specified, if both specified --in_file will be used')
parser.add_argument('--out_dir', type=str, help='directory to save results in, default is ./working_dir')
parser.add_argument('--plot', type=str, help='1 to plot 0 to not, default is 1')
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

    plot = (args.plot == '1' or args.plot is None)
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
        run_behaviour_assay(simulator, weights)

    if opt in ['C', 'B']:
        run_calcium_experiments(simulator, weights)








        #np.save('/results/final_results/230111_mock/fitting_data/final_calcium_sims.npy', calcium_sims)
