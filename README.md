## Worm neural networks

- learn_sample_t.py is the code I used to fit the sampling time of the worms
- fit_to_behaviour is the code I used to fit worms to the behavioural assay
- code to run simulations of different parameter sets is found in run_worms_sim.py and instructions to use this from the command line are below

usage: 
 ```console 
 run_worms_sim.py [-h] [--weights_file WEIGHTS_FILE] [--weights WEIGHTS]
                        [--out_dir OUT_DIR] [--plot PLOT] [--opt OPT]
                        [--n_worms N_WORMS] 
```

optional arguments:
```console 
    -h, --help            show this help message and exit

    --weights_file WEIGHTS_FILE
                        input parameter file, can be csv or saved numpy array

    --weights WEIGHTS     can be used to quickly simulate a set of parameters,
                        either this or --in_file must be specified, if both
                        specified --in_file will be used

    --out_dir OUT_DIR     directory to save results in, default is ./working_dir

    --plot PLOT           1 to plot 0 to not, default is 1

    --opt OPT             A to run behaviour assay, C to run calcium plot, B to
                        run both, default is B

    --n_worms N_WORMS     number of worms to simulate in each experiment for the
                        violin plots, default is 100. If --calcium=1 this
                        argument is ignored as only one worm is required
```

examples:
```console
python run_worms_sim.py --weights_file test_weights.csv --opt B --n_worms 10
```