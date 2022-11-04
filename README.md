# Neural Langevin Dynamics
Code accompanying the paper Neural Langevin Dynamics, under review for AISTATS 2023.

The model parameters and settings used for the tables of results can be found under models. Functions for loading the settings files can be found in ./utils/settings.py, functions for creating the models from settings can be found in ./utils/common_components.py

To create and train new models from settings files, create settings files in ./settings/future_experiments (two examples are already there), and run run_experiments.py

For analysing the results it is helpful to use exp_gui.py. To automatically have this perform relevant computations (like encoding the training data) for all models, run it with --loop_through=True

Further tools for analysing the energy landscape can be found in ./utils/experiment_analysis.py and ./energy_tools.py

The data can be found on https://ufile.io/w8fuhfxx (for 30 days, we'll provide a new link afterwards). 