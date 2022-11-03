import utils
from utils.settings import Settings as s_dict
from utils.misc import DictList
import PySimpleGUI as sg
import numpy as np
import torch 
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
from matplotlib import pyplot as plt
import mpl_toolkits.mplot3d
from matplotlib import cm
import os 
import os.path 
import sys 
import logging
import pandas as pd
from sklearn.decomposition import PCA 
from sklearn.cluster import KMeans
import fire
import json 

# parameters =======================================================================================================================
# first get some important default folders
cwd = os.getcwd()  # change this if you put exp_gui.py in a different folder, or launch it from like a weird place

# the following assumes that cwd is either the directory 'NLD', or the parent directory
base_name = 'NLD'
if cwd.split(os.path.sep)[-1] != base_name:
    default_folder_name = os.path.sep.join([cwd, base_name, 'settings', 'past_experiments'])
    default_results_folder = os.path.sep.join([cwd, base_name, 'results'])
    cache_folder = os.path.sep.join([cwd, base_name, 'exp_gui_cache'])
    assumed_cwd = os.path.sep.join([cwd, base_name])
else:
    default_folder_name = os.path.sep.join([cwd, 'settings', 'past_experiments'])
    default_results_folder = os.path.sep.join([cwd, 'results'])
    cache_folder = os.path.sep.join([cwd, 'exp_gui_cache'])
    assumed_cwd = cwd 
# make sure that the cache folder actually exists
if not os.path.exists(cache_folder):
    os.makedirs(cache_folder)
sampler_path_format = '{exp_id}-{model_key}-latent_samples.npy'  # data for the initial distribution
samples_path_format = '{exp_id}-{model_key}-stationary_samples.npy'  # samples from the stationary distribution
post_flow_path_format = '{exp_id}-{model_key}-post_flow.npy'  # same samples after gradient flow
default_session_storage_folder = f'{cache_folder}{os.path.sep}sessions'
if not os.path.exists(default_session_storage_folder):
    os.makedirs(default_session_storage_folder)
default_session_storage_path = f'{default_session_storage_folder}{os.path.sep}exp_gui_session.json'


# other parameters
small_blank_space = 4*' '
cpu = torch.device('cpu')

# for creating samplers and samples
batch_size = 500
skip_first_i_times = 1
step_size = 40
num_samples = 1000
max_gradient_flow_steps = 50000 

# for plotting
default_relative_plot_margin = 1.1

# for analysing clusters
in_over_out_threshold = 1e-2
# if in-cluster distance / out of cluster distance is smaller than this, we automatically label things as successful
# (but after we still check things by hand)

# =====================================================================================================================================

# global state ===================================================================================================================
current = s_dict()  # will contain things like salt, experiment id, etc.
current.folder_name = default_folder_name
current.results_folder = default_results_folder
current.session_storage_path = default_session_storage_path
current.session_loading_path = default_session_storage_path

memory = s_dict()  # will contain data from earlier

# ================================================================================================================================

# stuff for plotting ================================================================================================================================
matplotlib.use("TkAgg")


def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side="top", fill="both", expand=1)
    return figure_canvas_agg


class MatplotlibCanvas:
    """ 
    So if I understood correctly, the way to modify your drawings to a canvas is to modify the underlying matplotlib figure and axes
    I.E. I need to keep track of not only the canvas element, but also all associated figures, axes, and FigureCanvasTkAgg objects
    Hence, this container
    """
    DEFAULT_SIZE = (350, 350)

    def __init__(self, canvas_elem, key=None):
        self.canvas_elem = canvas_elem 
        self.canvas = canvas_elem.TKCanvas

        kwargs = {}
        if canvas_elem.metadata:
            kwargs.update(canvas_elem.metadata)

        self.fig, self.ax = plt.subplots(subplot_kw=kwargs)
        dpi = self.fig.get_dpi()

        w, h = self.canvas_elem.Size 
        if w is None or h is None:
            w, h = self.DEFAULT_SIZE
        fig_size = (w/dpi, h/dpi)
        self.key = key

        self.fig.set_size_inches(fig_size)
        self.fig_agg = draw_figure(self.canvas, self.fig)


def df_to_mc(df: pd.DataFrame, mc:MatplotlibCanvas):
    """
    Draw a dataframe as a table to a MatplotlibCanvas
    """
    if not df.empty:
        # do the figure stuff
        mc.ax.cla() # clear the axis
        mc.ax.axis('off')
        
        # draw the table
        w = mc.fig.get_figwidth()
        h = mc.fig.get_figheight()
        table = pd.plotting.table(mc.ax, df, loc='center')
        table.auto_set_font_size(False)
        table.auto_set_column_width(col=range(len(df.columns)))
        mc.fig_agg.draw()
    
    else:
        mc.ax.cla()
        mc.ax.axis('off')
        mc.fig_agg.draw()

    
# ================================================================================================================================

# useful helper functions to get things
# actually most of the modular functionality is in this section at this point.

def get_salt_list(folder_name):
    try:
        contents = os.listdir(folder_name)
    except FileNotFoundError:
        contents = []
        logging.warning(f"directory for salt list not found: {folder_name}")
    salt_list = [
        candidate 
        for candidate in contents
        if os.path.isdir(os.path.join(folder_name, candidate))
    ]
    return salt_list 


current.salt_list = get_salt_list(current.folder_name)  # use it at the start of the program 
print(f'At start: {current.salt_list=}')


def sort_key_experiments(entry):
    return int(entry.split('-')[-1])


def get_experiment_ids():
    folder_name = current.experiment_settings_folder
    if 'experiment_ids' not in memory[current.salt]:
        try:
            contents = os.listdir(folder_name)
        except FileNotFoundError:
            contents = []
            logging.warning(f"directory for settings list not found: {folder_name}")
        experiment_ids = [
            candidate.split('.')[0]  # get the name without the file extension 
            for candidate in contents
            if candidate[-4:] == '.txt'
        ]
        experiment_ids.sort(key=sort_key_experiments)
        memory[current.salt].experiment_ids = experiment_ids
    return memory[current.salt].experiment_ids


def get_experimental_settings():
    if 'experimental_settings' not in memory[current.salt]:
        experimental_settings = {
            exp_id: utils.settings.load_settings(
                f'{current.folder_name}/{current.salt}/{exp_id}.txt',
                config=utils.common_components.config
            )
            for exp_id in current.experiment_ids
        }
        memory[current.salt].experimental_settings = experimental_settings

    return memory[current.salt].experimental_settings


def get_settings_tables():
    current_memory = memory[current.salt]
    if 'overdamped_table' not in current_memory or 'underdamped_table' not in current_memory:
        overdamped_settings = {}
        underdamped_settings = {}
        for exp_id, settings in current_memory.experimental_settings.items():
            if settings.overdamped:
                overdamped_settings[exp_id] = settings 
            else:
                underdamped_settings[exp_id] = settings 
        current_memory.overdamped_table = utils.settings.get_difference_table(overdamped_settings)
        current_memory.underdamped_table = utils.settings.get_difference_table(underdamped_settings)
    return current_memory.overdamped_table, current_memory.underdamped_table


def register_experiment_in_memory():
    if 'experiments' not in memory:
        memory.experiments = s_dict()
    if current.experiment_id not in memory.experiments:
        memory.experiments[current.experiment_id] = s_dict()


def get_profile_from_name(name):
    # NB assumes the format f'{exp_id}-{profile_key}-{run}.pth'
    return name.split('-')[-2]


def sort_key_models(key):
    return int(key.split('-')[-1].split('.')[0])


def get_model_dict(only_gradient_models=True):
    exp_id = current.experiment_id
    if 'models' not in memory.experiments[exp_id]:
        stored_models = [
            param_file for param_file in os.listdir(current.results_folder) if f'{exp_id}-' in param_file
        ]
        stored_models = sorted(stored_models, key=sort_key_models)
        models = DictList()
        for param_file in stored_models:
            profile_key = get_profile_from_name(param_file)
            if profile_key in ('ga', 'gr') or not only_gradient_models:
                model = utils.common_components.create_model(
                    current.experiment.settings.profiles[profile_key],
                    current.experiment.settings
                )
                model.load_state_dict(torch.load(os.path.join(current.results_folder, param_file), map_location=cpu))
                models[f'{profile_key}'] = model # automatically appends after first one

        memory.experiments[exp_id].models = models 
    return memory.experiments[exp_id].models 


def prepare_experiment():
    exp_id = current.experiment_id
    for storage_key in (
            'samplers',
            'samples',
            'post_flow',
            'pca',
            'embedded_samples',
            'embedded_post_flow',
            'plot_limits',
            'clustering'
    ):
        if storage_key not in memory.experiments[exp_id]:
            # don't actually make the samplers, just make place holders for them
            # because creating a sampler takes a lot of time, so it should only be created
            # when necessary
            dictlist = DictList()
            for key in memory.experiments[exp_id].models:
                dictlist[dictlist.split_key(key)[0]] = None 
            memory.experiments[exp_id][storage_key]= dictlist 
            current.experiment[storage_key] = dictlist
        else:
            current.experiment[storage_key] = memory.experiments[exp_id][storage_key]  


def transform_data_path(data_path):
    if data_path[:2] in ('./', '.\\'):
        return os.path.sep.join([assumed_cwd, data_path[2:]])
    else:
        # data_path has a different format than expected, so there's no point in transforming it blindly
        return data_path


@torch.no_grad()
def get_sampler(model_key):
    exp_id = current.experiment_id
    settings = current.experiment.settings
    model = memory.experiments[exp_id].models[model_key]
    if current.experiment.samplers[model_key] is None:  # NB current.experiment.samplers is memory.experiments[exp_id].samplers
        # we'll have to create a sampler
        # let's first check if we might have cached one on an earlier run
        cache_path = os.path.sep.join([
            cache_folder,
            sampler_path_format.format(exp_id=exp_id, model_key=model_key)
        ])
        if os.path.exists(cache_path):
            # get the sampler from cache
            initial_data = torch.from_numpy(np.load(cache_path))
        else:
            # we're going to need the training data to produce a sampler
            if current.experiment.training_data is None:
                if 'max_sequence_length' in settings:
                    current.experiment.training_data = np.load(
                        transform_data_path(settings.data_path)
                    )[:-settings.test_size, :settings.max_sequence_length, :]
                else:
                    current.experiment.training_data = np.load(
                        transform_data_path(settings.data_path)
                    )[:-settings.test_size]

            training_data = current.experiment.training_data

            # feed the data to the model to get latent trajectories
            device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
            ts = torch.linspace(
                    settings.t0 if 't0' in settings else 0., 
                    settings.t1 if 't1' in settings else training_data.shape[1]//10, 
                    settings.nt if 'nt' in settings else training_data.shape[1], 
                    device=device
                    )
            results = []
            model.to(device)
            for index in range(training_data.shape[0] // batch_size):
                batch = torch.from_numpy(
                    training_data[index*batch_size: (index+1)*batch_size]
                ).to(device=device, dtype=torch.float)
                if not settings.overdamped:
                    _, _, positions, _ = model(batch, ts, settings.noise_std, dt=settings.dt, return_latent=True)
                else:
                    _, _, positions = model(batch, ts, settings.noise_std, dt=settings.dt, return_latent=True)
                results.append(
                    positions[skip_first_i_times*step_size::step_size].detach().cpu()
                )
                del positions
            initial_data = torch.cat(results, 0).view(-1, settings.latent_size)
            # save this data so we don't have to do this whole process again for this model.
            np.save(cache_path, initial_data.numpy())
            # and move the model back to the cpu if it was on the gpu
            model.to(cpu)

        # make an empirical distribution out of this initial data as a starting distribution for the GHMCSampler. 
        empirical_distribution = utils.energy_tools.EmpiricalDistribution(initial_data, add_momentum=True)
        sampler = utils.energy_tools.GHMCSampler(
            settings.latent_size,
            model.energy_landscape,
            inverse_temperature=model.beta,
            initial_distribution=empirical_distribution
        )
        current.experiment.samplers[model_key] = sampler 
    return current.experiment.samplers[model_key]
    

def get_samples(model_key):
    """get_samples

    side-effect: if absent from memory, the new samples are stored in memory

    :param model_key: model key (e.g. 'ga-0')
    :return: samples: torch.Tensor
    """
    # N.B. results in a torch.Tensor
    if current.experiment.samples[model_key] is None:  # NB current.experiment.samples is memory.experiments[current.experiment_id].samples
        samples_path = os.path.sep.join([
            cache_folder,
            samples_path_format.format(exp_id=current.experiment_id, model_key=model_key)
            ])
        # try to get everything from cache
        if os.path.exists(samples_path):
            np_samples = np.load(samples_path)
            current.experiment.samples[model_key] = torch.from_numpy(np_samples).to(dtype=torch.float32)
        else:
            sampler = current.experiment.samplers[model_key]
            samples = sampler(num_samples) 
            current.experiment.samples[model_key] = samples 
            with torch.no_grad():
                np.save(samples_path, samples.numpy())
    return current.experiment.samples[model_key]


def get_post_flow(model_key):
    """get_post_flow

    side-effect: if absent from memory, the new post flow is stored in memory

    :param model_key: model key (e.g. 'ga-0')
    :return: post_flow: torch.Tensor
    """
    # N.B. results in a torch.Tensor
    if current.experiment.post_flow[model_key] is None:  # NB current.experiment.post_flow is memory.experiments[current.experiment_id].post_flow
        # first try to get stuff from cache
        post_flow_path = os.path.sep.join([
            cache_folder,
            post_flow_path_format.format(exp_id=current.experiment_id, model_key=model_key)
        ])
        if os.path.exists(post_flow_path):
            post_flow = torch.from_numpy(
                np.load(post_flow_path)
            ).to(dtype=torch.float32)
            current.experiment.post_flow[model_key] = post_flow 
        else:
            samples = current.experiment.samples[model_key]
            post_flow = utils.energy_tools.gradient_flow(
                current.model.model.energy_landscape,
                samples,
                max_steps=max_gradient_flow_steps
            )
            current.experiment.post_flow[model_key] = post_flow 
            with torch.no_grad():
                np.save(post_flow_path, post_flow.numpy())
    return current.experiment.post_flow[model_key]


@torch.no_grad()
def get_pca_and_embedded(model_key):
    """get_pca_and_embedded

    side effect: if absent from memory, all return values are stored in memory

    :param model_key: model key (e.g. 'ga-0')
    :return: pca: PCA, embedded_samples: np.ndarray, embedded_post_flow: np.ndarray 
    """
    if current.experiment.pca[model_key] is None:
        post_flow = current.experiment.post_flow[model_key]
        pca = PCA(n_components=2)
        pca.fit(post_flow)
        current.experiment.pca[model_key] = pca 

        current.experiment.embedded_post_flow[model_key] = pca.transform(post_flow.numpy())
        current.experiment.embedded_samples[model_key] = pca.transform(current.experiment.samples[model_key].numpy())

    return (
        current.experiment.pca[model_key],
        current.experiment.embedded_samples[model_key],
        current.experiment.embedded_post_flow[model_key]
    )


def get_plot_limits(model_key):
    if current.experiment.plot_limits[model_key] is None:
        # compute default plot_limits
        emb_samp = current.experiment.embedded_samples[model_key]
        emb_pf = current.experiment.embedded_post_flow[model_key]
        _x_min, _y_min = np.minimum(emb_samp.min(axis=0), emb_pf.min(axis=0))
        _x_max, _y_max = np.maximum(emb_samp.max(axis=0), emb_pf.max(axis=0))

        _x_range = _x_max - _x_min
        _y_range = _y_max - _y_min

        _x_mid = _x_min + _x_range/2 
        _y_mid = _y_min + _y_range/2

        x_min = default_relative_plot_margin * (_x_min -_x_mid) + _x_mid
        y_min = default_relative_plot_margin * (_y_min - _y_mid) + _y_mid

        x_max = default_relative_plot_margin * (_x_max - _x_mid) + _x_mid
        y_max = default_relative_plot_margin * (_y_max - _y_mid) + _y_mid

        current.experiment.plot_limits[model_key] = {
            'x_min': x_min,
            'x_max': x_max,
            'y_min': y_min,
            'y_max': y_max 
        }
    pl = current.experiment.plot_limits[model_key] 
    return (
        pl['x_min'],
        pl['x_max'],
        pl['y_min'], 
        pl['y_max']
    )


def average_in_and_out_of_cluster_distance(points, labels):
    average_in_cluster_distances = np.empty(shape=(points.shape[0]), dtype=np.float32)
    # ^ will be [sum(d(x,y) for y in c(x))/(size(c(x))-1) for x in points] where c(x) is the cluster of x
    average_out_of_cluster_distances = np.empty(shape=(points.shape[0]), dtype=np.float32)
    # ^ will be [sum(d(x,y) for y in points if y not in c(x))/(size(points) - size(c(x))) for x in points]

    for index, x in enumerate(points):
        in_cluster_indices = (labels == labels[index])
        out_of_cluster_indices = np.logical_not(in_cluster_indices)

        average_in_cluster_distances[index] = np.sum(
            np.linalg.norm(x[None, :] - points[in_cluster_indices], axis=-1)) / (
                                                          np.sum(in_cluster_indices.astype(np.float32)) - 1)

        average_out_of_cluster_distances[index] = np.mean(
            np.linalg.norm(x[None, :] - points[out_of_cluster_indices], axis=-1))

    return np.mean(average_in_cluster_distances), np.mean(average_out_of_cluster_distances)


def get_clustering(model_key):
    if current.experiment.clustering[model_key] is None:
        clustering = s_dict()
        post_flow = current.experiment.post_flow[model_key]
        clustering.k_means = KMeans(3).fit(post_flow)
        clustering.labels = clustering.k_means.labels_ 
        clustering.post_flow_in_cluster_distance, clustering.post_flow_out_of_cluster_distance = average_in_and_out_of_cluster_distance(post_flow, clustering.labels)

        # make a prediction on whether a triple well potential was successfully learned
        clustering.success_prediction = clustering.post_flow_in_cluster_distance/clustering.post_flow_out_of_cluster_distance < in_over_out_threshold
        clustering.success_selection = str(clustering.success_prediction)

        # find out what percentage of samples is in each well
        clustering.percentages = (clustering.labels[:, None] == np.arange(3, dtype=clustering.labels.dtype)).mean(axis=0)
        clustering.sorted_percentages = np.sort(clustering.percentages)

        # store all of this
        current.experiment.clustering[model_key] = clustering 
        assert memory.experiments[current.experiment_id].clustering[current.model.key] is clustering
    return current.experiment.clustering[model_key]

# ================================================================================================================================

# stuff for saving and loading sessions ================================================================================================================================


def maybe_array_to_list(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value 


def maybe_list_to_array(value):
    if isinstance(value, list):
        return np.array(value, dtype=np.float32)
    return value 


def k_means_to_params(k_means: KMeans):
    params_dict = k_means.get_params()
    full_dict = {
        key: maybe_array_to_list(getattr(k_means, key))
        for key in dir(k_means)
        if key.endswith('_') and not key.endswith('__') and not key in ('_repr_html_', '_repr_mimebundle_')
    }
    full_dict['_n_threads'] = k_means._n_threads
    return [params_dict, full_dict]


def params_to_k_means(params_dict:dict, full_dict: dict) -> KMeans:
    k_means = KMeans()
    k_means.set_params(**params_dict)
    for key, value in full_dict.items():
        setattr(k_means, key, maybe_list_to_array(value))
    return k_means 
    

def extract_session_dict_from_memory():
    session_dict = {}
    
    # first extract everything salt-specific
    for key, value in memory.items():
        if key != 'experiments':
            session_dict[key] = {
                'experiment_ids': list(value.experiment_ids),
                'underdamped_table': value.underdamped_table.to_json(),
                'overdamped_table': value.overdamped_table.to_json()
            }
    
    # now time for experiment-specific stuff
    experiments_dict = {}
    session_dict['experiments'] = experiments_dict

    for exp_id, exp in memory.experiments.items():
        experiments_dict[exp_id] = {
            'plot_limits': {
                base_key: list(exp.plot_limits[base_key])
                for base_key in exp.plot_limits.base_keys()
            },
            'clustering': {
                base_key: [
                    {
                        'k_means_params': k_means_to_params(clustering.k_means),
                        'post_flow_in_cluster_distance': clustering.post_flow_in_cluster_distance,
                        'post_flow_out_of_cluster_distance': clustering.post_flow_out_of_cluster_distance,
                        'percentages': list(clustering.percentages),
                        'sorted_percentages': list(clustering.sorted_percentages),
                        'success_prediction': clustering.success_prediction,
                        'success_selection': clustering.success_selection

                    } if clustering is not None else None 
                    for clustering in exp.clustering[base_key]
                ]
                for base_key in exp.clustering.base_keys()
            }
        }

    # some stuff that's in current instead of in memory:
    session_dict['folder_name'] = current.folder_name
    session_dict['results_folder'] = current.results_folder 
    return session_dict  # this thing should be json-serializable at this point


def store_session_dict_in_memory(session_dict):
    # first store all salt-related stuff
    for key, storage in session_dict.items():
        if key not in ('experiments', 'folder_name', 'results_folder'):
            this_salt = s_dict()
            memory[key] = this_salt 
            this_salt.experiment_ids = storage['experiment_ids']
            # can skip experimental settings, those aren't in the json
            this_salt.underdamped_table = pd.read_json(storage['underdamped_table'])
            this_salt.overdamped_table = pd.read_json(storage['overdamped_table'])

    # next, store all experiment-related stuff
    experiments = s_dict()
    for exp_id, exp in session_dict['experiments'].items():
        experiment = s_dict()
        experiments[exp_id] = experiment 

        experiment.plot_limits = DictList()
        for base_key in exp['plot_limits']:
            for model_plot_limits in exp['plot_limits'][base_key]:
                experiment.plot_limits[base_key] = model_plot_limits  # automatically appends (does not overwrite)
        
        experiment.clustering = DictList()
        for base_key in exp['clustering']:
            for clustering_dict_or_none in exp['clustering'][base_key]:
                if clustering_dict_or_none is None:
                    clustering = None 
                else:
                    clustering_dict = clustering_dict_or_none 
                    clustering = s_dict()
                    clustering.k_means = params_to_k_means(*clustering_dict['k_means_params'])
                    clustering.labels = clustering.k_means.labels_ 
                    for key in (
                        'post_flow_in_cluster_distance', 
                        'post_flow_out_of_cluster_distance',
                        'percentages',
                        'sorted_percentages',
                        'success_prediction',
                        'success_selection'
                        ):
                        clustering[key] = maybe_list_to_array(clustering_dict[key])
                experiment.clustering[base_key] = clustering  # automatically appends (does not overwrite)

    if 'experiments' not in memory:
        memory.experiments = experiments
    else:
        memory.experiments.update(experiments)

    # stuff that goes into current instead of memory
    current.folder_name = session_dict['folder_name']
    current.salt_list = get_salt_list(current.folder_name)
    window['-salt list-'].update(values=current.salt_list)
    window['-settings folder-'].update(value=current.folder_name)
    logging.debug(f'-salt list- updated with values from stored session: {current.salt_list=}')

    current.results_folder = session_dict['results_folder']
    window['-results folder-'].update(value=current.results_folder)
    

def np_type_converter(o):
    if isinstance(o, (np.float32, np.float16, np.float64)):
        return float(o)
    elif isinstance(o, (np.int16, np.int32, np.int64)):
        return int(o)
    elif isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(f'Object of type {type(o)} is not JSON serializable')


def store_session(path):
    session_dict = extract_session_dict_from_memory()
    with open(path, 'w') as session_file:
        json.dump(
            session_dict,
            session_file,
            default=np_type_converter
        )


def load_session(path):
    with open(path, 'r') as session_file:
        session_dict = json.load(session_file)
        store_session_dict_in_memory(session_dict)

# ================================================================================================================================


# layout:  ================================================================================================================================
# # left column: 
left_sub_column = [
    [sg.Text("Settings folder: "), sg.In(size=(25, 1), default_text=default_folder_name, enable_events=True, key="-settings folder-"), sg.FolderBrowse()],
    [sg.Listbox(values=current.salt_list, enable_events=True, size=(40, 18), key="-salt list-")]
]
middle_sub_column = [
    [sg.Text("Select Experiment:")],
    [sg.Listbox(values=[], enable_events=True, size=(40, 18), key="-experiment list-", bind_return_key=True)]
]
right_sub_column = [
    [sg.Text("Results folder: "), sg.In(size=(25, 1), default_text=default_results_folder, enable_events=True, key='-results folder-'), sg.FolderBrowse()],
    [sg.Text("Select model:")],
    [sg.Listbox(values=[], enable_events=True, size=(40, 15), key='-model list-', bind_return_key=True)],
]

left_column = [
    [sg.Column(left_sub_column), sg.Column(middle_sub_column),  sg.Column(right_sub_column)], 
    [sg.Canvas(key='-underdamped table-', size=(900, 350), expand_x=False)], 
    [sg.Canvas(key='-overdamped table-', size=(900, 350), expand_x=False)],
    ]

# # right column:
right_column = [
    [
        sg.Text('Store session:'), 
        sg.In(size=(25, 1), default_text=default_session_storage_path, enable_events=True, key='-session path store-'),
        sg.FolderBrowse(),
        sg.Button(button_text='Save', key='-save session-', enable_events=True),
        sg.VerticalSeparator(),
        sg.Text('Load session:'),
        sg.In(size=(25, 1), default_text=default_session_storage_path, enable_events=True, key='-session path load-'),
        sg.FolderBrowse(),
        sg.Button(button_text='Load', key='-load session-', enable_events=True)
    ],
    [sg.Canvas(key='-post flow-'), sg.Canvas(key='-pre flow-')],
    [sg.Canvas(key='-3d-', metadata={'projection': '3d'}), sg.Canvas(key='-contour-')],
    [sg.Button("Draw", key="-draw-", enable_events=True)],
    [
        sg.Text('x_min: '), sg.In(size=(8, 1), enable_events=True, key='-x min-'), 
        sg.Text(f'{small_blank_space}x_max: '), sg.In(size=(8, 1), enable_events=True, key='-x max-')
    ],
    [
        sg.Text('y_min: '), sg.In(size=(8, 1), enable_events=True, key='-y min-'), 
        sg.Text(f'{small_blank_space}y_max: '), sg.In(size=(8, 1), enable_events=True, key='-y max-')
    ],
    [
        sg.Text('Energy landscape successfully learned: '), 
        sg.Radio('True', 'success', key='-success true-', enable_events=True), 
        sg.Radio('False', 'success', key='-success false-', enable_events=True),
        sg.Radio('Dubious', 'success', key='-success dubious-', enable_events=True)
    ],
    [
        sg.Text('Relative sizes of clusters:'),
        sg.In(size=(5, 1), disabled_readonly_background_color='#FFA500', readonly=True, key='-cluster 0-'),
        sg.In(size=(5, 1), disabled_readonly_background_color='#008000', readonly=True, key='-cluster 1-'),
        sg.In(size=(5, 1), disabled_readonly_background_color='#0000FF', readonly=True, key='-cluster 2-', text_color='white')
    ],
    [sg.Text('Destination: '), sg.In(size=(60, 1), key='-target path-', enable_events=True), sg.Button("Store Report", key='-export-', enable_events=True)],
]

# # full layout
layout = [
    [
        sg.Column(left_column), 
        sg.VSeparator(),
        sg.Column(right_column)
    ]
]

window = sg.Window("experiment viewer", layout, finalize=True, resizable=True)
window.maximize()
canvases = {
    key: MatplotlibCanvas(element, key)
    for key, element in window.key_dict.items()
    if isinstance(element, sg.Canvas)
}

# =================================================================================================================================

# event loop =====================================================================================================================


def draw_current_model():
    # let's first do the energy landscape
    grid, gx, gy = utils.plot.get_grid(
        mi=current.model.x_min,
        ma=current.model.x_max,
        num=500,
        mi_y=current.model.y_min,
        ma_y=current.model.y_max
        )
    plane = current.model.pca.inverse_transform(grid)
    energy_function = utils.plot.torch_function_to_numpy(current.model.model.energy_landscape)

    energy_on_plane = energy_function(plane).squeeze(-1)
    norm = plt.Normalize(energy_on_plane.min(), energy_on_plane.max())
    colors = cm.jet(norm(energy_on_plane))

    three_d = canvases['-3d-']
    contour = canvases['-contour-']

    three_d.ax.cla()
    contour.ax.cla()

    surf = three_d.ax.plot_surface(gx, gy, energy_on_plane, facecolors=colors, shade=False)
    surf.set_facecolor((0, 0, 0, 0))

    contours = contour.ax.contour(gx, gy, energy_on_plane, 50, cmap='jet')
    # not sure if the colours will actually match between the surface and the contour :-/

    three_d.fig_agg.draw()
    contour.fig_agg.draw()

    # next we'll do the scatter plots
    mc_samples = canvases['-pre flow-']
    mc_post_flow = canvases['-post flow-']

    samples = current.model.embedded_samples
    post_flow = current.model.embedded_post_flow

    indices_per_label = current.model.clustering.k_means.labels_[None, :] == np.arange(3, dtype=int)[:, None]
    mc_samples.ax.cla()
    mc_post_flow.ax.cla()
    
    for label, color in zip(range(3), ('orange', 'green', 'blue')):
        samples_in_class = samples[indices_per_label[label]]
        post_flow_in_class = post_flow[indices_per_label[label]]
        
        mc_samples.ax.scatter(samples_in_class[:, 0], samples_in_class[:, 1], c=color)
        mc_samples.ax.set_xlim(left=current.model.x_min, right=current.model.x_max)
        mc_samples.ax.set_ylim(bottom=current.model.y_min, top=current.model.y_max)

        mc_post_flow.ax.scatter(post_flow_in_class[:, 0], post_flow_in_class[:, 1], c=color)
        mc_post_flow.ax.set_xlim(left=current.model.x_min, right=current.model.x_max)
        mc_post_flow.ax.set_ylim(bottom=current.model.y_min, top=current.model.y_max)
        
    mc_samples.fig_agg.draw()
    mc_post_flow.fig_agg.draw()


def model_selected(model_key):
    # get all relevant data for the model
    current.model = s_dict()
    current.model.key = model_key 
    current.model.model = current.experiment.models[model_key]
    current.model.sampler = get_sampler(model_key)
    current.model.samples = get_samples(model_key)
    current.model.post_flow = get_post_flow(model_key)
    current.model.pca, current.model.embedded_samples, current.model.embedded_post_flow = get_pca_and_embedded(model_key)
    current.model.x_min, current.model.x_max, current.model.y_min, current.model.y_max = get_plot_limits(model_key) 
    current.model.clustering = get_clustering(model_key)

    # update window with limits
    window['-x min-'].update(value=current.model.x_min)
    window['-y min-'].update(value=current.model.y_min)
    window['-x max-'].update(value=current.model.x_max)
    window['-y max-'].update(value=current.model.y_max)

    # update the success radio
    if current.model.clustering.success_selection == 'True':
        window['-success true-'].update(value=True)
    elif current.model.clustering.success_selection == 'False':
        window['-success false-'].update(value=True)
    else:
        window['-success dubious-'].update(value=True)

    # update the boxes with relative sizes of the clusters
    for i in range(3):
        window[f'-cluster {i}-'].update(value=f'{current.model.clustering.percentages[i]:.2f}')

    draw_current_model()


def update_limit(limit, value):
    axis, m = limit[1:-1].split(' ')
    key = f'{axis}_{m}'
    try:
        value = float(value)
        current.model[key] = value 
        current.experiment.plot_limits[current.model.key][key] = value  
    except ValueError:
        pass  # happens e.g when you empty the field, or when you start typing a negative number


def experiment_selected(experiment):
    current.experiment_id = experiment
    current.experiment = s_dict()
    current.experiment.settings = utils.settings.load_settings(
        f'{current.experiment_settings_folder}{os.path.sep}{experiment}.txt', 
        utils.common_components.config
        )
    register_experiment_in_memory()

    # get a DictList of models and prepare important place holders in the experiment
    current.experiment.models = get_model_dict()
    prepare_experiment()
    current.experiment.training_data = None 

    # update the list of models
    window['-model list-'].update(values=list(current.experiment.models.keys()))


def salt_selected(salt):
    current.salt = salt 
    if 'experiment' in current:
        del current['experiment']
    if salt not in memory:
        memory[salt] = s_dict()

    # get the available experiments
    current.experiment_settings_folder = os.path.join(current.folder_name, current.salt)
    current.experiment_ids = get_experiment_ids()

    # update the experiment list
    window['-experiment list-'].update(values=current.experiment_ids, set_to_index=0)

    # generate the tables for this salt
    current.all_experimental_settings = get_experimental_settings()
    overdamped_table, underdamped_table = get_settings_tables()
    df_to_mc(underdamped_table, canvases['-underdamped table-'])
    df_to_mc(overdamped_table, canvases['-overdamped table-'])

    # automatically select the first experiment from the experiment list
    experiment_selected(current.experiment_ids[0])


@torch.no_grad()
def main(loop_through=False):
    """main: event loop for the program

    :param loop_through: if set to True, will loop through all salt values, experiments, and models, defaults to False
        set this to True if you want to automatically do all necessary computations in one go, and write the results to cache.
    """
    if loop_through: 
        # with gpu support, there's somehow a gpu-memory leak that I can't find a fix for
        # hence:
        # torch.cuda.is_available = lambda: False
        # no gpu if we go through everything :-/
        # # ^ actually, using the cpu is not feasible for processing the full data sets.
        # # it takes way too much time.
        import time 
        import gc

        def get_time():
            return time.strftime(
                "%d-%m-%Y   %H:%M:%S",
                time.localtime()
            )

        window.refresh()
        for salt_index, salt in enumerate(current.salt_list):
            print(f'{get_time()}: move salt to {salt}')
            salt_selected(salt)
            window['-salt list-'].update(set_to_index=salt_index)

            for exp_index, experiment_id in enumerate(current.experiment_ids):
                print(f'{get_time()}: move experiment_id to {experiment_id}')
                experiment_selected(experiment_id)
                window['-experiment list-'].update(set_to_index=exp_index)

                for model_index, model_key in enumerate(current.experiment.models):
                    print(f'{get_time()}: start work on {model_key}')
                    model_selected(model_key)
                    window['-model list-'].update(set_to_index=model_index)
                    window.refresh()
                    gc.collect()

    while True:
        event, values = window.read()
        if event == 'Exit' or event == sg.WIN_CLOSED:
            break 

        if event == "-settings folder-":
            # the folder containing all experiment settings is changed
            current.folder_name = values['-settings folder-']
            current.salt_list = get_salt_list(current.folder_name)
            window['-salt list-'].update(values=current.salt_list)
            logging.debug(f'-salt list- updated with values= {current.salt_list}')

        if event == '-results folder-':
            current.results_folder = values['-results folder-']
            if 'experiment_id' in current:
                # TODO: maybe have this trigger a whipe of the models from memory
                # because otherwise, the old dictlists will remain in use
                experiment_selected(current.experiment_id)

        elif event == '-salt list-':
            # a salt was selected
            salt_selected(values['-salt list-'][0])

        elif event == '-experiment list-':
            experiment_selected(values['-experiment list-'][0])

        elif event == '-model list-':
            model_selected(values['-model list-'][0])

        elif event in (f'-{axis} {m}-' for axis in ('x', 'y') for m in ('min', 'max')):
            update_limit(event, values[event])

        elif event == '-draw-':
            draw_current_model()

        elif event == '-success true-':
            current.model.clustering.success_selection = 'True' 
        
        elif event == '-success false-':
            current.model.clustering.success_selection = 'False' 

        elif event == '-success dubious-':
            current.model.clustering.success_selection = 'Dubious'
            # for those results where it's neither a clear yes, nor a clear no.
            # these might be interesting to look into later

        elif event == '-session path store-':
            current.session_storage_path = values['-session path store-']

        elif event == '-session path load-':
            current.session_loading_path = values['-session path load-']

        elif event == '-save session-':
            store_session(current.session_storage_path)
        
        elif event == '-load session-':
            load_session(current.session_storage_path)
        
        
if __name__ == '__main__':
    try:
        fire.Fire(main)
    finally:
        import gc
        gc.collect()
        torch.cuda.empty_cache()

