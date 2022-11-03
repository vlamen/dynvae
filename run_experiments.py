import functools
import fire
import torch
import utils
import sys
import os 
import numpy as np
import pickle
import warnings 
import shutil


def ensure_existence(path, **kwargs):
    path = path.format(**kwargs)
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def main(
    source_directory='./settings/future_experiments',
    target_directory='./settings/past_experiments/{salt}',
    logs_location='./logs/{settings_name}.log',
    results_location='./results/{settings_name}-{profile_key}-{run}.pth',
    losses_location='./losses/{settings_name}-losses.pickled',
    config_location=None,
    config_name='config',
    cleanup=True
    ):
    """main basically the entire process

    :param source_directory: directory from which to load (all) settings files for experiments, 
        defaults to './settings/future_experiments'
    :param target_directory: directory to which to move/copy the loaded settings files after running the corresponding experiment, 
        defaults to './settings/past_experiments/{salt}'
    :param logs_location:  location at which to store the log-files
        defaults to './logs/{settings_name}.log'
    :param results_location: location at which to store the model parameters
        defaults to './results/{settings_name}-{profile_key}-run.pth'
    :param losses_location: location at which to store the losses during training
        defaults to './losses/{settings_name}-losses.pickled'
    :param config_location: python module from which to get a config dictionary, necessary for the reading of a settings file
        defaults to using the config from utils.common_components
    :param config_name: string to be parsed to utils.settings.load_settings
        defaults to 'config'
    :param cleanup: whether to keep the settings files in the source directory
    """
    if config_location is None:
        config = utils.common_components.config
    else:
        import importlib
        config = importlib.import_module(config_location)
        config = config.config
    
    list_of_settings = os.listdir(source_directory)
    list_of_settings.sort()

    for settings_path in list_of_settings:
        full_settings_path = f'{source_directory}/{settings_path}'
        settings = utils.settings.load_settings(full_settings_path, config, config_name=config_name)
        ensure_existence(target_directory, salt=settings.salt_number)
        settings_name = settings_path.split('.')[0]

        log_path = logs_location.format(settings_name=settings_name)
        log_writer = utils.misc.MyWriter(log_path)
        log_print = functools.partial(print, file=log_writer)

        # get the data
        data = np.load(settings.data_path)
        if 'max_sequence_length' in settings:
            data = data[:, :settings.max_sequence_length, ...]
        train_data = data[:-settings.test_size]
        test_data = data[-settings.test_size:]
        log_print(f'Running experiment from settings at {settings_name}')
        log_print(f'loaded data from {settings.data_path}')
        log_print(f'{train_data.shape=}')
        log_print(f'{test_data.shape=}')

        # ready storage of results
        results = utils.settings.Settings()

        # start running experiments based on the settings file 
        for profile_key in settings.profiles:
            results[profile_key] = []

            log_print(2*'\n')
            log_print(80*'=')
            log_print(80*'=')
            log_print(profile_key)
            log_print(80*'=')
            log_print(80*'=')

            for run in range(settings.runs):
                log_print('\n')
                log_print(f'{run=}')
                log_print(60*'=')
                model, losses = utils.common_components.run_experiment(
                    settings.profiles[profile_key],
                    settings,
                    train_data=train_data,
                    test_data=test_data,
                    plot_losses=False,
                    on_epoch_end=log_writer.flush,
                    print=log_print
                )
                if model is not None:
                    results[profile_key].append(losses)
                    model_path = results_location.format(settings_name=settings_name, profile_key=profile_key, run=run)
                    torch.save(model.state_dict(), model_path)
                    del model 
                    log_print(f'Stored model at {model_path}')
                else:
                    log_print(f'no successfully trained model for {settings_name} with profile {profile_key} at run {run}')
                log_writer.flush()
        loss_path = losses_location.format(settings_name=settings_name)
        try:
            with open(loss_path, 'wb+') as loss_dump:
                pickle.dump(results.data, loss_dump, protocol=4)
        except Exception as e:
            warnings.warn(f'Encountered following exception while trying to write the losses for {settings_name} to {loss_path}:\n {repr(e)}')
        
        # now we are done running experiments for these settings
        # move to the target directory
        if cleanup:
            shutil.move(
                full_settings_path, 
                target_directory.format(salt=settings.salt_number)+f'/{settings_name}.txt'
                )
        else:
            shutil.copy2(
                full_settings_path,
                target_directory.format(salt=settings.salt_number)+f'/{settings_name}.txt'
            )
        log_print(f'Finished experiments from {settings_name}')
        log_writer.flush()
    print('Finished all experiments')


if __name__ == '__main__':
    sys.setrecursionlimit(5000)
    import pdb
    try:
        fire.Fire(main)
    except Exception as e:
        pdb.post_mortem()
        raise e 
    finally:
        import gc
        gc.collect()
        torch.cuda.empty_cache()
