import torch
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors
import mpl_toolkits.mplot3d
import functools
import pandas as pd


def get_grid(mi, ma, num, mi_y=None, ma_y=None):
    """
    Get a square grid for plotting
    :param mi: minimum
    :param ma: maximum
    :param num: number of grid points
    :param mi_y: minimum for y if separate from x
    :param ma_y: maximum for y if separate from x
    :return: grid, x-coordinates, y-coordinates
    """
    linspace_x = np.linspace(mi, ma, num)
    mi_y = mi_y if mi_y is not None else mi
    ma_y = ma_y if ma_y is not None else ma
    linspace_y = np.linspace(mi_y, ma_y, num)

    gx, gy = np.meshgrid(linspace_x, linspace_y, indexing='ij')

    grid = np.stack((gx, gy), axis=-1)

    return grid, gx, gy


def torch_function_to_numpy(func):
    """
    just a wrapper for functions that take and return torch tensors to
    have them take and return numpy arrays instead
    :param func: function to be wrapped
    :return: wrapped function
    """
    @functools.wraps(func)
    def wrapped_func(*args, **kwargs):
        with torch.no_grad():
            modified_args = tuple(torch.from_numpy(arg).float() if isinstance(arg, np.ndarray) else arg
                                  for arg in args)
            modified_kwargs = {key: torch.from_numpy(value).float() if isinstance(value, np.ndarray) else value
                               for key, value in kwargs.items()}

            ret_tensor = func(*modified_args, **modified_kwargs)
            ret_array = ret_tensor.numpy()
        return ret_array
    return wrapped_func


def graph_plot_3d(function, mi_x, ma_x, mi_y=None, ma_y=None, is_torch_function=True, grid_points=500, ax=None, transform=None):
    """
    plot a graph of a R^2 \to R function
    :param function: the function to be plotted
    :param mi_x: minimal value for x
    :param ma_x: maximal value for x
    :param mi_y: minimal value for y, default: mi_x
    :param ma_y: maximal value for y, default: ma_x
    :param is_torch_function: whether the function works (only) on torch tensors
        if false, data is fed to it as numpy arrays
    :param grid_points: number of grid points per axis used for plotting (default: 500)
    :param ax: axis object on which to plot this (default: create new axes object)
    :return: None
    """
    grid, gx, gy = get_grid(mi_x, ma_x, grid_points, mi_y, ma_y)
    
    if transform is not None:
        grid = transform(grid)

    if is_torch_function:
        function = torch_function_to_numpy(function)

    values_at_grid = function(grid)
    if values_at_grid.shape[-1] == 1:
        values_at_grid = values_at_grid.squeeze(-1)

    norm = plt.Normalize(values_at_grid.min(), values_at_grid.max())
    colors = cm.jet(norm(values_at_grid))

    if ax is None:
        ax = plt.axes(projection='3d')

    surf = ax.plot_surface(gx, gy, values_at_grid, facecolors=colors, shade=False)
    surf.set_facecolor((0, 0, 0, 0))


def contour_plot(function, mi_x, ma_x, mi_y=None, ma_y=None,
                 contours=100,
                 is_torch_function=True,
                 grid_points=1000,
                 ax=None,
                 colorbar=False,
                 transform=None
                 ):
    """
    plot a contour plot of a R^2 \to R function
    :param function: the function to be plotted
    :param mi_x: minimal value for x
    :param ma_x: maximal value for x
    :param mi_y: minimal value for y, default: mi_x
    :param ma_y: maximal value for y, default: ma_x
    :param contours: number of contours to be plotted (default: 100)
    :param is_torch_function: whether the function works (only) on torch tensors
        if false, data is fed to it as numpy arrays
    :param grid_points: number of grid points per axis used for plotting (default: 1000)
    :param ax: axis object on which to plot this (default: just plot to plt)
    :param colorbar: whether to plot a colorbar
    :return: None
    """
    grid, gx, gy = get_grid(mi_x, ma_x, grid_points, mi_y, ma_y)

    if transform is not None:
        grid = transform(grid)

    if is_torch_function:
        function = torch_function_to_numpy(function)

    values_at_grid = function(grid)
    if values_at_grid.shape[-1] == 1:
        values_at_grid = values_at_grid.squeeze(-1)
    if ax is None:
        ax = plt.axes()
    contour = ax.contour(gx, gy, values_at_grid, contours, cmap='jet')
    if colorbar:
        plt.colorbar(contour, ax=ax)


def plot_vectorfield(function, mi_x, ma_x, mi_y=None, ma_y=None, is_torch_function=True, grid_points=50, ax=None):
    """
    plot a R^2 -> R^2 vector field
    :param function: the vector field to be plotted
    :param mi_x: minimal value for x
    :param ma_x: maximal value for x
    :param mi_y: minimal value for y, default: mi_x
    :param ma_y: maximal value for y, default: ma_x
    :param is_torch_function: whether the function works (only) on torch tensors
        if false, data is fed to it as numpy arrays
    :param grid_points: number of grid points per axis at which vectors are drawn (default: 50)
    :param ax: axis object on which to plot this (default: just plot to plt)
    :return: None
    """
    grid, gx, gy = get_grid(mi_x, ma_x, grid_points, mi_y, ma_y)
    if is_torch_function:
        function = torch_function_to_numpy(function)

    values_at_grid = function(grid)
    norms_at_grid = np.sum(np.square(values_at_grid), axis=-1)
    norm = plt.Normalize(norms_at_grid.min(), norms_at_grid.max())

    colors = cm.jet(norm(norms_at_grid))
    colors = np.reshape(colors, (-1, 4))

    if ax is None:
        ax = plt.axes()

    ax.quiver(
        gx,
        gy,
        values_at_grid[..., 0],
        values_at_grid[..., 1],
        color=colors
    )


@torch.no_grad()
def plot_flow_lines(vector_field, steps=100, dt=1e-2, pca=None, is_torch_function=True, ax=None, **grid_kwargs):
    grid, _, _ = get_grid(**grid_kwargs)
    grid = grid.reshape(-1, 2)

    plane = pca.inverse_transform(grid) if pca is not None else grid 
    plane_ = torch.from_numpy(plane).to(dtype=torch.float32) if is_torch_function else plane 

    flow_ = torch.empty((steps+1,) + plane_.shape) if is_torch_function else np.empty((steps+1, ) + plane_.shape)
    flow_[0] = plane_
    for index in range(steps):
        flow_[index+1] = flow_[index] + dt*vector_field(flow_[index])
    
    flow = flow_.numpy() if is_torch_function else flow_ 
    if pca is not None:
        hd_shape = flow.shape 
        ld_shape = (steps+1, grid.shape[0], 2)

        flow = pca.transform(flow.reshape(-1, hd_shape[-1])).reshape(ld_shape)

    flow = flow.transpose((1, 0, 2))  # swap time and batch axes

    colors = plt.cm.jet(np.linspace(0, 1, steps+1))
    if ax is None:
        ax = plt.axes()

    for curve in flow:
        for index in range(steps):
            ax.plot(curve[index:index+2, 0], curve[index:index+2, 1], color=colors[index], alpha=.3)


def plot_sequences(*sequences, labels=None):
    seq_len = len(sequences[0])
    if any(len(seq) != seq_len for seq in sequences):
        raise ValueError(f'Sequences should all be of equal length. Lengths received: {[len(seq) for seq in sequences]}')
    ax = plt.axes()
    num_seq = len(sequences)
    width = 1/seq_len
    height = 1/num_seq

    bar_height = .9*height
    space_height = .1*height

    default_color_list = list(mcolors.TABLEAU_COLORS.values())
    color_list = default_color_list[:2] + ['black']

    ax.set_yticks([index*height + .5*height for index in range(num_seq)], labels=labels)
    ax.set_xticks([0., .2, .4, .6, .8, 1.], labels=np.linspace(0., seq_len, 6))

    labels = ['State 0', 'State 1', 'State 2']
    labels_had = {label: False for label in labels}

    for seq_index, sequence in enumerate(sequences):
        for elem_index, elem in enumerate(sequence):
            label = labels[elem]
            rec = Rectangle(
                (elem_index*width, seq_index*height + .5*space_height),
                width,
                bar_height,
                color=color_list[elem],
                label=f'{"_" if labels_had[label] else ""}{label}'
            )
            ax.add_patch(rec)
            labels_had[label] = True
    ax.legend(loc=0, bbox_to_anchor=(1.0, 1.0))
