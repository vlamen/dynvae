import torch
import numpy as np
import utils.energy_tools as et
from utils.settings import Settings
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


def get_training_data(settings):
    return np.load(settings.data_path)[:-settings.test_size, :settings.max_sequence_length, :]


def get_test_data(settings):
    return np.load(settings.data_path)[-settings.test_size:, :, :]


def get_sampler(
        model,
        settings,
        training_data=None,
        encoded_data=None,
        batch_size=500,
        store_every=40,
        sampler_type='GHMC'):
    """
    function for creating a sampler for the stationary distribution learned by a model
    in practice, we didn't use this but instead used exp_gui.py
    """
    sampler_type = sampler_type.upper()
    if sampler_type not in ('GHMC', 'MAL'):
        raise ValueError(f'Unknown sampler type: {sampler_type}.\nShould be one of "GHMC" and "MAL".')
    if encoded_data is None:
        training_data = training_data if training_data is not None else get_training_data(settings)
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        cpu = torch.device('cpu')
        ts = torch.linspace(
            settings.t0 if 't0' in settings else 0., 
            settings.t1 if 't1' in settings else training_data.shape[1]//10, 
            settings.nt if 'nt' in settings else training_data.shape[1], 
            device=device
            )
        with torch.no_grad():
            results = []
            for index in range(training_data.shape[0] // batch_size):
                batch = torch.from_numpy(
                    training_data[index*batch_size:(index+1)*batch_size]
                ).to(device=device, dtype=torch.float)
                _, _, positions, _ = model(batch, ts, settings.noise_std, dt=settings.dt, return_latent=True)
                results.append(positions.detach()[::store_every].to(cpu))
            encoded_data = torch.cat(results, 0).view(-1, settings.latent_size)
            del positions
            del results 
            model.to(cpu)
    
    empirical_distribution = et.EmpiricalDistribution(encoded_data, add_momentum=(sampler_type == 'GHMC'))
    sampler_class = et.GHMCSampler if sampler_type == 'GHMC' else et.MALSampler
    sampler = sampler_class(
        settings.latent_size,
        model.energy_landscape,
        inverse_temperature=model.beta,
        initial_distribution=empirical_distribution)

    return sampler


def rough_analysis(model, settings, sampler_kwargs, gradient_flow_kwargs, num_samples=1000, k_means_k=3):
    """
    outline of how to do the first stages of the analysis of a model.
    In practice, we used exp_gui.py instead of this function
    """

    # first stage: we get a sampler for the model, sample a bunch of points from it, and apply gradient flow to find the local minima
    sampler = get_sampler(model, settings, **sampler_kwargs)

    samples = sampler(num_samples)
    post_flow = et.gradient_flow(model.energy_landscape, samples, **gradient_flow_kwargs)

    with torch.no_grad():
        samples = samples.numpy()
        post_flow = post_flow.numpy()
    
    # second stage: we find the pca-plane by applying pca to the samples post-flow (so we get the plane through the hopefully 3 minima)
    pca = PCA(n_components=2)
    pca.fit(post_flow)

    samples_embedded = pca.transform(samples)
    post_flow_embedded = pca.transform(post_flow)

    # third stage: we apply clustering to the post flow embedded samples to find out what percentage of samples ended-up in what local minimum
    k_means = KMeans(k_means_k).fit(post_flow_embedded)
    labels = k_means.labels_

    # get percentages of samples in each minimum
    graph_distribution = (labels[:, None] == np.array(list(range(k_means_k)))).mean(axis=0)

    return sampler, pca, k_means, samples, post_flow, samples_embedded, post_flow_embedded, labels, graph_distribution


def get_hessian(model, locations):
    with torch.set_grad_enabled(True):
        if not locations.requires_grad:
            locations.requires_grad_()
        size = locations.shape[-1]

        eye = torch.eye(size)[:, None, :]
        output = []
        grad = model.energy_gradient(locations)  # (batch, size)

        for i in range(size):
            grad_i_sum = (eye[i]*grad).sum()  # scalar
            output.append(torch.autograd.grad(grad_i_sum, locations, retain_graph=True)[0].detach())

        return torch.stack(output, dim=1)


def get_approximations_nld(model, locations):
    """
    To get the zeroth and second order approximations of the distribution over the minima
    """
    locations_ = torch.from_numpy(locations).to(dtype=torch.float32)
    energy_at_minima = model.energy_landscape(locations_)
    # since only the relative heights matter, let's subtract the minimum value for numeric stability

    energy_at_minima = energy_at_minima - torch.min(energy_at_minima)

    exp_energy = torch.exp(-model.beta*energy_at_minima)
    if exp_energy.shape[-1] == 1:
        exp_energy = torch.squeeze(exp_energy, -1)  # (num_minima)

    first_order_approximation = exp_energy / torch.sum(exp_energy)

    hessian_at_minima = get_hessian(model, locations_)

    pi = torch.tensor(np.pi, dtype=torch.float32)

    dim = locations.shape[-1]

    Ci = model.beta * torch.inverse(hessian_at_minima)  # (num_minima, dim, dim)

    rho_i_x_i_inv = torch.sqrt(torch.pow(2*pi, dim)*torch.det(Ci))  # (num_minima)

    second_order_score = exp_energy * rho_i_x_i_inv

    second_order_approximation = second_order_score/torch.sum(second_order_score)

    return first_order_approximation.detach().numpy(), second_order_approximation.detach().numpy()


def invert_permutation(perm):
    inv = np.empty_like(perm)
    inv[perm] = np.arange(perm.size)
    return inv


def find_permutation(y_pred, y):
    """
    Generates the permutation of [0, 1, 2] which, if applied to y_pred, gives the best match with y.

    :param y_pred: the array of predictions
    :param y: the array of labels
    :return: permutation, number of matches
    """
    # since the system always starts in state 0, lets first find the most common first value
    first_values = y_pred[:, 0]
    occurrences = (first_values[:, None] == np.arange(3, dtype=y_pred.dtype)[None, :]).sum(axis=0)
    most_occurring = np.argmax(occurrences)

    # this really only leaves us with two options
    inv_perm_1 = np.empty(3, np.int32)
    inv_perm_2 = np.empty(3, np.int32)
    inv_perm_1[0] = inv_perm_2[0] = most_occurring
    a, b = tuple(i for i in range(3) if i != most_occurring)
    inv_perm_1[1] = inv_perm_2[2] = a
    inv_perm_2[1] = inv_perm_1[2] = b

    # now let's see which one is the best
    score_1 = (inv_perm_1[y] == y_pred).sum()
    score_2 = (inv_perm_2[y] == y_pred).sum()

    if score_1 >= score_2:
        return invert_permutation(inv_perm_1), score_1
    else:
        return invert_permutation(inv_perm_2), score_2


def test_seq_class(model, settings, k_means, data, labels):
    """
    function for finding the performance of the model as a sequence classifier
    """
    results = Settings()

    overdamped = settings.overdamped
    noise_std = settings.noise_std
    dt = settings.dt
    latent_size = settings.latent_size

    with torch.no_grad():
        # encode the data using the model
        # apply gradient flow to encoding
        # classify post_flow
        # find optimal permutation

        ts = torch.linspace(
            0.,
            data.shape[1] // 10,
            data.shape[1],
        )

        if not overdamped:
            _, _, positions, _ = model(data, ts, noise_std, dt=dt, return_latent=True)
        else:
            _, _, positions = model(data, ts, noise_std, dt=dt, return_latent=True)

        results.post_flow = et.gradient_flow(model.energy_landscape, positions,
                                             max_steps=1000).detach().numpy()

        results.labeled_raw = k_means.predict(results.post_flow.reshape((-1, latent_size))).reshape(data.shape[:-1])

        results.encoding = positions.detach().numpy()

        perm, score = find_permutation(results.labeled_raw, labels)

        results.perm = perm
        results.accuracy = score / labels.shape[-1]

        results.labeled = perm[results.labeled_raw]

        return results
