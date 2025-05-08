#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Infer discrete spikes from probabilities using GPU (CuPy).
"""

import cupy as cp
import numpy as np
import os
from copy import deepcopy
from cupyx.scipy.ndimage import gaussian_filter
import scipy.ndimage as cpu_ndimage  # For unsupported ops


from . import config  # Keep as-is if this is a relative import

# ----- HELPER FUNCTIONS -----

def fill_up_APs(prob_density, smoothingX, nb_spikes, spike_locs):
    approximation = cp.zeros_like(prob_density)
    for spike in spike_locs:
        approximation[spike] += 1
    approximation = gaussian_filter(approximation.astype(cp.float32), sigma=smoothingX)

    counter = 0
    while cp.sum(approximation) < nb_spikes and counter < nb_spikes * 20:
        if counter % cp.ceil(nb_spikes / 10) == 0:
            norm_cum_distribution = cp.cumsum(cp.exp(prob_density - approximation) - 1)
            norm_cum_distribution /= cp.max(norm_cum_distribution)

        spike_location = cp.argmin(cp.abs(norm_cum_distribution - cp.random.uniform()))

        approximation_temp = approximation.copy()
        this_spike = cp.zeros_like(prob_density)
        this_spike[spike_location] = 1
        this_spike = gaussian_filter(this_spike.astype(cp.float32), sigma=smoothingX)
        approximation += this_spike

        error_change = cp.sum(cp.abs(prob_density - approximation)) - cp.sum(cp.abs(prob_density - approximation_temp))

        if error_change <= 0:
            spike_locs.append(int(spike_location.get()))
        else:
            approximation = approximation_temp.copy()

        counter += 1

    return spike_locs, approximation, counter


def divide_and_conquer(prob_density, smoothingX):
    # GPU thresholding
    support_gpu = prob_density > 0.03 / smoothingX

    # Move to CPU
    support_cpu = cp.asnumpy(support_gpu)

    # CPU binary dilation + labeling
    structure = np.ones((int(round(smoothingX * 4)),), dtype=bool)
    dilated = cpu_ndimage.binary_dilation(support_cpu, structure=structure)
    labeled, _ = cpu_ndimage.label(dilated)
    return cpu_ndimage.find_objects(labeled)


def systematic_exploration(prob_density, smoothingX, nb_spikes, spike_locs, approximation):
    T = len(approximation)
    spike_reservoir = cp.zeros((T, T), dtype=cp.float32)
    for t in range(T):
        spike_reservoir[t, t] = 1
        spike_reservoir[t] = gaussian_filter(spike_reservoir[t], sigma=smoothingX)

    error = cp.zeros(T, dtype=cp.float32)
    for i, spike in enumerate(spike_locs):
        for t in range(T):
            approx_suggestion = approximation + spike_reservoir[t] - spike_reservoir[spike]
            error[t] = cp.sum(cp.abs(prob_density - approx_suggestion))

        ix = cp.argmin(error).item()
        spike_locs[i] = ix

        approximation = cp.zeros_like(prob_density)
        for s in spike_locs:
            approximation[s] += 1
        approximation = gaussian_filter(approximation.astype(cp.float32), sigma=smoothingX)

    return spike_locs, approximation


def prune_APs(prob_density, smoothing, nb_spikes, spike_locs, approximation):
    for i, spike in enumerate(spike_locs):
        spike_neg = cp.zeros_like(prob_density)
        spike_neg[spike] = 1
        spike_neg = gaussian_filter(spike_neg.astype(cp.float32), sigma=smoothing)
        approx_temp = approximation - spike_neg

        error_change = cp.sum(cp.abs(prob_density - approx_temp)) - cp.sum(cp.abs(prob_density - approximation))

        if error_change < 0:
            spike_locs[i] = -1
            approximation = approx_temp

    spike_locs = [x for x in spike_locs if x >= 0]
    return spike_locs, approximation


# ----- PER-NEURON FUNCTION (ON GPU) -----

def _process_neuron(neuron, spike_rates, smoothing, sampling_rate, verbosity):
    prob_density = cp.asarray(spike_rates[neuron, :])
    spike_locs_all = []

    nnan_indices = ~cp.isnan(prob_density)
    offset = int(cp.argmax(nnan_indices == True).item()) - 1

    if cp.sum(nnan_indices) > 0:
        prob_density = prob_density[nnan_indices]
        vector_of_indices = cp.arange(len(prob_density))
        support_slices = divide_and_conquer(prob_density, smoothing * sampling_rate)
        approximation = cp.zeros_like(prob_density)

        for s in support_slices:
            spike_locs = []
            nb_spikes = float(cp.sum(prob_density[s]))

            spike_locs, approximation[s], _ = fill_up_APs(prob_density[s], smoothing * sampling_rate, nb_spikes, spike_locs)
            spike_locs, approximation[s] = systematic_exploration(prob_density[s], smoothing * sampling_rate, nb_spikes, spike_locs, approximation[s])

            for _ in range(5):
                spike_locs, approximation[s] = prune_APs(prob_density[s], smoothing * sampling_rate, nb_spikes, spike_locs, approximation[s])
                nb_spikes = float(cp.sum(prob_density[s]) - cp.sum(approximation[s]))
                spike_locs, approximation[s], _ = fill_up_APs(prob_density[s], smoothing * sampling_rate, nb_spikes, spike_locs)

            spike_locs, approximation[s] = systematic_exploration(prob_density[s], smoothing * sampling_rate, nb_spikes, spike_locs, approximation[s])

            temporal_offset = int(vector_of_indices[s][0].item())
            new_spikes = [s + temporal_offset for s in spike_locs]
            spike_locs_all.extend(new_spikes)

        full_approx = cp.nan * cp.ones(spike_rates.shape[1])
        full_approx[nnan_indices] = approximation
    else:
        full_approx = cp.nan * cp.ones(spike_rates.shape[1])

    return (neuron, spike_locs_all, full_approx.get())


# ----- MAIN FUNCTION -----

def infer_discrete_spikes(spike_rates, model_name, model_folder='Pretrained_models', verbosity=1):
    model_path = os.path.join(model_folder, model_name)
    cfg_file = os.path.join(model_path, 'config.yaml')

    if not os.path.isfile(cfg_file):
        raise FileNotFoundError(f'Config file missing: {os.path.abspath(cfg_file)}')

    cfg = config.read_config(cfg_file)
    sampling_rate = cfg['sampling_rate']
    smoothing = cfg['smoothing']

    num_neurons = spike_rates.shape[0]
    approximations_all = np.zeros_like(spike_rates) * np.nan
    spikes_all = [None] * num_neurons

    if verbosity:
        print(f"Starting GPU inference on {num_neurons} neurons")

    for neuron in range(num_neurons):
        _, spikes, approximation = _process_neuron(neuron, spike_rates, smoothing, sampling_rate, verbosity)
        spikes_all[neuron] = spikes
        approximations_all[neuron, :] = approximation

    return approximations_all, spikes_all