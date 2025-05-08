#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Infer discrete spikes from probabilities using multiprocessing.
"""

from scipy.ndimage import gaussian_filter
import scipy.ndimage as ndim
from copy import deepcopy
import numpy as np
import os
import scipy.io as sio
from multiprocessing import Pool, cpu_count
from functools import partial

from . import config  # Keep as-is if this is a relative import

# ----- HELPER FUNCTIONS (unchanged except for indent where necessary) -----

def fill_up_APs(prob_density, smoothingX, nb_spikes, spike_locs):
    approximation = np.zeros(prob_density.shape)
    for spike in spike_locs:
        approximation[spike] += 1
    approximation = gaussian_filter(approximation.astype(float), sigma=smoothingX)

    counter = 0
    while np.sum(approximation) < nb_spikes and counter < nb_spikes * 20:
        if np.mod(counter, np.ceil(nb_spikes / 10)) == 0:
            norm_cum_distribution = np.cumsum(np.exp(prob_density - approximation) - 1)
            norm_cum_distribution /= np.max(norm_cum_distribution)

        spike_location = np.argmin(np.abs(norm_cum_distribution - np.random.uniform()))

        approximation_temp = deepcopy(approximation)
        this_spike = np.zeros(prob_density.shape)
        this_spike[spike_location] = 1
        this_spike = gaussian_filter(this_spike.astype(float), sigma=smoothingX)
        approximation += this_spike

        error_change = np.sum(np.abs(prob_density - approximation)) - np.sum(np.abs(prob_density - approximation_temp))

        if error_change <= 0:
            spike_locs.append(spike_location)
        else:
            approximation = deepcopy(approximation_temp)

        counter += 1

    return spike_locs, approximation, counter


def divide_and_conquer(prob_density, smoothingX):
    support = prob_density > 0.03 / (smoothingX)
    support = ndim.binary_dilation(support, np.ones((round(smoothingX * 4),)))
    segmentation = ndim.label(support)
    support_slices = ndim.find_objects(segmentation[0])
    return support_slices


def systematic_exploration(prob_density, smoothingX, nb_spikes, spike_locs, approximation):
    spike_reservoir = np.zeros((len(approximation), len(approximation)))
    for timepoint in range(len(approximation)):
        spike_reservoir[timepoint, timepoint] = 1
        spike_reservoir[timepoint, :] = gaussian_filter(spike_reservoir[timepoint, :].astype(float), sigma=smoothingX)

    error = np.zeros(approximation.shape)
    for spike_index, spike in enumerate(spike_locs):
        for timepoint in range(len(approximation)):
            approximation_suggestion = approximation + spike_reservoir[timepoint] - spike_reservoir[spike]
            error[timepoint] = np.sum(np.abs(prob_density - approximation_suggestion))

        ix = np.argmin(error)
        spike_locs[spike_index] = ix

        approximation = np.zeros(prob_density.shape)
        for spike in spike_locs:
            approximation[spike] += 1
        approximation = gaussian_filter(approximation.astype(float), sigma=smoothingX)

    return spike_locs, approximation


def prune_APs(prob_density, smoothing, nb_spikes, spike_locs, approximation):
    for spike_ix, spike1 in enumerate(spike_locs):
        spike_neg = np.zeros(prob_density.shape)
        spike_neg[spike1] = 1
        spike_neg = gaussian_filter(spike_neg.astype(float), sigma=smoothing)
        approximation_temp = approximation - spike_neg

        error_change = np.sum(np.abs(prob_density - approximation_temp)) - np.sum(np.abs(prob_density - approximation))

        if error_change < 0:
            spike_locs[spike_ix] = -1
            approximation -= spike_neg

    spike_locs = [x for x in spike_locs if x >= 0]
    return spike_locs, approximation


# ----- PER-NEURON WORKER FUNCTION -----

def _process_neuron(neuron, spike_rates, smoothing, sampling_rate, verbosity):
    prob_density = spike_rates[neuron, :]
    spike_locs_all = []

    nnan_indices = ~np.isnan(prob_density)
    offset = np.argmax(nnan_indices == True) - 1

    approximation = np.zeros(prob_density.shape)
    if np.sum(nnan_indices) > 0:
        prob_density = prob_density[nnan_indices]
        vector_of_indices = np.arange(0, len(prob_density))
        support_slices = divide_and_conquer(prob_density, smoothing * sampling_rate)
        approximation = np.zeros(prob_density.shape)

        for k in range(len(support_slices)):
            spike_locs = []
            nb_spikes = np.sum(prob_density[support_slices[k]])

            spike_locs, approximation[support_slices[k]], _ = fill_up_APs(
                prob_density[support_slices[k]], smoothing * sampling_rate, nb_spikes, spike_locs
            )

            spike_locs, approximation[support_slices[k]] = systematic_exploration(
                prob_density[support_slices[k]], smoothing * sampling_rate, nb_spikes, spike_locs, approximation[support_slices[k]]
            )

            for _ in range(5):
                spike_locs, approximation[support_slices[k]] = prune_APs(
                    prob_density[support_slices[k]], smoothing * sampling_rate, nb_spikes, spike_locs, approximation[support_slices[k]]
                )
                nb_spikes = np.sum(prob_density[support_slices[k]]) - np.sum(approximation[support_slices[k]])
                spike_locs, approximation[support_slices[k]], _ = fill_up_APs(
                    prob_density[support_slices[k]], smoothing * sampling_rate, nb_spikes, spike_locs
                )

            spike_locs, approximation[support_slices[k]] = systematic_exploration(
                prob_density[support_slices[k]], smoothing * sampling_rate, nb_spikes, spike_locs, approximation[support_slices[k]]
            )

            temporal_offset = vector_of_indices[support_slices[k]][0]
            new_spikes = spike_locs + temporal_offset
            spike_locs_all.extend(new_spikes)

        full_approx = np.nan * np.ones(spike_rates.shape[1])
        full_approx[nnan_indices] = approximation
    else:
        full_approx = np.nan * np.ones(spike_rates.shape[1])

    return (neuron, spike_locs_all + offset, full_approx)


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
        print(f"Starting parallel inference with {cpu_count()} cores")

    # Partial function with fixed args
    worker = partial(_process_neuron, spike_rates=spike_rates, smoothing=smoothing,
                     sampling_rate=sampling_rate, verbosity=verbosity)

    with Pool(cpu_count()) as pool:
        results = pool.map(worker, list(range(num_neurons)))

    for neuron_idx, spikes, approximation in results:
        spikes_all[neuron_idx] = spikes
        approximations_all[neuron_idx, :] = approximation

    return approximations_all, spikes_all