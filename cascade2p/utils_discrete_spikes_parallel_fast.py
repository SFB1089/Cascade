#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimized: Infer discrete spikes from probabilities using multiprocessing.
"""

import numpy as np
import os
import scipy.io as sio
from scipy.ndimage import gaussian_filter1d, binary_dilation, label, find_objects
from multiprocessing import Pool, cpu_count
from functools import partial
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from . import config  # Leave as is


# --- Core Functions ---

def fill_up_APs(prob_density, smoothingX, nb_spikes, spike_locs):
    approximation = np.zeros_like(prob_density)
    approximation[spike_locs] += 1
    approximation = gaussian_filter1d(approximation, sigma=smoothingX)

    norm_cum_distribution = None
    counter = 0  # <--- Initialize here
    for counter in range(nb_spikes * 20):
        if approximation.sum() >= nb_spikes:
            break

        if counter % max(1, nb_spikes // 10) == 0:
            delta = np.exp(prob_density - approximation) - 1
            norm_cum_distribution = np.cumsum(delta)
            norm_cum_distribution /= norm_cum_distribution[-1]

        spike_location = np.searchsorted(norm_cum_distribution, np.random.rand())
        spike_location = min(spike_location, len(prob_density) - 1)

        this_spike = np.zeros_like(prob_density)
        this_spike[spike_location] = 1
        new_contrib = gaussian_filter1d(this_spike, sigma=smoothingX)

        new_approx = approximation + new_contrib
        if np.sum(np.abs(prob_density - new_approx)) <= np.sum(np.abs(prob_density - approximation)):
            spike_locs.append(spike_location)
            approximation = new_approx

    return spike_locs, approximation, counter


def divide_and_conquer(prob_density, smoothingX):
    support = prob_density > 0.03 / smoothingX
    support = binary_dilation(support, np.ones(int(round(smoothingX * 4))))
    segmentation, _ = label(support)
    return find_objects(segmentation)


def systematic_exploration(prob_density, smoothingX, nb_spikes, spike_locs, approximation):
    T = len(approximation)
    spike_reservoir = np.eye(T)
    spike_reservoir = gaussian_filter1d(spike_reservoir, sigma=smoothingX, axis=1)

    for i, spike in enumerate(spike_locs):
        errors = np.sum(np.abs(prob_density - (approximation + spike_reservoir - spike_reservoir[spike][:, None].T)), axis=1)
        best_ix = np.argmin(errors)
        spike_locs[i] = best_ix

    new_approx = np.zeros_like(prob_density)
    new_approx[spike_locs] += 1
    new_approx = gaussian_filter1d(new_approx, sigma=smoothingX)
    return spike_locs, new_approx


def prune_APs(prob_density, smoothingX, nb_spikes, spike_locs, approximation):
    new_locs = []
    for spike in spike_locs:
        this_spike = np.zeros_like(prob_density)
        this_spike[spike] = 1
        filtered = gaussian_filter1d(this_spike, sigma=smoothingX)
        approx_temp = approximation - filtered

        if np.sum(np.abs(prob_density - approx_temp)) <= np.sum(np.abs(prob_density - approximation)):
            approximation = approx_temp
        else:
            new_locs.append(spike)

    return new_locs, approximation


# --- Worker Function ---

def _process_neuron(neuron, spike_rates, smoothing, sampling_rate, verbosity):
    prob_density = spike_rates[neuron, :]
    nnan_indices = ~np.isnan(prob_density)
    offset = np.argmax(nnan_indices) - 1

    full_approx = np.full_like(prob_density, np.nan)
    if np.sum(nnan_indices) == 0:
        return (neuron, [], full_approx)

    prob_density = prob_density[nnan_indices]
    vector_indices = np.arange(len(prob_density))
    approximation = np.zeros_like(prob_density)
    spike_locs_all = []

    slices = divide_and_conquer(prob_density, smoothing * sampling_rate)
    for sl in slices:
        segment = prob_density[sl]
        nb_spikes = int(np.round(np.sum(segment)))
        spike_locs = []

        spike_locs, approximation[sl], _ = fill_up_APs(segment, smoothing * sampling_rate, nb_spikes, spike_locs)
        spike_locs, approximation[sl] = systematic_exploration(segment, smoothing * sampling_rate, nb_spikes, spike_locs, approximation[sl])

        for _ in range(5):
            spike_locs, approximation[sl] = prune_APs(segment, smoothing * sampling_rate, nb_spikes, spike_locs, approximation[sl])
            nb_spikes = int(np.round(np.sum(segment) - np.sum(approximation[sl])))
            spike_locs, approximation[sl], _ = fill_up_APs(segment, smoothing * sampling_rate, nb_spikes, spike_locs)

        spike_locs, approximation[sl] = systematic_exploration(segment, smoothing * sampling_rate, nb_spikes, spike_locs, approximation[sl])

        spike_locs = np.array(spike_locs) + vector_indices[sl][0]
        spike_locs_all.extend(spike_locs.tolist())

    full_approx[nnan_indices] = approximation
    return neuron, spike_locs_all + offset, full_approx


# --- Main Entry Point ---

def infer_discrete_spikes(spike_rates, model_name, model_folder='Pretrained_models', verbosity=1):
    cfg_path = os.path.join(model_folder, model_name, 'config.yaml')
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f'Missing config: {cfg_path}')

    cfg = config.read_config(cfg_path)
    sampling_rate = cfg['sampling_rate']
    smoothing = cfg['smoothing']

    num_neurons = spike_rates.shape[0]
    approximations_all = np.full_like(spike_rates, np.nan)
    spikes_all = [None] * num_neurons

    if verbosity:
        print(f"Starting inference with {cpu_count()} cores...")

    worker = partial(_process_neuron, spike_rates=spike_rates,
                     smoothing=smoothing, sampling_rate=sampling_rate,
                     verbosity=verbosity)

    with Pool(cpu_count()) as pool:
        iterator = pool.imap_unordered(worker, range(num_neurons))
        results = list(tqdm(iterator, total=num_neurons, desc="Inferring spikes")) if tqdm and verbosity else list(iterator)

    for neuron_idx, spikes, approx in results:
        spikes_all[neuron_idx] = spikes
        approximations_all[neuron_idx, :] = approx

    return approximations_all, spikes_all