#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import datetime
import logging as log
from math import ceil
import sys
from time import perf_counter

import numpy as np
from openvino.runtime import Core, get_version, AsyncInferQueue
from openvino.runtime.utils.types import get_dtype


def percentile(values, percent):
    return values[ceil(len(values) * percent / 100) - 1]


def fill_tensor_random(tensor):
    dtype = get_dtype(tensor.element_type)
    rand_min, rand_max = (0, 1) if dtype == np.bool else (np.iinfo(np.uint8).min, np.iinfo(np.uint8).max)
    # np.random.uniform excludes high: add 1 to have it generated
    if np.dtype(dtype).kind in ['i', 'u', 'b']:
        rand_max += 1
    rs = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(0)))
    if 0 == tensor.get_size():
        raise RuntimeError("Models with dynamic shapes aren't supported. Input tensors must have specific shapes before inference")
    tensor.data[:] = rs.uniform(rand_min, rand_max, list(tensor.shape)).astype(dtype)


def print_perf_counters(perf_counts_list):
    max_layer_name = 30
    for ni in range(len(perf_counts_list)):
        perf_counts = perf_counts_list[ni]
        total_time = datetime.timedelta()
        total_time_cpu = datetime.timedelta()
        log.info(f"Performance counts for {ni}-th infer request")
        for pi in perf_counts:
            print(f"{pi.node_name[:max_layer_name - 4] + '...' if (len(pi.node_name) >= max_layer_name) else pi.node_name:<30}"
                                                                f"{str(pi.status):<15}"
                                                                f"{'layerType: ' + pi.node_type:<30}"
                                                                f"{'realTime: ' + str(pi.real_time):<20}"
                                                                f"{'cpu: ' +  str(pi.cpu_time):<20}"
                                                                f"{'execType: ' + pi.exec_type:<20}")
            total_time += pi.real_time
            total_time_cpu += pi.cpu_time
        log.info(f'Total time:     {total_time} seconds')
        log.info(f'Total CPU time: {total_time_cpu} seconds\n')


def main():
    log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.INFO, stream=sys.stdout)
    log.info(f"OpenVINO:\n{'': <9}{'API version':.<24} {get_version()}")
    if len(sys.argv) != 2:
        log.info(f'Usage: {sys.argv[0]} <path_to_model>')
        return 1
    # Optimize for throughput. Best throughput can be reached by
    # running multiple ov::InferRequest instances asyncronously
    tput = {'PERFORMANCE_HINT': 'THROUGHPUT'}

    # Uncomment the following line to enable detailed performace counters
    # tput['PERF_COUNT'] = True

    # Create Core and use it to compile a model
    # Pick device by replacing CPU, for example MULTI:CPU(4),GPU(8)
    core = Core()
    compiled_model = core.compile_model(sys.argv[1], 'CPU', tput)
    # AsyncInferQueue creates optimal number of InferRequest instances
    ireqs = AsyncInferQueue(compiled_model)
    # Fill input data for ireqs
    for ireq in ireqs:
        for model_input in compiled_model.inputs:
            fill_tensor_random(ireq.get_tensor(model_input))
    # Warm up
    for _ in ireqs:
        ireqs.start_async()
    for ireq in ireqs:
        ireq.wait()
    # Benchmark for seconds_to_run seconds and at least niter iterations
    seconds_to_run = 15
    init_niter = 12
    niter = int((init_niter + len(ireqs) - 1) / len(ireqs)) * len(ireqs)
    if init_niter != niter:
        log.warning('Number of iterations was aligned by request number '
                    f'from {init_niter} to {niter} using number of requests {len(ireqs)}')
    latencies = []
    in_fly = set()
    start = perf_counter()
    time_point_to_finish = start + seconds_to_run
    while perf_counter() < time_point_to_finish and len(latencies) + len(in_fly) < niter:
        idle_id = ireqs.get_idle_request_id()
        if idle_id in in_fly:
            latencies.append(ireqs[idle_id].latency)
        else:
            in_fly.add(idle_id)
        ireqs.start_async()
    ireqs.wait_all()
    duration = perf_counter() - start
    for infer_request_id in in_fly:
        latencies.append(ireqs[infer_request_id].latency)
    # Report results

    # Uncomment the following lines if performace counters are enabled on top
    # perfs_count_list = []
    # for ireq in ireqs:
    #     perfs_count_list.append(ireq.profiling_info)
    # print_perf_counters(perfs_count_list)

    latencies.sort()
    percent = 50
    percentile_latency_ms = percentile(latencies, percent)
    avg_latency_ms = sum(latencies) / len(latencies)
    min_latency_ms = latencies[0]
    max_latency_ms = latencies[-1]
    fps = len(latencies) / duration
    log.info(f'Count:          {len(latencies)} iterations')
    log.info(f'Duration:       {duration * 1e3:.2f} ms')
    log.info('Latency:')
    if percent == 50:
        log.info(f'    Median:     {percentile_latency_ms:.2f} ms')
    else:
        log.info(f'({percent} percentile):     {percentile_latency_ms:.2f} ms')
    log.info(f'    AVG:        {avg_latency_ms:.2f} ms')
    log.info(f'    MIN:        {min_latency_ms:.2f} ms')
    log.info(f'    MAX:        {max_latency_ms:.2f} ms')
    log.info(f'Throughput: {fps:.2f} FPS')


if __name__ == '__main__':
    main()
