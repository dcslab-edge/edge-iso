#!/usr/bin/env python3
# coding: UTF-8

import argparse
import datetime
import logging
import os
import subprocess
import sys
import time
from typing import Dict, Optional, Set

import psutil
from heracles_func import HeraclesFunc, State

import libs
from libs.isolation import NextStep
from libs.isolation.isolators import Isolator, SchedIsolator, CacheIsolator, CPUFreqThrottleIsolator
from libs.isolation.policies import XeonPolicy, XeonWViolationPolicy, IsolationPolicy
# from libs.isolation.swapper import SwapIsolator
from pending_queue import PendingQueue
from polling_thread import PollingThread
from libs.utils.hyphen import convert_to_set
from libs.utils.cgroup.cpuset import CpuSet

from isolation_thread import IsolationThread

MIN_PYTHON = (3, 6)


class Controller:
    def __init__(self, metric_buf_size: int, binding_cores: Set[int]) -> None:
        self._pending_queue: PendingQueue = PendingQueue(XeonWViolationPolicy)

        self._interval: float = 15.0  # scheduling interval (sec)
        self._profile_interval: float = 1.0  # check interval for phase change (sec)
        self._solorun_interval: float = 2.0  # the FG's solorun profiling interval (sec)
        self._solorun_count: Dict[IsolationPolicy, Optional[int]] = dict()

        self._switching_interval: float = 0.4   # FIXME: hard-coded
        self._total_profile_time: Optional[float] = None

        self._isolation_groups: Dict[IsolationPolicy, int] = dict()

        self._polling_thread = PollingThread(metric_buf_size, self._pending_queue)

        self._isolation_threads = (IsolationThread(SchedIsolator),
                                   IsolationThread(CacheIsolator),
                                   IsolationThread(CPUFreqThrottleIsolator))

        self._cpuset = CpuSet('Heracles')
        self._binding_cores = binding_cores
        #self._isolator_changed = False
        #self._cpuset.create_group()
        #self._cpuset.assign_cpus(binding_cores)
        # FIXME: hard-coded file path for reading latency (heracles)
        lat_file_path = '/home/dcslab/ysnam/benchmarks/tailbench/tailbench/harness/heracles_latency.txt'
        self._heracles = HeraclesFunc(interval=self._interval,        # 15 seconds
                                      tail_latency=0.0,
                                      load=0.0,
                                      slo_target=0.003,               # 0.0025s(2.5ms)*1.2 == 3ms or 0.0045s (4.5ms) * 0.7 = 3.15ms
                                      file_path=lat_file_path)
        # Swapper init
        # self._swapper: SwapIsolator = SwapIsolator(self._isolation_groups)

    def _isolate_workloads(self) -> None:
        logger = logging.getLogger(__name__)
        heracles = self._heracles

        for group, iteration_num in self._isolation_groups.items():
            #heracles.group = group

            logger.critical('')
            logger.critical(f'***************isolation of {group.name} #{iteration_num} ({group.cur_isolator})***************')
            try:
                logger.info(f'[_isolate_workloads] int(self._profile_interval/self._interval): '
                            f'{int(self._profile_interval / self._interval)}')

                heracles.poll_lc_app_latency()      # get Tail latency

                heracles.poll_lc_app_load()         # get QPS / Peak_QPS

                latency = heracles.tail_latency
                load = heracles.load                # currently, QPS
                target = heracles.slo_taget
                slack: float = (target-latency)/target
                logger.critical(f'[_isolate_workloads] slack: {slack}, load: {load}, target: {target}, latency: {latency}')
                if slack < 0:
                    # FIXME: hard-coded for single best-effort workloads
                    HeraclesFunc.disable_be_wls(group.best_effort_workloads)
                    heracles.state = State.STOP_GROWTH
                    logger.critical(f'[_isolate_workloads] slack < 0 case, slack: {slack}, load: {load}, heracles.state: {heracles.state}')
                    # EnterCooldown()
                elif load > 0.85:
                    HeraclesFunc.disable_be_wls(group.best_effort_workloads)
                    heracles.state = State.STOP_GROWTH
                    logger.critical(f'[_isolate_workloads] load > 0.85 case, slack: {slack}, load: {load}, heracles.state: {heracles.state}')
                elif load < 0.8:
                    heracles.enable_be_wls(group.best_effort_workloads)
                    logger.critical(f'[_isolate_workloads] load < 0.8 case, slack: {slack}, load: {load}, heracles.state: {heracles.state}')
                elif slack < 0.1:
                    heracles.disallow_be_growth()
                    logger.critical(f'[_isolate_workloads] slack < 0.1 case, slack: {slack}, load: {load}, heracles.state: {heracles.state}')
                    if slack < 0.05:
                        logger.critical(f'[_isolate_workloads] slack < 0.05 case, slack: {slack}, load: {load}, heracles.state: {heracles.state}')
                        group._cur_isolator = group._isolator_map[SchedIsolator]
                        cur_isolator = group._cur_isolator
                        logger.critical(f'[_isolate_workloads] slack < 0.05 case, cur_isolator: {cur_isolator}, group: {group}')
                        for be_wl in group.best_effort_workloads:
                            group.dealloc_target_wl = be_wl
                            cur_isolator.dealloc_target_wl = be_wl
                            cur_isolator.strengthen()
                            cur_isolator.enforce()

            except (psutil.NoSuchProcess, subprocess.CalledProcessError, ProcessLookupError):
                pass

            finally:
                self._isolation_groups[group] += 1
        """
        if len(tuple(g for g in self._isolation_groups if g.safe_to_swap)) >= 2:
            if self._swapper.swap_is_needed():
                self._swapper.do_swap()
        """

    def _register_pending_workloads(self) -> None:
        """
        This function detects and registers the spawned workloads(threads).
        """
        logger = logging.getLogger(__name__)

        # set pending workloads as active
        # print(f'len of pending queue: {len(self._pending_queue)}')
        while len(self._pending_queue):
            pending_group: IsolationPolicy = self._pending_queue.pop()
            logger.critical(f'{pending_group} is created')

            self._isolation_groups[pending_group] = 0
            # FIXME: assumption: only one group
            self._heracles.group = pending_group
            for sub_con in self._heracles.sub_controllers:
                sub_con.group = pending_group

    def _remove_ended_groups(self) -> None:
        """
        deletes the finished workloads(threads) from the dict.
        """
        logger = logging.getLogger(__name__)

        ended = tuple(filter(lambda g: g.ended, self._isolation_groups))
        """
        for group in ended:
            if group.foreground_workload.is_running:
                ended_workload = group.background_workload
            else:
                ended_workload = group.foreground_workload
            logger.info(f'{group} of {ended_workload.name} is ended')
        """
        #print("ended groups:")
        #print(ended)
        for group in ended:
            # remove from containers
            group.reset()
            del self._isolation_groups[group]
            if group.in_solorun_profiling:
                for bg_wl in group.best_effort_workloads:
                    bg_wl.resume()
                del self._solorun_count[group]

    def run(self) -> None:

        self._cpuset.create_group()
        self._cpuset.assign_cpus(self._binding_cores)

        self._polling_thread.start()            # running in background
        for sub_controller in self._isolation_threads:
            self._heracles.sub_controllers.append(sub_controller)

        logger = logging.getLogger(__name__)
        logger.critical('[controller:run] starting isolation loop')
        first = True
        while True:
            self._remove_ended_groups()
            self._register_pending_workloads()

            time.sleep(self._interval)

            self._isolate_workloads()           # Heracles High-level controller
            if first:
                self._heracles.start_sub_controllers()
                first = False


def main() -> None:
    parser = argparse.ArgumentParser(description='Run workloads that given by parameter.')
    parser.add_argument('-b', '--metric-buf-size', dest='buf_size', default='50', type=int,
                        help='metric buffer size per thread. (default : 50)')
    parser.add_argument('-c', '--binding-cores', dest='cores', default='4,5', type=str,
                        help='binding cores ids where controller runs on. (default: 4,5 where bg tasks run on)')

    os.makedirs('logs', exist_ok=True)

    args = parser.parse_args()

    formatter = logging.Formatter('%(asctime)s [%(levelname)s]: %(message)s')
    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(f'logs/debug_{datetime.datetime.now().isoformat()}.log')
    stream_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    controller_logger = logging.getLogger(__name__)
    controller_logger.setLevel(logging.WARNING)
    controller_logger.addHandler(stream_handler)
    controller_logger.addHandler(file_handler)

    module_logger = logging.getLogger(libs.__name__)
    module_logger.setLevel(logging.WARNING)    # INFO
    module_logger.addHandler(stream_handler)
    module_logger.addHandler(file_handler)

    monitoring_logger = logging.getLogger('monitoring')
    monitoring_logger.setLevel(logging.WARNING)    # INFO
    monitoring_logger.addHandler(stream_handler)
    monitoring_logger.addHandler(file_handler)

    binding_cores: Set[int] = convert_to_set(args.cores)
    controller = Controller(args.buf_size, binding_cores)
    controller.run()


if __name__ == '__main__':
    if sys.version_info < MIN_PYTHON:
        sys.exit('Python {}.{} or later is required.\n'.format(*MIN_PYTHON))

    main()
