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
from heracles_func import HeraclesFunc

import libs
from libs.isolation import NextStep
from libs.isolation.isolators import Isolator
from libs.isolation.policies import XeonPolicy, XeonWViolationPolicy, IsolationPolicy
# from libs.isolation.swapper import SwapIsolator
from pending_queue import PendingQueue
from polling_thread import PollingThread
from libs.utils.hyphen import convert_to_set
from libs.utils.cgroup.cpuset import CpuSet


MIN_PYTHON = (3, 6)


class Controller:
    def __init__(self, metric_buf_size: int, binding_cores: Set[int]) -> None:
        self._pending_queue: PendingQueue = PendingQueue(XeonWViolationPolicy)

        self._interval: float = 0.1  # scheduling interval (sec)
        self._profile_interval: float = 1.0  # check interval for phase change (sec)
        self._solorun_interval: float = 2.0  # the FG's solorun profiling interval (sec)
        self._solorun_count: Dict[IsolationPolicy, Optional[int]] = dict()

        self._switching_interval: float = 0.4   # FIXME: hard-coded
        self._total_profile_time: Optional[float] = None

        self._isolation_groups: Dict[IsolationPolicy, int] = dict()

        self._polling_thread = PollingThread(metric_buf_size, self._pending_queue)

        self._cpuset = CpuSet('EdgeIso')
        self._binding_cores = binding_cores
        #self._isolator_changed = False
        #self._cpuset.create_group()
        #self._cpuset.assign_cpus(binding_cores)
        self._heracles = HeraclesFunc(tail_latency=0.0,
                                      load=0.5,
                                      slo_target=10.0)
        # Swapper init
        # self._swapper: SwapIsolator = SwapIsolator(self._isolation_groups)

    def _isolate_workloads(self) -> None:
        logger = logging.getLogger(__name__)

        for group, iteration_num in self._isolation_groups.items():
            logger.critical('')
            logger.critical(f'***************isolation of {group.name} #{iteration_num} ({group.cur_isolator})***************')
            try:
                logger.info(f'[_isolate_workloads] int(self._profile_interval/self._interval): '
                            f'{int(self._profile_interval / self._interval)}')





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
            logger.info(f'{pending_group} is created')

            self._isolation_groups[pending_group] = 0

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

        self._polling_thread.start()

        logger = logging.getLogger(__name__)
        logger.info('starting isolation loop')

        while True:
            self._remove_ended_groups()
            self._register_pending_workloads()

            time.sleep(self._interval)
            self._isolate_workloads()   ## Heracles High-level controller


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
