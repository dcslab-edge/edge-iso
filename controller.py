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

import libs
from libs.isolation import NextStep
from libs.isolation.isolators import Isolator
from libs.isolation.policies import EdgePolicy, EdgeWViolationPolicy, IsolationPolicy
# from libs.isolation.swapper import SwapIsolator
from pending_queue import PendingQueue
from polling_thread import PollingThread
from libs.utils.hyphen import convert_to_set
from libs.utils.cgroup.cpuset import CpuSet


MIN_PYTHON = (3, 6)


class Controller:
    def __init__(self, metric_buf_size: int, binding_cores: Set[int]) -> None:
        self._pending_queue: PendingQueue = PendingQueue(EdgeWViolationPolicy)

        self._interval: float = 0.2  # scheduling interval (sec)
        self._profile_interval: float = 1.0  # check interval for phase change (sec)
        self._solorun_interval: float = 2.0  # the FG's solorun profiling interval (sec)
        self._solorun_count: Dict[IsolationPolicy, Optional[int]] = dict()

        self._switching_interval: float = 1.0   # FIXME: hard-coded
        self._total_profile_time: Optional[float] = None

        self._isolation_groups: Dict[IsolationPolicy, int] = dict()

        self._polling_thread = PollingThread(metric_buf_size, self._pending_queue)

        self._cpuset = CpuSet('EdgeIso')
        self._binding_cores = binding_cores
        #self._cpuset.create_group()
        #self._cpuset.assign_cpus(binding_cores)

        # Swapper init
        # self._swapper: SwapIsolator = SwapIsolator(self._isolation_groups)

    def _isolate_workloads(self) -> None:
        logger = logging.getLogger(__name__)

        for group, iteration_num in self._isolation_groups.items():
            logger.info('')
            logger.info(f'***************isolation of {group.name} #{iteration_num} ({group.cur_isolator})***************')
            #print(group.in_solorun_profiling)
            try:
                #print("============val debug=============")
                #if group.in_solorun_profiling:
                    #print(self._solorun_count[group])
                #print(int(self._solorun_interval / self._interval))

                #print(group.profile_needed())
                if group.in_solorun_profiling:
                    # Stop condition
                    #if iteration_num - self._solorun_count[group] >= int(self._solorun_interval / self._interval):
                    # if iteration_num - self._solorun_count[group] >= int(self._total_profile_time / self._interval):
                    #     logger.info('Try to choose a solorun profiling target...')
                    #     group.set_next_solorun_target()
                    #
                    #     if group.curr_profile_target is None:
                    #         logger.info('There is no solorun target leftover')
                    #         logger.info('Finished profiling and stopping solorun profiling...')
                    #         group.stop_solorun_profiling()
                    #         del self._solorun_count[group]
                    #     elif group.curr_profile_target is not None:
                    #         logger.info(f'The chosen profiling target is '
                    #                     f'{group.curr_profile_target.name}-{group.curr_profile_target.pid}')
                    #         group.switching_profile_target()
                    #
                    #     logger.info('skipping isolation... because corun data isn\'t collected yet')
                    # Not stop condition 1 (Not all LC tasks are profiled)
                    # FIXME: Below code needs Testing! (duration / timing of switching)
                    if (iteration_num - self._solorun_count[group]) \
                            % int(self._switching_interval / self._interval) == 0:
                        logger.info('Try to choose a solorun profiling target...')
                        group.set_next_solorun_target()

                        # FIXME: This assumes all profile target workloads have the same profile time
                        if iteration_num - self._solorun_count[group] >= int(self._solorun_interval / self._interval):
                            group._curr_profile_target = None
                        if group.curr_profile_target is not None:
                            logger.info(f'The chosen profiling target is '
                                        f'{group.curr_profile_target.name}-{group.curr_profile_target.pid}')
                            group.switching_profile_target()
                        elif group.curr_profile_target is None:
                            logger.info('There is no solorun target leftover...')
                            logger.info('Finished profiling and stopping solorun profiling...')
                            group.stop_solorun_profiling()
                            del self._solorun_count[group]

                        logger.info('skipping isolation... because corun data isn\'t collected yet')
                    # Not stop condition 2 (Ongoing profile stage)
                    else:
                        logger.info(f'skipping isolation because of solorun profiling for'
                                    f' {group.curr_profile_target}...')

                    continue

                # TODO: first expression can lead low reactivity
                elif iteration_num % int(self._profile_interval / self._interval) == 0 and group.profile_needed():
                    logger.info('Starting solorun profiling...')

                    # check and choose workloads to be profiled
                    # and also set the interval for invoking switching_profile_target
                    self._total_profile_time = group.check_profile_target(self._profile_interval)
                    group.start_solorun_profiling()
                    self._solorun_count[group] = iteration_num
                    group.set_idle_isolator()
                    logger.info('skipping isolation because of solorun profiling...')
                    continue

                if group.new_isolator_needed:
                    #print(group)
                    # workloads and isolator are selected in the below code
                    group.choose_next_isolator()

                cur_isolator: Isolator = group.cur_isolator

                # FIXME: decide_next_step belongs to isolator
                # `calc_metric_diff()` is invoked in the below code to determine `next_step`.
                decided_next_step: NextStep = cur_isolator.decide_next_step()
                logger.info(f'Monitoring Result : {decided_next_step.name}')

                if decided_next_step is NextStep.STRENGTHEN:
                    cur_isolator.strengthen()
                elif decided_next_step is NextStep.WEAKEN:
                    cur_isolator.weaken()
                elif decided_next_step is NextStep.STOP:
                    group.set_idle_isolator()
                    continue
                elif decided_next_step is NextStep.IDLE:
                    continue
                else:
                    raise NotImplementedError(f'unknown isolation result : {decided_next_step}')

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
            self._isolate_workloads()


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
    controller_logger.setLevel(logging.DEBUG)
    controller_logger.addHandler(stream_handler)
    controller_logger.addHandler(file_handler)

    module_logger = logging.getLogger(libs.__name__)
    module_logger.setLevel(logging.DEBUG)
    module_logger.addHandler(stream_handler)
    module_logger.addHandler(file_handler)

    monitoring_logger = logging.getLogger('monitoring')
    monitoring_logger.setLevel(logging.INFO)
    monitoring_logger.addHandler(stream_handler)
    monitoring_logger.addHandler(file_handler)

    binding_cores: Set[int] = convert_to_set(args.cores)
    controller = Controller(args.buf_size, binding_cores)
    controller.run()


if __name__ == '__main__':
    if sys.version_info < MIN_PYTHON:
        sys.exit('Python {}.{} or later is required.\n'.format(*MIN_PYTHON))

    main()
