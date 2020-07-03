
import logging
import time

from typing import Deque
from threading import Thread
from libs.isolation.policies import IsolationPolicy
from libs.isolation.isolators import Isolator, CacheIsolator, SchedIsolator, CPUFreqThrottleIsolator
from enum import IntEnum
from heracles_func import State
from libs.metric_container.basic_metric import BasicMetric, MetricDiff
from libs.workload import Workload
from libs.utils.resctrl import ResCtrl



class NextStep(IntEnum):
    STRENGTHEN = 1
    WEAKEN = 2
    STOP = 3
    IDLE = 4


class IsolationThread(Thread):

    def __init__(self, isolator: Isolator) -> None:
        super().__init__(daemon=True)
        self._isolator: Isolator = isolator

        self._group = None
        self._heracles = None
        self._bw_derivative = None  # bw_derivative = (cur_bw - prev_bw) / (2-0)
        self._cur_total_mem_bw = 0.0        # local_mem_ps (per second)
        self._prev_total_mem_bw = 0.0       # local_mem_ps (per second)
        self._sub_controller_interval = 2.0

    @property
    def group(self):
        return self._group

    @group.setter
    def group(self, new_group):
        self._group = new_group

    @property
    def heracles(self):
        return self._heracles

    @heracles.setter
    def heracles(self, new_heracles):
        self._heracles = new_heracles

    def measure_memory_bw(self) -> None:
        self._prev_total_mem_bw = self._cur_total_mem_bw
        total_mem_bw_ps = 0
        for lc_wl in self._group.latency_critical_workloads:
            metrics: Deque[BasicMetric] = lc_wl.metrics
            local_mem_ps = metrics[0].local_mem_ps      # cur local_mem_ps of lc_wl
            total_mem_bw_ps += local_mem_ps
        for be_wl in self._group.best_effort_workloads:
            metrics: Deque[BasicMetric] = be_wl.metrics
            local_mem_ps = metrics[0].local_mem_ps      # cur local_mem_ps of lc_wl
            total_mem_bw_ps += local_mem_ps

        self._cur_total_mem_bw = total_mem_bw_ps
        self._bw_derivative = self._cur_total_mem_bw - self._prev_total_mem_bw

    def decide_next_state(self) -> None:
        """
        It decides next state for each sub-controller
        :return:
        """
        logger = logging.getLogger('sub_controller')
        self.measure_memory_bw()
        state = self.heracles.state
        dram_limit = float(MetricDiff._MAX_MEM_BANDWIDTH_PS * 0.9)
        logger.critical(f'[decide_next_state] self._cur_total_mem_bw: {self._cur_total_mem_bw}')
        logger.critical(f'[decide_next_state] dram_limit: {dram_limit}')
        if self._cur_total_mem_bw > dram_limit:
            state = State.STOP_GROWTH
        else:   # state is State.START_GROWTH
            isolator = self._isolator
            if isinstance(isolator, SchedIsolator):
                state = State.GROW_CORES
            elif isinstance(isolator, CacheIsolator):
                state = State.GROW_LLC
            elif isinstance(isolator, CPUFreqThrottleIsolator):
                state = State.GROW_FREQ
        #elif state is State.GROW_LLC:
        #    state = State.GROW_CORES
        #elif state is State.GROW_CORES:
        #    state = State.GROW_LLC
        logger.critical(f'[decide_next_state] state: {state}')

        self.heracles.state = state

    def decide_next_step(self) -> NextStep:
        """
        It decides whether the cur_isolator perform weaken() or strengthen()

        :return:
        """
        logger = logging.getLogger('sub_controller')
        state = self.heracles.state
        isolator = self._isolator

        # NOTE:
        # state : state for best-effort workload's resource growth
        # is_min_level : isolator reaches to the maximum value for stronger configuration (strengthen)
        # is_max_level : isolator reaches to the minimum value of weaker configuration (weaken)

        if state is State.STOP_GROWTH:
            logger.warning(f'[decide_next_step] state is STOP_GROWTH.')
            if isolator.is_max_level:
                next_step = NextStep.STOP
            else:
                next_step = NextStep.STRENGTHEN

        elif state is State.GROW_CORES and isinstance(isolator, SchedIsolator):
            logger.warning(f'[decide_next_step] state is GROW_CORES.')
            if isolator.is_min_level:
                next_step = NextStep.STOP
            else:
                next_step = NextStep.WEAKEN

        elif state is State.GROW_LLC and isinstance(isolator, CacheIsolator):
            logger.warning(f'[decide_next_step] state is GROW_LLC.')
            if isolator.is_min_level:
                next_step = NextStep.STOP
            else:
                next_step = NextStep.WEAKEN

        elif state is State.GROW_FREQ and isinstance(isolator, CPUFreqThrottleIsolator):
            logger.warning(f'[decide_next_step] state is GROW_FREQ.')
            if isolator.is_min_level:
                next_step = NextStep.STOP
            else:
                next_step = NextStep.WEAKEN

        else:
            logger.warning(f'[decide_next_step] state is UNKNOWN. so next_step will be IDLE')
            next_step = NextStep.IDLE

        logger.warning(f'[decide_next_step] state: {state}, next step: {next_step}')
        return next_step

    def setup_isolation(self, next_step: NextStep) -> None:
        logger = logging.getLogger('sub_controller')

        cur_isolator = self._isolator
        state = self.heracles.state
        lc_wls = self._group.latency_critical_workloads.copy()
        be_wls = self._group.best_effort_workloads.copy()

        # For updating keys of cur_steps
        cur_isolator._latency_critical_wls = lc_wls
        cur_isolator._best_effort_wls = be_wls
        """
        for isolator in self._group._isolator_map.values():
            logger.warning(f'[setup_isolation] isolator: {isolator}')
            isolator._latency_critical_wls = lc_wls
            isolator._best_effort_wls = be_wls
            logger.warning(f'[setup_isolation] isolator: {isolator}, isolator._latency_critical_wls: {isolator._latency_critical_wls}')
            logger.warning(f'[setup_isolation] isolator: {isolator}, isolator._best_effort_wls: {isolator._best_effort_wls}')
        """

        # FIXME: Assumption - the number of latency-critical and best-effort workload is "only one" in this setup.
        logger.warning(f'[setup_isolation] lc_wls: {lc_wls}')
        logger.warning(f'[setup_isolation] be_wls: {be_wls}')
        if len(lc_wls) > 0 and len(be_wls) > 0:
            lc_wl = lc_wls.pop()
            be_wl = be_wls.pop()

            if isinstance(cur_isolator, SchedIsolator) or isinstance(cur_isolator, CPUFreqThrottleIsolator):
                logger.warning(f'[setup_isolation] next_step: {next_step}')
                if next_step is NextStep.WEAKEN:
                    self._group.alloc_target_wl = be_wl
                    cur_isolator.alloc_target_wl = be_wl
                    if cur_isolator.is_min_level:
                        self._group.alloc_target_wl = None
                        cur_isolator.alloc_target_wl = None
                    self._group.dealloc_target_wl = None
                    cur_isolator.dealloc_target_wl = None
                elif next_step is NextStep.STRENGTHEN:
                    self._group.alloc_target_wl = None
                    cur_isolator.alloc_target_wl = None
                    self._group.dealloc_target_wl = be_wl
                    cur_isolator.dealloc_target_wl = be_wl
                    if cur_isolator.is_max_level:
                        self._group.dealloc_target_wl = None
                        cur_isolator.dealloc_target_wl = None
            elif isinstance(cur_isolator, CacheIsolator):
                if next_step is NextStep.WEAKEN:
                    self._group.alloc_target_wl = be_wl
                    cur_isolator.alloc_target_wl = be_wl        # CacheIsolator's weaken() for be_wl
                    self._group.dealloc_target_wl = lc_wl
                    cur_isolator.dealloc_target_wl = lc_wl      # CacheIsolator's strengthen() for lc_wl

                elif next_step is NextStep.STRENGTHEN:
                    self._group.alloc_target_wl = lc_wl
                    cur_isolator.alloc_target_wl = lc_wl        # CacheIsolator's weaken() for lc_wl
                    self._group.dealloc_target_wl = be_wl
                    cur_isolator.dealloc_target_wl = be_wl      # CacheIsolator's strengthen() for be_wl

    def run(self) -> None:
        logger = logging.getLogger('sub_controller')
        while True:
            state = self.heracles.state
            state_done = self.heracles.state_done

            logger.critical(f'[isolation_thread:run] state: {state}')
            logger.critical(f'[isolation_thread:run] self._group: {self._group}')
            logger.critical(f'[isolation_thread:run] self._isolator: {self._isolator}')
            logger.critical(f'[isolation_thread:run] isinstance(self._isolator_type, SchedIsolator): {isinstance(self._isolator, SchedIsolator)}')

            cur_isolator = self._isolator
            if state_done is False:
                self.decide_next_state()                # Monitor bw, set state (State.GROW_LLC or State.GROW_CORES)

            next_step = self.decide_next_step()
            self.setup_isolation(next_step)
            logger.warning(f'[isolation_thread:run] next_step: {next_step}, state: {self.heracles.state}')
            if isinstance(cur_isolator, SchedIsolator) or isinstance(cur_isolator, CPUFreqThrottleIsolator):
                if next_step is NextStep.WEAKEN:
                    cur_isolator.weaken()
                    cur_isolator.enforce()
                elif next_step is NextStep.STRENGTHEN:
                    cur_isolator.strengthen()
                    cur_isolator.enforce()

            elif isinstance(cur_isolator, CacheIsolator):
                logger.critical(f'[isolation_thread:run] next_step: {next_step}')
                if next_step is NextStep.WEAKEN:
                    logger.critical(f'[isolation_thread:run] first strengthen()!')
                    cur_isolator.strengthen()
                    cur_isolator.enforce()

                    logger.critical(f'[isolation_thread:run] weaken()!')
                    cur_isolator.weaken()
                    cur_isolator.enforce()

                elif next_step is NextStep.STRENGTHEN:
                    logger.critical(f'[isolation_thread:run] first weaken()!')
                    cur_isolator.weaken()
                    cur_isolator.enforce()

                    logger.critical(f'[isolation_thread:run] strengthen()!')
                    cur_isolator.strengthen()
                    cur_isolator.enforce()

            else:
                # unknown isolator
                print(f'isolator is not chosen. isolator:{cur_isolator}')
            #cur_isolator.enforce()
            time.sleep(self._sub_controller_interval)

