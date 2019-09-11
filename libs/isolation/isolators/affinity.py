# coding: UTF-8

import logging
import random
from typing import Optional, Set, Dict, Tuple

#from ..policies.base import IsolationPolicy

from .base import Isolator
from ...metric_container.basic_metric import MetricDiff
from ...workload import Workload
from ...utils.machine_type import MachineChecker, NodeType


class AffinityIsolator(Isolator):
    def __init__(self, latency_critical_wls: Set[Workload], best_effort_wls: Set[Workload]) -> None:
        super().__init__(latency_critical_wls, best_effort_wls)

        self._cur_steps: Dict[Workload, Tuple[int]] = dict()
        for lc_wl in latency_critical_wls:
            self._cur_steps[lc_wl] = lc_wl.orig_bound_cores
        for be_wl in best_effort_wls:
            self._cur_steps[be_wl] = be_wl.orig_bound_cores

        # FIXME: hard-coded for Jetson TX2
        self._node_type = MachineChecker.get_node_type()
        if self._node_type is NodeType.IntegratedGPU:
            # FIXME: hard-coded part (core range)
            self._all_cores = tuple([0, 3, 4, 5])
            self._MIN_CORES = 1
        elif self._node_type is NodeType.DiscreteGPU:
            # FIXME: hard-coded part (core range)
            self._all_cores = tuple(range(0, 8, 1))
            self._MIN_CORES = 1
        self._available_cores: Optional[Tuple[int]] = Isolator.available_cores()
        self._chosen_alloc: Optional[int] = None
        self._chosen_dealloc: Optional[int] = None
        self._cur_alloc: Optional[Tuple[int]] = None
        self._cur_dealloc: Optional[Tuple[int]] = None

        self._stored_config: Optional[Dict[Workload, Tuple[int]]] = None

    def _get_metric_type_from(self, metric_diff: MetricDiff) -> float:
        return metric_diff.instruction_ps - metric_diff.diff_slack

    def strengthen(self) -> 'AffinityIsolator':
        # FIXME: AffinityIsolator should contain each workload's core affinity
        # TODO: This function (randomly) allocates more cores to workload.
        wl = self.perf_target_wl
        self.get_available_cores()
        if len(self._available_cores) > 0:
            self._chosen_alloc = random.choice(self._available_cores)
            self._cur_alloc = tuple(self._cur_steps[wl]+(self._chosen_alloc,))
        else:
            self._chosen_alloc = None
            self._cur_alloc = None
        return self

    @property
    def is_max_level(self) -> bool:
        # Check available cores and excess cpu flag (either one on both conditions should be met)
        logger = logging.getLogger(__name__)
        logger.info(f'[is_max_level] self.alloc_target_wl: {self.alloc_target_wl}')

        if self.alloc_target_wl is None:
            return False
        self.get_available_cores()
        return len(self._available_cores) <= 0 or self.alloc_target_wl.excess_cpu_flag is True

    @property
    def is_min_level(self) -> bool:
        logger = logging.getLogger(__name__)
        logger.info(f'[is_min_level] self.dealloc_target_wl: {self.dealloc_target_wl}')
        if self.dealloc_target_wl is None:
            return False
        else:
            return len(self.dealloc_target_wl.bound_cores) - 1 < self._MIN_CORES

    def weaken(self) -> 'AffinityIsolator':
        # TODO: This function (randomly) deallocates cores from workload.
        wl = self.perf_target_wl
        if len(wl.bound_cores) > 1:
            self._chosen_dealloc = random.choice(wl.bound_cores)
            self._cur_dealloc = tuple(filter(lambda x: x is not self._chosen_dealloc, wl.bound_cores))
        else:
            self._chosen_dealloc = None
            self._cur_dealloc = None
        return self

    def enforce(self) -> None:
        logger = logging.getLogger(__name__)
        #logger.info(f'affinity of foreground is {self._latency_critical_wls.orig_bound_cores[0]}-{self._cur_step}')

        if self._cur_alloc is not None and self.alloc_target_wl is not None:
            self.alloc_target_wl.bound_cores = self._cur_alloc
            self._cur_steps[self.alloc_target_wl] = self._cur_alloc
            self._update_other_values("alloc")
            logger.info(f'affinity of {self.perf_target_wl.name}-{self.perf_target_wl.pid} is {self._cur_alloc}')
        elif self._cur_dealloc is not None and self.dealloc_target_wl is not None:
            self.dealloc_target_wl.bound_cores = self._cur_dealloc
            self._cur_steps[self.dealloc_target_wl] = self._cur_dealloc
            self._update_other_values("dealloc")
            logger.info(f'affinity of {self.perf_target_wl.name}-{self.perf_target_wl.pid} is {self._cur_dealloc}')

    def reset(self) -> None:
        for wl in self._all_wls:
            if wl.is_running:
                wl.bound_cores = wl.orig_bound_cores

    def store_cur_config(self) -> None:
        self._stored_config = self._cur_steps

    def load_cur_config(self) -> None:
        super().load_cur_config()

        self._cur_steps = self._stored_config
        self._stored_config = None

    def _update_other_values(self, action: str) -> None:
        if action is "alloc":
            self._available_cores = tuple(filter(lambda x: x is not self._chosen_alloc, self._available_cores))
            self.update_available_cores()
            self._cur_alloc = None
            self._chosen_alloc = None
        elif action is "dealloc":
            self._available_cores = tuple(self._available_cores + self._chosen_dealloc)
            self.update_available_cores()
            self._cur_dealloc = None
            self._chosen_dealloc = None

    def get_available_cores(self) -> None:
        self._available_cores = Isolator.available_cores()

    def update_available_cores(self) -> None:
        Isolator.set_available_cores(self._available_cores)
