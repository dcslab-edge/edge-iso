# coding: UTF-8

import logging
from typing import Optional, Set, Dict, Iterable

from .base import Isolator
from ...metric_container.basic_metric import MetricDiff
from ...workload import Workload
from ...utils.machine_type import MachineChecker, NodeType
from ...metric_container.basic_metric import BasicMetric
from .. import NextStep


class SchedIsolator(Isolator):
    def __init__(self, foreground_wl: Workload, background_wls: Set[Workload]) -> None:
        super().__init__(foreground_wl, background_wls)

        # FIXME: hard coded (self._MAX_CORES is fixed depending on node type)
        self._bgs_list = list(background_wls)
        self._node_type = MachineChecker.get_node_type()
        if self._node_type == NodeType.IntegratedGPU:
            self._MAX_CORES = 4
        elif self._node_type == NodeType.CPU:
            self._MAX_CORES = 4

        # TODO: Currently all BGs have same number of cores (self._cur_step)
        # TODO: Also, cur_step should indicate the cpuset of background tasks.
        # self._bg_wl = self._bgs_list[0]
        # self._num_of_bg_wls = len(self._bgs_list)
        self._cur_step: Dict[Workload, Iterable[int]] = dict()
        self._cur_action: NextStep = NextStep.IDLE
        for bg_wl in self._bgs_list:
            self._cur_step[bg_wl] = bg_wl.orig_bound_cores
        # self._cur_step: Dict[Workload, int] = [(bg_wl, bg_wl.num_cores) for bg_wl in self._bgs_list]

        self._target_bg: Workload = None        # The target bg for re-assigning bounded cores
        self._max_mem_bg: Workload = None
        self._min_cores_bg: Workload = None
        self._next_assignment = None

        self._stored_config: Optional[int] = None

    @classmethod
    def _get_metric_type_from(cls, metric_diff: MetricDiff) -> float:
        return metric_diff.local_mem_util_ps

    def strengthen(self) -> 'SchedIsolator':
        logger = logging.getLogger(__name__)
        self.update_max_membw_bg()
        logger.info(f'[strengthen] self._target_bg : {self._target_bg}')
        # FIXME: hard coded (All workloads are running on the same contiguous CPU ID)
        if self._target_bg is not None:
            #self._cur_step[self._target_bg] -= 1
            logger.info(f'[strengthen] target_bg\'s bound_cores : {self._target_bg.bound_cores[0]} ~ {self._target_bg.bound_cores[-1]}')
            start_idx = self._target_bg.bound_cores[0]+1
            end_idx = self._target_bg.bound_cores[-1]
            if start_idx is not end_idx:
                self._cur_step[self._target_bg] = range(self._target_bg.bound_cores[0]+1, self._target_bg.bound_cores[-1])
            else:
                self._cur_step[self._target_bg] = [self._target_bg.bound_cores[0]+1,]*2
            self._cur_action = NextStep.STRENGTHEN
        return self

    def weaken(self) -> 'SchedIsolator':
        self.update_min_cores_bg()
        if self._target_bg is not None:
            #self._cur_step[self._target_bg] += 1
            self._cur_step[self._target_bg] = range(self._target_bg.bound_cores[0]-1, self._target_bg.bound_cores[-1])
            start_core_id = self._target_bg.bound_cores[0]-1
            end_core_id = self._target_bg.bound_cores[-1]
            if start_idx is not end_idx:
                self._cur_step[self._target_bg] = range(self._target_bg.bound_cores[0]-1, self._target_bg.bound_cores[-1])
            else:
                self._cur_step[self._target_bg] = [self._target_bg.bound_cores[0]-1,]*2
            self._cur_action = NextStep.WEAKEN
        return self

    @property
    def is_max_level(self) -> bool:
        # At least a process needs one core for its execution
        logger = logging.getLogger(__name__)
        bgs = self._cur_step.keys()
        for bg in bgs:
            logger.info(f'[is_max_level] bg.bound_cores: {bg.bound_cores}')
            # FIXME: We will add 'stop' option for background task. (Currently, we don't stop any bg task!)
            if len(bg.bound_cores) < 2:
                return True
            else:
                return False

        """
        min_cores = min(self._cur_step.values())
        if min_cores < 1:
            return True
        else:
            return False
        """

    @property
    def is_min_level(self) -> bool:
        # FIXME: hard coded
        # At most background processes can not preempt cores of the foreground process
        # FIXME: all bg tasks have the same number of cores (hard-coded)
        fg_cores = self._foreground_wl.num_cores
        for bg, bg_bound_cores in self._cur_step.items():
            bg_cores = len(bg_bound_cores)
            if bg_cores + fg_cores > self._MAX_CORES or self.is_overlapped_assignment():
                return True
            else:
                return False

        """
        bg_cores = sum(self._cur_step.values())
        fg_cores = self._foreground_wl.num_cores
        return (bg_cores + fg_cores) > self._MAX_CORES or self.is_overlapped_assignment()
        """

    def is_overlapped_assignment(self) -> bool:
        fg_cores: Set[int] = self._foreground_wl.cgroup_cpuset.read_cpus()
        bg_cores: Set[int] = self._target_bg.cgroup_cpuset.read_cpus()
        overlapped = fg_cores & bg_cores
        if overlapped is not None:
            return True
        else:
            return False

    def enforce(self) -> None:
        logger = logging.getLogger(__name__)
        # FIXME: hard coded & Assuming that bg task is placed to CPUs which have higher CPU IDs (e.g., 4,5)
        logger.info(f'[enforce] self._cur_action : {self._cur_action}')
        logger.info(f'[enforce] before, self._target_bg.bound_cores : {self._target_bg.bound_cores}')
        logger.info(f'[enforce] before, self._cur_step[{self._target_bg}] : {self._cur_step[self._target_bg]}, type: {type(self._cur_step[self._target_bg])}')
        core_ids = ','.join(map(str, self._cur_step[self._target_bg]))
        logger.info(f'[enforce] core_ids: {core_ids}, type: {type(core_ids)}')
        self._target_bg.bound_cores = self._cur_step[self._target_bg]
        logger.info(f'[enforce] after, changed self._target_bg.bound_cores : {self._target_bg.bound_cores}')
        for bg_wl, bg_bound_cores in self._cur_step.items():
            logger.info(f'[enforce] affinity of background [{bg_wl.group_name}] is {bg_bound_cores}')

    def reset(self) -> None:
        """
        reset() is used to set the original configuration for foregrounds (for measuring solo-run data),
        and it is also used to run bg task alone.
        :return:
        """
        for bg_wl in self._background_wls:
            if bg_wl.is_running:
                bg_wl.bound_cores = bg_wl.orig_bound_cores

    def store_cur_config(self) -> None:
        self._stored_config = self._cur_step

    def load_cur_config(self) -> None:
        """
        This function should restore the configurations of multiple background tasks.

        :return:
        """
        super().load_cur_config()

        self._cur_step = self._stored_config
        
        self.update_max_membw_bg()
        self._stored_config = None

    def update_max_membw_bg(self) -> None:
        logger = logging.getLogger(__name__)
        max_membw = -1
        max_membw_bg = None
        for bg_wl, _ in self._cur_step.items():
            avg_bg_wl_statistics = BasicMetric.calc_avg(bg_wl.metrics, 30)
            bg_wl_membw = avg_bg_wl_statistics.llc_miss_ps
            # FIXME: currently, this func. selects max membw bg_wl with at least two cores
            if bg_wl_membw > max_membw and bg_wl.num_cores > 1:
                max_membw = bg_wl_membw
                max_membw_bg = bg_wl
        self._max_mem_bg = max_membw_bg
        self._target_bg = max_membw_bg
        logger.info(f'self._max_mem_bg : {self._max_mem_bg}')
        logger.info(f'self._target_bg : {self._target_bg}')

        #print(f'self._max_mem_bg : {self._max_mem_bg}')
        #print(f'self._target_bg : {self._target_bg}')

    def update_min_cores_bg(self) -> None:
        logger = logging.getLogger(__name__)
        min_cores = 100000
        min_cores_bg = 100000
        for bg, bg_bound_cores in self._cur_step.items():
            if len(bg_bound_cores) < min_cores:
                min_cores = len(bg_bound_cores)
                min_cores_bg = bg
        self._min_cores_bg = min_cores_bg
        self._target_bg = min_cores_bg

        logger.info(f'self._min_cores_bg : {self._min_cores_bg}')
        logger.info(f'self._target_bg : {self._target_bg}')
