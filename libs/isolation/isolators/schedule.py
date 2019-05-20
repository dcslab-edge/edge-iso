# coding: UTF-8

import logging
from typing import Optional, Set, Dict

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
        # self._bg_wl = self._bgs_list[0]
        # self._num_of_bg_wls = len(self._bgs_list)
        self._cur_step: Dict[Workload, int] = dict()
        self._cur_action: NextStep = NextStep.IDLE
        for bg_wl in self._bgs_list:
            self._cur_step[bg_wl] = bg_wl.num_cores
        # self._cur_step: Dict[Workload, int] = [(bg_wl, bg_wl.num_cores) for bg_wl in self._bgs_list]

        self._target_bg: Workload = None        # The target bg for re-assigning bounded cores
        self._max_mem_bg: Workload = None
        self._min_cores_bg: Workload = None

        self._stored_config: Optional[int] = None

    @classmethod
    def _get_metric_type_from(cls, metric_diff: MetricDiff) -> float:
        return metric_diff.local_mem_util_ps

    def strengthen(self) -> 'SchedIsolator':
        self.update_max_membw_bg()
        if self._target_bg is not None:
            self._cur_step[self._target_bg] -= 1
            self._cur_action = NextStep.STRENGTHEN
            #self._cur_step[self._max_mem_bg] -= 1
        # FIXME: hard coded (All workloads are running on the contiguous CPU ID)
        return self

    def weaken(self) -> 'SchedIsolator':
        self.update_min_cores_bg()
        if self._target_bg is not None:
            self._cur_step[self._target_bg] += 1
            self._cur_action = NextStep.WEAKEN
            #self._cur_step[self._min_cores_bg] += 1
        return self

    @property
    def is_max_level(self) -> bool:
        # At least a process needs one core for its execution
        min_cores = min(self._cur_step.values())
        if min_cores < 1:
            return True
        else:
            return False

    @property
    def is_min_level(self) -> bool:
        # FIXME: hard coded
        # At most background processes can not preempt cores of the foreground process
        bg_cores = sum(self._cur_step.values())
        fg_cores = self._foreground_wl.num_cores
        return (bg_cores + fg_cores) > self._MAX_CORES or self.is_overlapped_assignment()

    def is_overlapped_assignment(self) -> bool:
        overlapped = self._foreground_wl.bound_cores & self._target_bg.bound_cores
        if overlapped is not None:
            return True
        else:
            return False


    def enforce(self) -> None:
        logger = logging.getLogger(__name__)
        # FIXME: hard coded & Assuming that bg task is placed to CPUs which have higher CPU IDs (e.g., 4,5)
        #logger.info(f'{self._target_bg.bound_cores[0]}, {self._target_bg.bound_cores[-1]}, {self._target_bg.bound_cores[0]}')
        if self._cur_action == NextStep.STRENGTHEN:
            self._target_bg.bound_cores = range(self._target_bg.bound_cores[0]+1,
                                                self._target_bg.bound_cores[-1])
        elif self._cur_action == NextStep.WEAKEN:
            self._target_bg.bound_cores = range(self._target_bg.bound_cores[0]-1,
                                                self._target_bg.bound_cores[-1])
        self._cur_action = NextStep.IDLE
        #self._target_bg.bound_cores = range(self._target_bg.bound_cores)
        for bg_wl, bg_cores in self._cur_step.items():
            logger.info(f'affinity of background [{bg_wl.group_name}] is {bg_wl.bound_cores}')

    def reset(self) -> None:
        for bg_wl in self._background_wls:
            if bg_wl.is_running:
                bg_wl.bound_cores = bg_wl.orig_bound_cores

    def store_cur_config(self) -> None:
        self._stored_config = self._cur_step

    def load_cur_config(self) -> None:
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
        min_cores = -1
        min_cores_bg = -1
        for bg, bg_cores in self._cur_step.items():
            if bg_cores > min_cores:
                min_cores = bg_cores
                min_cores_bg = bg
        self._min_cores_bg = min_cores_bg
        self._target_bg = min_cores_bg

        logger.info(f'self._min_cores_bg : {self._min_cores_bg}')
        logger.info(f'self._target_bg : {self._target_bg}')