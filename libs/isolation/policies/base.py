# coding: UTF-8

import logging
from abc import ABCMeta, abstractmethod
from typing import ClassVar, Dict, Tuple, Type, Set, Iterable, Optional

from .. import ResourceType
from ..isolators import Isolator, IdleIsolator, CycleLimitIsolator, \
GPUFreqThrottleIsolator, CPUFreqThrottleIsolator, SchedIsolator, AffinityIsolator
# from ..isolators import CacheIsolator, IdleIsolator, Isolator, MemoryIsolator, SchedIsolator
# from ..isolators.affinity import AffinityIsolator
from ...metric_container.basic_metric import BasicMetric, MetricDiff
from ...workload import Workload
from ...utils.machine_type import MachineChecker, NodeType


class IsolationPolicy(metaclass=ABCMeta):
    _IDLE_ISOLATOR: ClassVar[IdleIsolator] = IdleIsolator()
    _VERIFY_THRESHOLD: ClassVar[int] = 3

    def __init__(self, fg_wl: Workload, bg_wls: Set[Workload]) -> None:
        self._fg_wl = fg_wl
        self._bg_wls = bg_wls

        self._node_type = MachineChecker.get_node_type()
        # TODO: Discrete GPU case
        if self._node_type == NodeType.CPU:
            self._isolator_map: Dict[Type[Isolator], Isolator] = dict((
                (CycleLimitIsolator, CycleLimitIsolator(self._fg_wl, self._bg_wls)),
                (SchedIsolator, SchedIsolator(self._fg_wl, self._bg_wls)),
            ))
        if self._node_type == NodeType.IntegratedGPU:
            self._isolator_map: Dict[Type[Isolator], Isolator] = dict((
                (AffinityIsolator, AffinityIsolator(self._fg_wl, self._bg_wls)),
                (CycleLimitIsolator, CycleLimitIsolator(self._fg_wl, self._bg_wls)),
                (GPUFreqThrottleIsolator, GPUFreqThrottleIsolator(self._fg_wl, self._bg_wls)),
                (SchedIsolator, SchedIsolator(self._fg_wl, self._bg_wls))
            ))
        self._cur_isolator: Isolator = IsolationPolicy._IDLE_ISOLATOR

        self._in_solorun_profile: bool = False
        self._cached_fg_num_threads: int = fg_wl.number_of_threads
        self._solorun_verify_violation_count: int = 0

    def __hash__(self) -> int:
        return id(self)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__} <fg: {self._fg_wl}, bgs: {self._bg_wls}>'

    def __del__(self) -> None:
        isolators = tuple(self._isolator_map.keys())
        for isolator in isolators:
            del self._isolator_map[isolator]

    @property
    @abstractmethod
    def new_isolator_needed(self) -> bool:
        pass

    @abstractmethod
    def choose_next_isolator(self) -> bool:
        pass

    def contentious_resource(self) -> ResourceType:
        return self.contentious_resources()[0][0]

    def contentious_resources(self, diff_slack: Optional[float] = None) -> Tuple[Tuple[ResourceType, float], ...]:
        metric_diff: MetricDiff = self._fg_wl.calc_metric_diff()

        logger = logging.getLogger(__name__)
        logger.info(f'foreground : {metric_diff}')
        # logger.info(f'backgrounds : {self._bg_wl.calc_metric_diff()}')
        for idx, bg_wl in enumerate(self._bg_wls):
            logger.info(f'background[{idx}] : {bg_wl.calc_metric_diff()}')

        #resources = ((ResourceType.CACHE, metric_diff.llc_hit_ratio),
        #             (ResourceType.MEMORY, metric_diff.local_mem_util_ps))

        """
        Sorting resource contention (diff) descending order (when all value is positive, which means FG is fast)
        Sorting resource contention (diff) ascending order (if any non-positive value exist)
        """
        if diff_slack is None:
            resource_slacks = metric_diff.calc_by_diff_slack(0.0)
        elif diff_slack is not None:
            resource_slacks = metric_diff.calc_by_diff_slack(diff_slack)

        if all(v > 0 for m, v in resource_slacks.items()):
            return tuple(sorted(resource_slacks, key=lambda x: x[1], reverse=True))

        else:
            return tuple(sorted(resource_slacks, key=lambda x: x[1]))

    @property
    def foreground_workload(self) -> Workload:
        return self._fg_wl

    @foreground_workload.setter
    def foreground_workload(self, new_workload: Workload):
        self._fg_wl = new_workload
        for isolator in self._isolator_map.values():
            isolator.change_fg_wl(new_workload)
            isolator.enforce()

    @property
    def background_workloads(self) -> Set[Workload]:
        return self._bg_wls

    @background_workloads.setter
    def background_workloads(self, new_workloads: Set[Workload]):
        self._bg_wls = new_workloads
        for isolator in self._isolator_map.values():
            isolator.change_bg_wls(new_workloads)
            isolator.enforce()

    @property
    def ended(self) -> bool:
        """
        This returns true when the foreground workload is ended
        """
        ret: bool = True
        new_bg_wls: Set[Workload] = set()

        for wl in self._bg_wls:
            ret = ret and (not wl.is_running)
            if wl.is_running:
                new_bg_wls.add(wl)
        self._bg_wls = new_bg_wls
        return not self._fg_wl.is_running or ret
        #
        # if not self._fg_wl.is_running:
        #     return True
        # else:
        #     return False
        # return not self._fg_wl.is_running or not self._bg_wls.all_is_running

    @property
    def cur_isolator(self) -> Isolator:
        return self._cur_isolator

    @property
    def name(self) -> str:
        return f'{self._fg_wl.name}({self._fg_wl.pid})'

    def set_idle_isolator(self) -> None:
        self._cur_isolator.yield_isolation()
        self._cur_isolator = IsolationPolicy._IDLE_ISOLATOR

    def reset(self) -> None:
        for isolator in self._isolator_map.values():
            #print(isolator)
            isolator.reset()

    # Solorun profiling related

    @property
    def in_solorun_profiling(self) -> bool:
        return self._in_solorun_profile

    def start_solorun_profiling(self) -> None:
        #print("solorun starting")
        """ profile solorun status of a workload """
        if self._in_solorun_profile:
            raise ValueError('Stop the ongoing solorun profiling first!')

        self._in_solorun_profile = True
        self._cached_fg_num_threads = self._fg_wl.number_of_threads
        self._solorun_verify_violation_count = 0

        # suspend all workloads and their perf agents
        for bg_wl in self._bg_wls:
            bg_wl.pause()

        self._fg_wl.metrics.clear()
        #self.foreground_workload.check_gpu_task()
        # store current configuration
        for isolator in self._isolator_map.values():
            isolator.store_cur_config()
            isolator.reset()

    def stop_solorun_profiling(self) -> None:
        #print("solorun stopping")
        if not self._in_solorun_profile:
            raise ValueError('Start solorun profiling first!')

        logger = logging.getLogger(__name__)
        logger.debug(f'number of collected solorun data: {len(self._fg_wl.metrics)}')
        self._fg_wl.avg_solorun_data = BasicMetric.calc_avg(self._fg_wl.metrics, len(self._fg_wl.metrics))
        logger.debug(f'calculated average solorun data: {self._fg_wl.avg_solorun_data}')

        logger.debug('Enforcing restored configuration...')
        # restore stored configuration
        for isolator in self._isolator_map.values():
            isolator.load_cur_config()
            isolator.enforce()
        #print(self._fg_wl.metrics)
        self._fg_wl.metrics.clear()

        for bg_wl in self._bg_wls:
            bg_wl.resume()

        self._in_solorun_profile = False

    def profile_needed(self) -> bool:
        """
        This function checks if the profiling procedure should be called
        :return: Decision whether to initiate online solorun profiling
        """
        logger = logging.getLogger(__name__)

        if self._fg_wl.avg_solorun_data is None:
            logger.debug('initialize solorun data')
            return True

        if not self._fg_wl.calc_metric_diff().verify():
            self._solorun_verify_violation_count += 1

            if self._solorun_verify_violation_count == self._VERIFY_THRESHOLD:
                logger.debug(f'fail to verify solorun data. {{{self._fg_wl.calc_metric_diff()}}}')
                return True

        cur_num_threads = self._fg_wl.number_of_threads
        if cur_num_threads is not 0 and self._cached_fg_num_threads != cur_num_threads:
            logger.debug(f'number of threads. cached: {self._cached_fg_num_threads}, current : {cur_num_threads}')
            return True

        return False

    # Swapper related

    @property
    def safe_to_swap(self) -> bool:
        return not self._in_solorun_profile and len(self._fg_wl.metrics) > 0 and self._fg_wl.calc_metric_diff().verify()

    # Multiple BGs related

    def check_any_bg_running(self) -> bool:
        not_running_bgs = 0
        for bg_wl in self._bg_wls:
            if bg_wl.is_running:
                return True
            elif not bg_wl.is_running:
                not_running_bgs += 1
        if not_running_bgs == len(self._bg_wls):
            return False

    def check_bg_wls_metrics(self) -> bool:
        logger = logging.getLogger(__name__)
        not_empty_metrics = 0
        logger.info(f'len of self._bg_wls {len(self._bg_wls)}')
        for bg_wl in self._bg_wls:
            if len(bg_wl.metrics) > 0:
                not_empty_metrics += 1
            else:
                logger.info(f'return False')
                return False
        if not_empty_metrics == len(self._bg_wls):
            logger.info(f'return True')
            return True

    def check_cores_are_overlapped(self) -> bool:
        logger = logging.getLogger(__name__)
        fg_cores: Set[int] = self._fg_wl.cgroup_cpuset.read_cpus()
        for bg_wl in self._bg_wls:
            bg_cores: Set[int] = bg_wl.cgroup_cpuset.read_cpus()
            overlapped = fg_cores & bg_cores
            if overlapped is not None:
                return True
            else:
                return False
