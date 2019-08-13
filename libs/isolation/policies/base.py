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

    def __init__(self, lc_wls: Set[Workload], be_wls: Set[Workload]) -> None:
        self._lc_wls = lc_wls
        self._be_wls = be_wls

        self._node_type = MachineChecker.get_node_type()
        # TODO: Discrete GPU case
        if self._node_type == NodeType.CPU:\

            self._isolator_map: Dict[Type[Isolator], Isolator] = dict((
                (CycleLimitIsolator, CycleLimitIsolator(self._lc_wls, self._be_wls)),
                (SchedIsolator, SchedIsolator(self._lc_wls, self._be_wls)),
            ))
        if self._node_type == NodeType.IntegratedGPU:
            self._isolator_map: Dict[Type[Isolator], Isolator] = dict((
                (AffinityIsolator, AffinityIsolator(self._lc_wls, self._be_wls)),
                (CycleLimitIsolator, CycleLimitIsolator(self._lc_wls, self._be_wls)),
                (GPUFreqThrottleIsolator, GPUFreqThrottleIsolator(self._lc_wls, self._be_wls)),
                (SchedIsolator, SchedIsolator(self._lc_wls, self._be_wls))
            ))
        self._cur_isolator: Isolator = IsolationPolicy._IDLE_ISOLATOR

        self._in_solorun_profiling_stage: bool = False
        self._cached_lc_num_threads: Dict[Workload, int] = dict()
        for lc_wl in self._lc_wls:
            self._cached_lc_num_threads[lc_wl] = lc_wl.number_of_threads
        self._solorun_verify_violation_count: Dict[Workload, int] = dict()
        self._all_lc_cores = set()
        self._all_be_cores = set()
        self._excess_cpu_wls = set()

    def __hash__(self) -> int:
        return id(self)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__} <LCs: {self._lc_wls}, BEs: {self._be_wls}>'

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

    def contentious_resource(self, target_wl: Workload) -> ResourceType:
        """
        It returns the most contentious resource considering diff_slack
        """
        return self.contentious_resources(target_wl)[0][0]

    def contentious_resources(self, target_wl: Workload) \
            -> Tuple[Tuple[ResourceType, float], ...]:
        metric_diff: MetricDiff = target_wl.calc_metric_diff()

        logger = logging.getLogger(__name__)
        logger.info(f'original diffs\' values')

        for idx, lc_wl in enumerate(self._lc_wls):
            logger.info(f'latency-critical[{idx}] {lc_wl.name}-{lc_wl.pid} : {lc_wl.calc_metric_diff()}')

        for idx, be_wl in enumerate(self._be_wls):
            logger.info(f'best-effort[{idx}] {be_wl.name}-{be_wl.pid} : {be_wl.calc_metric_diff()}')

        #resources = ((ResourceType.CACHE, metric_diff.llc_hit_ratio),
        #             (ResourceType.MEMORY, metric_diff.local_mem_util_ps))

        """
        Get the target workload's diff values
        Sorting resource contention (diff) descending order (when all value is positive, which means FG is fast)
        Sorting resource contention (diff) ascending order (if any non-positive value exist)
        """
        diff_slack = target_wl.diff_slack

        if diff_slack is None:
            resource_slacks = metric_diff.calc_by_diff_slack(0.0)
        elif diff_slack is not None:
            resource_slacks = metric_diff.calc_by_diff_slack(diff_slack)

        logger.info(f'resource_slack of {target_wl.name}-{target_wl.pid} is calculated '
                    f'by using `diff_slack` of {diff_slack}: {resource_slacks}')

        # logger.info(f'resource_slacks: {resource_slacks}, type: {type(resource_slacks)}')

        if all(v > 0 for m, v in resource_slacks):
            return tuple(sorted(resource_slacks, key=lambda x: x[1], reverse=True))

        else:
            return tuple(sorted(resource_slacks, key=lambda x: x[1]))

    @property
    def latency_critical_workloads(self) -> Set[Workload]:
        return self._lc_wls

    @latency_critical_workloads.setter
    def latency_critical_workloads(self, new_workloads: Set[Workload]):
        self._lc_wls = new_workloads
        for isolator in self._isolator_map.values():
            isolator.change_lc_wls(new_workloads)
            isolator.enforce()

    @property
    def best_effort_workloads(self) -> Set[Workload]:
        return self._be_wls

    @best_effort_workloads.setter
    def best_effort_workloads(self, new_workloads: Set[Workload]):
        self._be_wls = new_workloads
        for isolator in self._isolator_map.values():
            isolator.change_be_wls(new_workloads)
            isolator.enforce()

    @property
    def all_lc_cores(self) -> Set[int]:
        return self._all_lc_cores

    @property
    def all_be_cores(self) -> Set[int]:
        return self._all_be_cores

    @property
    def ended(self) -> bool:
        """
        This returns true when all latency-critical workloads are ended
        It also checks which workloads are alive
        This function triggers re-allocation of released resources from terminated workloads
        """
        be_ended: bool = True
        lc_ended: bool = True
        new_be_wls: Set[Workload] = set()
        new_lc_wls: Set[Workload] = set()

        for wl in self._be_wls:
            be_ended = be_ended and (not wl.is_running)
            if wl.is_running:
                new_be_wls.add(wl)
        self._be_wls = new_be_wls

        for wl in self._lc_wls:
            lc_ended = lc_ended and (not wl.is_running)
            if wl.is_running:
                new_lc_wls.add(wl)
        self._lc_wls = new_lc_wls

        return lc_ended or be_ended
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
        return f'{self._lc_wls.name}({self._lc_wls.pid})'

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
        return self._in_solorun_profiling_stage

    def start_solorun_profiling(self, target_wl: Workload) -> None:
        #print("solorun starting")
        """ profile solorun status of a latency-critical workload """
        if self._in_solorun_profiling_stage:
            raise ValueError('Stop the ongoing solorun profiling first!')

        self._in_solorun_profiling_stage = True
        self._cached_lc_num_threads[target_wl] = target_wl.number_of_threads
        self._solorun_verify_violation_count[target_wl] = 0

        # suspend all workloads and their perf agents
        for be_wl in self._be_wls:
            be_wl.pause()

        # clear currently collected metric values of target_lc_wl
        target_wl.metrics.clear()
        #target_wl.check_gpu_task()
        # store current configuration
        for isolator in self._isolator_map.values():
            isolator.store_cur_config()
            isolator.reset()

    def stop_solorun_profiling(self, target_wl: Workload) -> None:
        #print("solorun stopping")
        if not self._in_solorun_profiling_stage:
            raise ValueError('Start solorun profiling first!')

        logger = logging.getLogger(__name__)
        logger.debug(f'number of collected solorun data: {len(target_wl.metrics)}')
        self._lc_wls.remove(target_wl)
        target_wl.avg_solorun_data = BasicMetric.calc_avg(target_wl.metrics, len(target_wl.metrics))
        logger.debug(f'calculated average solorun data: {self._lc_wls.avg_solorun_data}')

        logger.debug('Enforcing restored configuration...')
        # restore stored configuration
        for isolator in self._isolator_map.values():
            isolator.load_cur_config()
            isolator.enforce()
        #print(self._fg_wl.metrics)
        target_wl.metrics.clear()
        self._lc_wls.add(target_wl)

        for be_wl in self._be_wls:
            be_wl.resume()

        self._in_solorun_profiling_stage = False

    def profile_needed(self, target_wl: Workload) -> bool:
        """
        This function checks if the profiling procedure should be called
        :return: Decision whether to initiate online solorun profiling
        """
        logger = logging.getLogger(__name__)

        #if self._lc_wls.avg_solorun_data is None:
        if target_wl.avg_solorun_data is None:
            logger.debug('initialize solorun data')
            return True

        if not target_wl.calc_metric_diff().verify():

            self._solorun_verify_violation_count[target_wl] += 1

            if self._solorun_verify_violation_count[target_wl] == self._VERIFY_THRESHOLD:
                logger.debug(f'fail to verify solorun data. {{{target_wl.calc_metric_diff()}}}')
                return True

        cur_num_threads = target_wl.number_of_threads
        if cur_num_threads is not 0 and self._cached_lc_num_threads[target_wl] != cur_num_threads:
            logger.debug(f'number of threads. cached: {self._cached_lc_num_threads[target_wl]}, '
                         f'current : {cur_num_threads}')
            return True

        return False

    """
    # Swapper related
    
    @property
    def safe_to_swap(self) -> bool:
        return not self._in_solorun_profiling_stage and self.check_lc_wls_metrics() and self._lc_wls.calc_metric_diff().verify()
    """

    # Multiple BGs related

    def check_any_bg_running(self) -> bool:
        not_running_bgs = 0
        for bg_wl in self._be_wls:
            if bg_wl.is_running:
                return True
            elif not bg_wl.is_running:
                not_running_bgs += 1
        if not_running_bgs == len(self._be_wls):
            return False

    def check_lc_wls_metrics(self) -> bool:
        logger = logging.getLogger(__name__)
        not_empty_metrics = 0
        logger.info(f'len of self._LC_wls {len(self._be_wls)}')
        for lc_wl in self._lc_wls:
            if len(lc_wl.metrics) > 0:
                not_empty_metrics += 1
            else:
                logger.info(f'return False')
            return False
        if not_empty_metrics == len(self._lc_wls):
            logger.info(f'return True')
            return True

    def check_be_wls_metrics(self) -> bool:
        logger = logging.getLogger(__name__)
        not_empty_metrics = 0
        logger.info(f'len of self._BE_wls {len(self._be_wls)}')
        for be_wl in self._be_wls:
            if len(be_wl.metrics) > 0:
                not_empty_metrics += 1
            else:
                logger.info(f'return False')
                return False
        if not_empty_metrics == len(self._be_wls):
            logger.info(f'return True')
            return True

    def update_allocated_cores(self):
        logger = logging.getLogger(__name__)

        all_lc_cores = set()
        all_be_cores = set()

        for be_wl in self._be_wls:
            be_cores: Set[int] = be_wl.cgroup_cpuset.read_cpus()
            all_be_cores |= be_cores

        for lc_wl in self._lc_wls:
            lc_cores: Set[int] = lc_wl.cgroup_cpuset.read_cpus()
            all_lc_cores |= lc_cores

        self._all_lc_cores = all_lc_cores
        self._all_be_cores = all_be_cores

    def update_excess_cpus_wls(self):
        for lc_wl in self._lc_wls:
            if lc_wl.number_of_threads < len(lc_wl.bound_cores):
                lc_wl.excess_cpu_flag = True

    def check_excess_cpus_wls(self) -> bool:
        self.update_excess_cpus_wls()
        ret = set()
        for lc_wl in self._lc_wls:
            if lc_wl.excess_cpu_flag is True:
                ret |= lc_wl
        # Update excess_cpu_wls
        self._excess_cpu_wls = ret
        if len(self._excess_cpu_wls) > 0:
            return True
        else:
            return False

    @staticmethod
    def choose_wl_of_negative_ips_diff(set_of_wl: Set[Workload]) -> Workload:
        """
        Choose most contentious workload or SLO-violated workload in terms of `instruction diff`
        :return:
        """
        logger = logging.getLogger(__name__)

        target_wl = None
        min_inst_diff = 0
        for wl in set_of_wl:
            curr_metric_diff = wl.calc_metric_diff()
            curr_inst_diff = curr_metric_diff.instruction_ps
            if curr_inst_diff < min_inst_diff:
                min_inst_diff = curr_inst_diff
                target_wl = wl

        if target_wl is None:
            logger.info(f'target_wl ([{target_wl.wl_type}] {target_wl.name}-{target_wl.pid}) is None!')
        if target_wl is not None:
            logger.info(f'lowest_instruction_diff target_wl: [{target_wl.wl_type}] '
                        f'{target_wl.name}-{target_wl.pid}, '
                        f'inst_diff: {min_inst_diff}')

        return target_wl

    @staticmethod
    def choose_wl_of_positive_ips_diff(set_of_wl: Set[Workload]) -> Workload:
        """
        Choose least contentious workload or SLO-compliant workload in terms of `instruction diff`
        :return:
        """
        logger = logging.getLogger(__name__)

        target_wl = None
        max_inst_diff = -1000
        for wl in set_of_wl:
            curr_metric_diff = wl.calc_metric_diff()
            curr_inst_diff = curr_metric_diff.instruction_ps
            if curr_inst_diff > max_inst_diff:
                max_inst_diff = curr_inst_diff
                target_wl = wl

        if target_wl is None:
            logger.info(f'target_wl ([{target_wl.wl_type}] {target_wl.name}-{target_wl.pid}) is None!')
        if target_wl is not None:
            logger.info(f'lowest_instruction_diff target_wl: [{target_wl.wl_type}] '
                        f'{target_wl.name}-{target_wl.pid}, '
                        f'inst_diff: {max_inst_diff}')

        return target_wl

    def choose_workload_to_be_allocated(self) -> Workload:
        """
        This function finds which workload should be prioritized for performance improvement
        :return:
        """
        logger = logging.getLogger(__name__)
        target_wl = None
        target_lc_wl = self.choose_wl_of_negative_ips_diff(self._lc_wls)
        if target_lc_wl is None:
            logger.info('Not any latency-critical workload is chosen!')
            target_be_wl = self.choose_wl_of_negative_ips_diff(self._be_wls)
            if target_be_wl is None:
                logger.info('Not any workload is running!')
            elif target_be_wl is not None:
                target_wl = target_be_wl
                logger.info(f'Best-effort workload ({target_be_wl.name}-{target_be_wl.pid})is chosen!')
        elif target_lc_wl is not None:
            logger.info(f'Latency-critical workload ({target_lc_wl.name}-{target_lc_wl.pid})is chosen!')
            target_wl = target_lc_wl

        return target_wl

    def choose_workload_to_be_deallocated(self) -> Workload:
        """
        This function finds which workload should be prioritized for performance improvement
        :return:
        """
        # FIXME: target workload for deallocation needs to be changed!! (BE first & LC last)
        logger = logging.getLogger(__name__)
        target_wl = None
        target_be_wl = self.choose_wl_of_positive_ips_diff(self._be_wls)
        if target_be_wl is None:
            logger.info('Not any best-effort workload is chosen!')
            target_lc_wl = self.choose_wl_of_positive_ips_diff(self._lc_wls)
            if target_lc_wl is None:
                logger.info('Not any workload is running!')
            elif target_lc_wl is not None:
                target_wl = target_lc_wl
                logger.info(f'Latency-critical workload ({target_lc_wl.name}-{target_lc_wl.pid})is chosen!')
        elif target_be_wl is not None:
            logger.info(f'Best-effort workload ({target_be_wl.name}-{target_be_wl.pid})is chosen!')
            target_wl = target_be_wl

        return target_wl

    """
    def most_contentious_workload(self) -> Workload:
        
        This function finds the most contentious workload, which has the largest diffs
        among multiple workloads (including LC & BE)
        :return:
        

        lc_diffs = list()
        be_diffs = list()
        if len(self._lc_wls) > 0:
            for lc_wl in self._lc_wls:
                lc_metric_diff = lc_wl.calc_metric_diff()
                lc_diffs.append((lc_wl, lc_metric_diff))
            target_diffs = lc_diffs
        else:
            if len(self._be_wls) > 0:
                for be_wl in self._be_wls:
                    be_metric_diff = be_wl.calc_metric_diff()
                    be_diffs.append((be_wl, be_metric_diff))
                target_diffs = be_diffs
        
        # tuple is sorted in descending order (e.g., -5, -4, -3, ... , -1, 0)
        # TODO: How can we sorting different diffs? metric_diff is 2-dim or 3-dim tuple
        sorted_tuple_list = sorted(target_diffs, key=lambda x: x[1])
        return sorted_tuple_list[0][0]
        """



    """
    # It is not used 
    def check_cores_are_overlapped(self) -> bool:
        logger = logging.getLogger(__name__)
        
        all_lc_cores = set()
        for lc_wl in self._lc_wls:
            lc_cores: Set[int] = lc_wl.cgroup_cpuset.read_cpus()
            all_lc_cores.add(lc_cores)
            
        for bg_wl in self._be_wls:
            bg_cores: Set[int] = bg_wl.cgroup_cpuset.read_cpus()
            overlapped = fg_cores & bg_cores
            if overlapped is not None:
                return True
            else:
                return False
    """