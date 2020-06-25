# coding: UTF-8

import logging
from abc import ABCMeta, abstractmethod
from typing import ClassVar, Dict, Tuple, Type, Set, Iterable, Optional, Any, List

from .. import ResourceType, NextStep
from ..isolators import Isolator, IdleIsolator, CycleLimitIsolator, CacheIsolator, \
    GPUFreqThrottleIsolator, CPUFreqThrottleIsolator, SchedIsolator, AffinityIsolator
# from ..isolators import CacheIsolator, IdleIsolator, Isolator, MemoryIsolator, SchedIsolator
# from ..isolators.affinity import AffinityIsolator
from ...metric_container.basic_metric import BasicMetric, MetricDiff
from ...workload import Workload
from ...utils.machine_type import MachineChecker, NodeType


class IsolationPolicy(metaclass=ABCMeta):
    #_IDLE_ISOLATOR: ClassVar[IdleIsolator] = IdleIsolator()
    _VERIFY_THRESHOLD: ClassVar[int] = 3
    #_available_cores: Optional[Tuple[int]] = None

    def __init__(self, lc_wls: Set[Workload], be_wls: Set[Workload]) -> None:
        self._IDLE_ISOLATOR: IdleIsolator = IdleIsolator(lc_wls, be_wls)
        self._lc_wls = lc_wls
        self._be_wls = be_wls
        self._all_wls = lc_wls | be_wls

        self._orig_lc_wls = lc_wls
        self._orig_be_wls = be_wls
        self._orig_all_wls = lc_wls | be_wls

        self._perf_target_wl = None  # selected workload to for `victim`
        self._alloc_target_wl = None  # selected workload to for `alloc`
        self._dealloc_target_wl = None  # selected workload to for `dealloc`
        self._dom_res_cont: Optional[ResourceType] = None  # currently dominant resource contention (type)
        self._dom_res_diff: Optional[float] = None   # diff value of dominant resource contention

        self._node_type = MachineChecker.get_node_type()
        # TODO: Discrete GPU case
        if self._node_type == NodeType.CPU:
            self._all_cores = tuple(range(0, 16, 1))
            self._isolator_map: Dict[Type[Isolator], Isolator] = dict((
                (AffinityIsolator, AffinityIsolator(self._lc_wls, self._be_wls)),
                (CacheIsolator, CacheIsolator(self._lc_wls, self._be_wls)),
                (CPUFreqThrottleIsolator, CPUFreqThrottleIsolator(self._lc_wls, self._be_wls)),
                (CycleLimitIsolator, CycleLimitIsolator(self._lc_wls, self._be_wls)),
                (SchedIsolator, SchedIsolator(self._lc_wls, self._be_wls)),
            ))
        if self._node_type == NodeType.IntegratedGPU:
            self._all_cores = tuple([0, 3, 4, 5])
            self._isolator_map: Dict[Type[Isolator], Isolator] = dict((
                (AffinityIsolator, AffinityIsolator(self._lc_wls, self._be_wls)),
                (CycleLimitIsolator, CycleLimitIsolator(self._lc_wls, self._be_wls)),
                (GPUFreqThrottleIsolator, GPUFreqThrottleIsolator(self._lc_wls, self._be_wls)),
                (SchedIsolator, SchedIsolator(self._lc_wls, self._be_wls))
            ))
        self._cur_isolator: Isolator = self._IDLE_ISOLATOR   # isolator init
        # TODO: cur_metric can be extended to maintain various resource contention for workloads
        self._cur_metrics: Dict[str, Dict[Workload, Any]] = dict()      # workload metric init
        # FIXME: metrics is hard-coded
        metrics = ['mem_bw', 'mem_bw_diff', 'llc_hr_diff', 'instr_diff']
        for metric in metrics:
            self._cur_metrics[metric] = dict()

        self._in_solorun_profiling_stage: bool = False
        self._cached_lc_num_threads: Dict[Workload, int] = dict()
        for lc_wl in self._lc_wls:
            self._cached_lc_num_threads[lc_wl] = lc_wl.number_of_threads
        self._solorun_verify_violation_count: Dict[Workload, int] = dict()
        self._all_lc_cores = set()
        self._all_be_cores = set()
        self.update_allocated_cores()   # Update allocated cores after initialization
        self._excess_cpu_wls = set()
        self._profile_target_wls = list()    # Workloads need to be profiled
        self._curr_profile_target: Optional[Workload] = None

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

    # def contentious_resources(self, target_wl: Workload) \
    #         -> Tuple[Tuple[ResourceType, float], ...]:
    def contentious_resources(self, target_wl: Workload) -> Iterable:
        """
        :param target_wl:
        :return: resource_diffs, Tuple[Tuple[ResourceType, float], ...]
        """
        metric_diff: MetricDiff = target_wl.calc_metric_diff()

        logger = logging.getLogger(__name__)
        logger.info(f'original diffs\' values')

        for idx, lc_wl in enumerate(self._lc_wls):
            logger.critical(f'latency-critical[{idx}] {lc_wl.name}-{lc_wl.pid} : {lc_wl.calc_metric_diff()}')

        for idx, be_wl in enumerate(self._be_wls):
            logger.critical(f'best-effort[{idx}] {be_wl.name}-{be_wl.pid} : {be_wl.calc_metric_diff()}')

        #resources = ((ResourceType.CACHE, metric_diff.llc_hit_ratio),
        #             (ResourceType.MEMORY, metric_diff.local_mem_util_ps))

        """
        Get the target workload's diff values
        Sorting resource contention (diff) descending order (when all value is positive, which means FG is fast)
        Sorting resource contention (diff) ascending order (if any non-positive value exist)
        """
        diff_slack = target_wl.diff_slack

        # Below code is for SLO_slack ver.
        #resource_slacks = metric_diff.calc_by_diff_slack(0.0)

        # Below code is for diff_slack ver. and diff+SLO ver.
        if diff_slack is None:
            resource_slacks = metric_diff.calc_by_diff_slack(0.0)
        elif diff_slack is not None:
            resource_slacks = metric_diff.calc_by_diff_slack(diff_slack)

        logger.critical(f'resource_slack of {target_wl.name}-{target_wl.pid} is calculated '
                    f'by using `diff_slack` of {diff_slack}: {resource_slacks}')

        # logger.info(f'resource_slacks: {resource_slacks}, type: {type(resource_slacks)}')

        if all(v > 0 for m, v in resource_slacks):
            return tuple(sorted(resource_slacks, key=lambda x: x[1], reverse=True))

        else:
            return tuple(sorted(resource_slacks, key=lambda x: x[1]))

    def update_dominant_contention(self) -> Iterable:
        logger = logging.getLogger(__name__)
        dom_res_cont = None
        dom_res_diff = None

        resource_slacks = self.contentious_resources(self.perf_target_wl)
        # Check whether resource contention is ResourceType.CPU
        free_cores_set = self.update_allocated_cores()
        logger.info(f'[update_dominant_contention] resoruces_slack: {resource_slacks}')
        logger.critical(f'[update_dominant_contention] free_cores_set: {free_cores_set}')
        logger.critical(f'[update_dominant_contention] self.check_excess_cpus_wls(): {self.check_excess_cpus_wls()}')
        #if len(free_cores_set) > 0 or self.check_excess_cpus_wls():
        if len(free_cores_set) > 0 and self.check_excess_cpus_wls():
            if AffinityIsolator in self._isolator_map:
                for res, diff in resource_slacks:
                    if res is ResourceType.CPU:
                        dom_res_cont = ResourceType.CPU
                        dom_res_diff = diff
                        self._dom_res_cont = dom_res_cont
                        self._dom_res_diff = dom_res_diff
                        logger.critical(f'[update_dominant_contention] self._dom_res_cont: {self._dom_res_cont}, '
                                        f'self._dom_res_diff: {self._dom_res_diff}')
                        #logger.info(f'[update_dominant_contention] self._dom_res_diff: {self._dom_res_diff}')
                        return resource_slacks

        # Find dominant resource contention rather than ResourceType.CPU
        for res, diff in resource_slacks:
            if res is not ResourceType.CPU:
                dom_res_cont = res
                dom_res_diff = diff
                break

        self._dom_res_cont = dom_res_cont
        self._dom_res_diff = dom_res_diff
        logger.critical(f'[update_dominant_contention] self._dom_res_cont: {self._dom_res_cont}, '
                        f'self._dom_res_diff: {self._dom_res_diff}')
        #logger.info(f'[update_dominant_contention] self._dom_res_diff: {self._dom_res_diff}')
        return resource_slacks

    @property
    def dom_res_cont(self) -> Optional[ResourceType]:
        return self._dom_res_cont

    @dom_res_cont.setter
    def dom_res_cont(self, new_val) -> None:
        self._dom_res_cont = new_val

    @property
    def dom_res_diff(self) -> Optional[float]:
        return self._dom_res_diff

    @dom_res_diff.setter
    def dom_res_diff(self, new_val) -> None:
        self._dom_res_diff = new_val

    # def sync(self) -> None:
    #     self._cur_isolator.perf_target_wl = self._perf_target_wl
    #     self._cur_isolator.alloc_target_wl = self._alloc_target_wl
    #     self._cur_isolator.dealloc_target_wl = self._dealloc_target_wl

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

    # @classmethod
    # def available_cores(cls) -> Tuple[int]:
    #     return cls._available_cores
    #
    # @classmethod
    # def set_available_cores(cls, new_values) -> None:
    #     cls._available_cores = new_values

    @property
    def curr_profile_target(self) -> Workload:
        return self._curr_profile_target

    @property
    def profile_target_wls(self) -> List[Workload]:
        return self._profile_target_wls

    @property
    def ended(self) -> bool:
        """
        This returns true when either all latency-critical workloads or best-effort ones are ended
        It also checks which workloads are alive
        This function triggers re-allocation of released resources from terminated workloads
        """
        logger = logging.getLogger(__name__)

        be_ended: bool = True
        lc_ended: bool = True
        solo_run: bool = False
        new_be_wls: Set[Workload] = set()
        new_lc_wls: Set[Workload] = set()
        orig_started_wls: int = len(self._orig_all_wls)

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

        #logger.info(f'lc_ended: {lc_ended}, be_ended: {be_ended}')
        if len(self._orig_be_wls) > 0 and be_ended:
            be_ended = True
        elif len(self._orig_be_wls) == 0:
            be_ended = False

        if len(self._orig_lc_wls) > 0 and lc_ended:
            lc_ended = True
        elif len(self._orig_lc_wls) == 0:
            lc_ended = False
        logger.info(f'lc_ended: {lc_ended}, be_ended: {be_ended}')

        total_running_wls = len(self._lc_wls) + len(self._be_wls)
        if total_running_wls == 1 and orig_started_wls > 1:
            solo_run = True

        return lc_ended or be_ended or solo_run
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
    def perf_target_wl(self) -> Workload:
        return self._perf_target_wl

    @perf_target_wl.setter
    def perf_target_wl(self, workload) -> None:
        self._perf_target_wl = workload

    @property
    def alloc_target_wl(self) -> Workload:
        return self._alloc_target_wl

    @alloc_target_wl.setter
    def alloc_target_wl(self, workload) -> None:
        self._alloc_target_wl = workload

    @property
    def dealloc_target_wl(self) -> Workload:
        return self._dealloc_target_wl

    @dealloc_target_wl.setter
    def dealloc_target_wl(self, workload) -> None:
        self._dealloc_target_wl = workload

    @property
    def name(self) -> str:
        logger = logging.getLogger(__name__)
        logger.debug(f'IdleIslator: {isinstance(self.cur_isolator, IdleIsolator)}')
        if not isinstance(self.cur_isolator, IdleIsolator):
            if self.cur_isolator.perf_target_wl is not None:
                return f'{self.cur_isolator.perf_target_wl.name}({self.cur_isolator.perf_target_wl.pid})'
            else:
                return f'No performance target workload'
        elif isinstance(self.cur_isolator, IdleIsolator):
            return f'None (IdleIsolator)'

    def set_idle_isolator(self) -> None:
        self._cur_isolator.yield_isolation()
        self._cur_isolator = self._IDLE_ISOLATOR

    def reset(self) -> None:
        for isolator in self._isolator_map.values():
            #print(isolator)
            isolator.reset()

    # Solorun profiling related

    @property
    def in_solorun_profiling(self) -> bool:
        return self._in_solorun_profiling_stage

    def start_solorun_profiling(self) -> None:
        #print("solorun starting")
        """ profile solorun status of a latency-critical workload """
        if self._in_solorun_profiling_stage:
            raise ValueError('Stop the ongoing solorun profiling first!')

        # FIXME: Hard-coded Assumption: there is at least one LC_WL app. in the group
        target_wl = self._profile_target_wls[0]
        self._curr_profile_target = target_wl
        #target_wl = self._curr_profile_target
        print(f'[start_solorun_profiling] target_wl: {target_wl}')

        self._in_solorun_profiling_stage = True
        self._cached_lc_num_threads[target_wl] = target_wl.number_of_threads
        self._solorun_verify_violation_count[target_wl] = 0

        # suspend all other workloads and their perf agents
        # All BE workloads are suspended and all LC workloads, which do not have profile data, are profiled.
        #for be_wl in self._be_wls:
        #    be_wl.pause()
        print(f'[start_solorun_profiling] self._all_wls: {self._all_wls}')
        for wl in self._all_wls:
            print(f'[start_solorun_profiling] wl: {wl}, target_wl: {target_wl}')
            if wl is not target_wl:
                wl.pause()

        print(f'[start_solorun_profiling] here!!, target_wl: {target_wl}')

        # clear currently collected metric values of target_lc_wl
        target_wl.metrics.clear()
        #target_wl.check_gpu_task()
        # store current configuration
        for isolator in self._isolator_map.values():
            isolator.store_cur_config()
            #print(f'[start_solorun_profiling] ')
            isolator.reset()

    def stop_solorun_profiling(self) -> None:
        #print("solorun stopping")
        if not self._in_solorun_profiling_stage:
            raise ValueError('Start solorun profiling first!')

        logger = logging.getLogger(__name__)

        # calculate average solo-run data
        for wl in self._profile_target_wls:
            wl.collect_metrics()
            logger.debug(f'[stop_solorun_profiling] number of collected solorun data: {len(wl.profile_metrics)}')
            wl._avg_solorun_data = BasicMetric.calc_avg(wl.profile_metrics, len(wl.profile_metrics))
            logger.info(f'[stop_solorun_profiling] calculated average solorun data: {wl.avg_solorun_data}')
            wl.metrics.clear()  # clear metrics

        logger.debug('[stop_solorun_profiling] Restoring configuration...')
        # restore stored configuration
        for isolator in self._isolator_map.values():
            isolator.load_cur_config()
            isolator.enforce()

        for wl in self._all_wls:
            wl.resume()

        self._in_solorun_profiling_stage = False

    def set_next_solorun_target(self) -> None:
        logger = logging.getLogger(__name__)
        next_wl_idx = -1

        logger.info(f"[set_next_solorun_target] profilie_target_wls : {self._profile_target_wls}")
        for idx, wl in enumerate(self._profile_target_wls):
            logger.info(f"[set_next_solorun_target] idx: {idx}, wl: {wl}, self._curr_profile_target: {self._curr_profile_target}")
            if wl is self._curr_profile_target:
                curr_wl_idx = idx
                next_wl_idx = (curr_wl_idx + 1) % len(self._profile_target_wls)
                logger.info(f"[set_next_solorun_target] curr_wl_idx: {curr_wl_idx}, next_wl_idx: {next_wl_idx}, len(self._profile_target_wls): {len(self._profile_target_wls)}")
        try:
            self._curr_profile_target = self._profile_target_wls[next_wl_idx]
        except IndexError or ValueError:
            logger.info(f" there is no profilie_target : {self._curr_profile_target}, next_wl_idx: {next_wl_idx}")
            self._curr_profile_target = None

    def switching_profile_target(self) -> None:
        logger = logging.getLogger(__name__)

        # Step1. Suspend currently running LC workload which is profiled
        curr_wl = self.curr_profile_target
        logger.info(f'[switching_profile_target] curr_wl: {curr_wl}')
        curr_wl.pause()
        curr_wl.collect_metrics()   # Move items from `metrics` queue to `profile_metric` queue
        curr_wl.metrics.clear()     # Clear collected metrics during the solo-run stage

        # Step2. Choose the next LC workload (Round Robin) & Resume
        self.set_next_solorun_target()
        next_wl = self._curr_profile_target
        logger.info(f'[switching_profile_target] next_wl: {next_wl}')
        # init value for `next_wl`
        self._cached_lc_num_threads[next_wl] = next_wl.number_of_threads
        self._solorun_verify_violation_count[next_wl] = 0

        logger.info(f'[switching_profile_target] resume next_wl : {next_wl}')
        next_wl.resume()

        # Step3. Reset some values (e.g., curr_profile_target) to ready for the next profiling
        # Deal with reset function!
        #

    def check_profile_target(self, profile_interval: float) -> float:
        logger = logging.getLogger(__name__)
        # check profile target workloads
        profile_target_wls = list()
        for lc_wl in self._lc_wls:
            if lc_wl.need_profiling:
                profile_target_wls.append(lc_wl)

        self._profile_target_wls = profile_target_wls

        total_profile_time = float(len(self._profile_target_wls) * profile_interval)
        logger.info(f'profile_target_wls: {profile_target_wls}, len: {len(profile_target_wls)}')
        logger.info(f'total_profile_time: {total_profile_time}')
        # ex) 3 * 2.0 = 6.0
        return total_profile_time

    def profile_needed(self) -> bool:
        """
        This function checks if the profiling procedure should be called
        :return: Decision whether to initiate online solorun profiling
        """
        logger = logging.getLogger(__name__)

        ret = None
        for lc_wl in self._lc_wls:
            # There is no avg solorun data
            if lc_wl.avg_solorun_data is None:
                logger.info(f'[profile_needed] initialize solorun data of {lc_wl}')
                lc_wl.need_profiling = True
                ret = True

            # There is detected anomaly in `metric diff`
            if lc_wl.avg_solorun_data is not None and not lc_wl.calc_metric_diff().verify():
                self._solorun_verify_violation_count[lc_wl] += 1
                if self._solorun_verify_violation_count[lc_wl] == self._VERIFY_THRESHOLD:
                    logger.info(f'[profile_needed] fail to verify solorun data. {lc_wl.calc_metric_diff()}')
                    lc_wl.need_profiling = True
                    ret = True

            # When the number of threads are changed (`avg_solorun_data` needs to be updated)
            cur_num_threads = lc_wl.number_of_threads
            if cur_num_threads is not 0 and self._cached_lc_num_threads[lc_wl] != cur_num_threads:
                logger.info(f'[profile_needed] number of threads. cached: {self._cached_lc_num_threads[lc_wl]}, '
                             f'current : {cur_num_threads}')
                lc_wl.need_profiling = True
                ret = True

        if ret is True:
            return True
        else:
            return False

    """
    # Swapper related
    
    @property
    def safe_to_swap(self) -> bool:
        return not self._in_solorun_profiling_stage and self.check_lc_wls_metrics() and self._lc_wls.calc_metric_diff().verify()
    """

    # for supporting multiple workloads

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

    # for CPU core allocation

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
        #logger.info(f'[update_allocated_cores] self._all_lc_cores: {self._all_lc_cores}')
        #logger.info(f'[update_allocated_cores] self._all_be_cores: {self._all_be_cores}')
        available_cores = tuple(set(self._all_cores) - set(self._all_lc_cores) - set(self._all_be_cores))
        #logger.info(f'[update_allocated_cores] available_cores: {available_cores}')
        return available_cores
        #IsolationPolicy.set_available_cores(available_cores)

    def update_excess_cpus_wls(self):
        for lc_wl in self._lc_wls:
            if lc_wl.number_of_threads < len(lc_wl.bound_cores):
                lc_wl.excess_cpu_flag = True

    def check_excess_cpus_wls(self) -> bool:
        logger = logging.getLogger(__name__)
        self.update_excess_cpus_wls()
        ret = set()
        for lc_wl in self._lc_wls:
            #logger.info(f'[check_excess_cpus_wls] lc_wl.excess_cpu_flag: {lc_wl.excess_cpu_flag}')
            if lc_wl.excess_cpu_flag is True:
                ret.add(lc_wl)
        # Update excess_cpu_wls
        self._excess_cpu_wls = ret
        #logger.info(f'[check_excess_cpus_wls] self._excess_cpu_wls: {self._excess_cpu_wls}, len: {len(self._excess_cpu_wls)}')
        if len(self._excess_cpu_wls) > 0:
            return True
        else:
            return False

    # for choosing a workload to monitoring performance (sorting & choosing)

    def update_all_workloads_res_info(self) -> None:
        logger = logging.getLogger(__name__)
        # FIXME: metrics is hard-coded
        metrics = ['mem_bw', 'mem_bw_diff', 'llc_hr_diff', 'instr_diff']
        for wl in self._all_wls:
            if len(wl.metrics) > 0:
                wl.update_calc_metrics()
                for metric in metrics:
                    #logger.info(f'{wl.calc_metrics}')
                    self._cur_metrics[metric][wl] = wl.calc_metrics[metric]
            #else:?

    def choosing_wl_for(self, objective: str, sort_criteria: str, highest: bool) -> None:
        """
        This function is used for choosing workload subject to certain condition.
        :param objective: It indicates the intention for choosing workload (weaken, strengthen, or victim)
        :param sort_criteria: It indicates the sorting criteria (i.e., mem_bw for strengthen, mem_bw_diff for weaken)
        :param highest: It indicates the sort direction (e.g., True -> descending or False -> ascending)
        :return:
        """
        # TODO: This function only choose a workload of the highest memory diff
        # FIXME: This function should override the `self._alloc_target_wl` & `self._dealloc_target_wl`
        logger = logging.getLogger(__name__)
        if isinstance(self._cur_isolator, AffinityIsolator) or objective is "victim":
            # Affinity isolator performs isolation on `self._perf_target_wl`
            excluded: Iterable = ()
        else:
            # other isolators do not perform isolation on `self._perf_target_wl`
            excluded: Iterable = (self._perf_target_wl, )

        # update resource info
        self.update_all_workloads_res_info()

        # sorting workloads by sort_criteria (metric_type)
        wls_info = tuple(sorted(self._cur_metrics[sort_criteria].items(), key=lambda x: x[1], reverse=highest))
        chosen = False
        idx = 0
        logger.debug(f'[choosing_wl_for] wls_info: {wls_info}')
        logger.debug(f'[choosing_wl_for] self._cur_metrics[{sort_criteria}]: {self._cur_metrics[sort_criteria]}')
        if wls_info is None:
            logger.debug("[choosing_wl_for] There is no any workloads to sort!")
            self._cur_isolator.alloc_target_wl = None
            self._cur_isolator.dealloc_target_wl = None
            self._cur_isolator.perf_target_wl = None
            self._perf_target_wl = self._cur_isolator.perf_target_wl
            self._alloc_target_wl = self._cur_isolator.alloc_target_wl
            self._dealloc_target_wl = self._cur_isolator.dealloc_target_wl
            return

        # Choosing a workload which has the highest (or the lowest) values for a given metric
        candidate = tuple(filter(lambda x: x[0] not in excluded, wls_info))
        num_candidates = len(candidate)
        logger.debug(f'[choosing_wl_for] 1st candidate: {candidate}, before loop')
        logger.debug(f'[choosing_wl_for] wls_info: {wls_info}, before loop')
        logger.debug(f'[choosing_wl_for] num_candidates: {num_candidates}, before loop')
        while not chosen and idx < num_candidates:
            logger.debug(f'[choosing_wl_for] idx: {idx}, candidate: {candidate}')
            cur_target_wl: Workload = candidate[idx][0]   # idx is the ordinal position from the very first one

            logger.debug(f'[choosing_wl_for] idx: {idx}, num_candidates: {num_candidates}, candidate: {candidate}, [{cur_target_wl.wl_type}] cur_target_wl: {cur_target_wl} in loop')
            logger.debug(f'[choosing_wl_for] idx: {idx}, num_candidates: {num_candidates}, excluded: {excluded} in loop')
            # cur_memory_bw_diff is criteria for choosing a deallocable candidate for weakening
            # cur_memory_bw is criteria for choosing an allocable candidate for strengthening
            if objective is "victim":
                logger.debug(f"[choosing_wl_for] wl_type : {cur_target_wl.wl_type}")
                # FIXME: Comment in/out for choosing different ver.
                #if cur_target_wl.wl_type == "LC" and self._cur_metrics[sort_criteria][cur_target_wl] < 0:  # diff + slo_slack ver.
                #if cur_target_wl.wl_type == "LC" and self._cur_metrics[sort_criteria][cur_target_wl] - cur_target_wl.diff_slack < 0:  # slo_only ver.
                if cur_target_wl.wl_type == "LC":   # diff_slack_only ver.
                    self._cur_isolator.perf_target_wl = cur_target_wl
                    self._perf_target_wl = cur_target_wl
                    chosen = True
                elif cur_target_wl.wl_type == "BE":
                    excluded += (cur_target_wl, )

            if not isinstance(self._cur_isolator, AffinityIsolator):
                if objective is "strengthen":
                    logger.info(f"[choosing_wl_for] cur_isolator: {self._cur_isolator}, obj: {objective}")
                    logger.info(f"[choosing_wl_for] cur_target_wl: {cur_target_wl}")
                    logger.info(f"[choosing_wl_for] self._cur_isolator.dealloc_target_wl: {self.cur_isolator.dealloc_target_wl}")
                    logger.info(f"[choosing_wl_for] dealloc_target_wl: {self._dealloc_target_wl}")
                    if not self._cur_isolator.is_max_level:
                        self._cur_isolator.dealloc_target_wl = cur_target_wl
                        self._dealloc_target_wl = cur_target_wl
                        self._cur_isolator.alloc_target_wl = None
                        self._alloc_target_wl = None
                        chosen = True
                        continue
                    else:
                        excluded += (cur_target_wl, )
                elif objective is "weaken":
                    if not self._cur_isolator.is_min_level:
                        self._cur_isolator.alloc_target_wl = cur_target_wl
                        self._alloc_target_wl = cur_target_wl
                        self._cur_isolator.dealloc_target_wl = None
                        self._dealloc_target_wl = None
                        chosen = True
                        continue
                    else:
                        excluded += (cur_target_wl, )

            elif isinstance(self._cur_isolator, AffinityIsolator):
                # Setting cur_target_wl as `perf_target_wl`
                cur_target_wl = self._cur_isolator.perf_target_wl
                if objective is "strengthen":
                    if not self._cur_isolator.is_max_level:
                        self._cur_isolator.alloc_target_wl = cur_target_wl
                        self._alloc_target_wl = cur_target_wl
                        self._cur_isolator.dealloc_target_wl = None
                        self._dealloc_target_wl = None
                        chosen = True
                        continue
                    else:
                        excluded += (cur_target_wl, )
                elif objective is "weaken":
                    if not self._cur_isolator.is_min_level:
                        self._cur_isolator.dealloc_target_wl = cur_target_wl
                        self._dealloc_target_wl = cur_target_wl
                        self._cur_isolator.alloc_target_wl = None
                        self._alloc_target_wl = None
                        chosen = True
                        continue
                    else:
                        excluded += (cur_target_wl, )
            idx += 1
            logger.debug(f"[choosing_wl_for] chosen: {chosen}, idx: {idx}, num_candidates: {num_candidates}")

        # If it is not chosen, initialize all other variables
        if not chosen:
            logger.debug(f"[choosing_wl_for] There is no chosen workload for {objective}, "
                         f"objective:{objective}, chosen: {chosen}")
            if objective is "strengthen" or "weaken":
                self._cur_isolator.alloc_target_wl = None
                self._cur_isolator.dealloc_target_wl = None
                self._alloc_target_wl = self._cur_isolator.alloc_target_wl
                self._dealloc_target_wl = self._cur_isolator.dealloc_target_wl
            if objective is "victim":
                self._cur_isolator.perf_target_wl = None
                self._perf_target_wl = self._cur_isolator.perf_target_wl

        logger.debug(f"[choosing_wl_for] Chosen Workloads for {self._cur_isolator}")
        logger.debug(f"[choosing_wl_for] for perf_target: {self._cur_isolator.perf_target_wl}, "
                     f"for alloc: {self._cur_isolator.alloc_target_wl}, "
                     f"for dealloc: {self._cur_isolator.dealloc_target_wl}")

    # Matching res_type and sort_criteria
    # FIXME: The below function may be relocated to policy/base.py to be provided for making other policy
    #def choose_isolation_target(self, res_type: ResourceType, diff_value: float) -> None:
    def choose_isolation_target(self) -> None:
        logger = logging.getLogger(__name__)

        res_type = self._dom_res_cont
        diff_value = self._dom_res_diff

        # Currently, this function considers only "current" diff values of "current" dominant resource contention
        # check `diff_value` of perf_target_wl
        if diff_value < 0:
            action = 'strengthen'
        else:   # diff_value >= 0 case
            action = 'weaken'

        # TODO: Below code is dummy
        #target_wl = self.perf_target_wl
        #action = decided_next_step
        #
        #cur_isolator = self.cur_isolator
        #curr_metric_diff = cur_isolator._prev_metric_diff[target_wl]
        #res_type = cur_isolator._get_res_type_from()

        # sort_metric[0] : strengthen criteria, sort_metric[1] : weaken criteria
        sort_metric = ["instr_diff"] * 2    # default
        pick_order = [True, False]          # default for pick_order (for
        if res_type is ResourceType.CPU:
            sort_metric = ["instr_diff"] * 2
        elif res_type is ResourceType.CACHE:
            sort_metric = ["llc_hr_diff"] * 2
        elif res_type is ResourceType.MEMORY:
            sort_metric = ["mem_bw", "mem_bw_diff"] # FIXME: NEED TESTING
        else:   # IdleIsolator is included in this case
            logger.info(f"[choose_isolation_target] Unknown resource type: {res_type}, "
                        f"sort_criteria is determined to instr_diff")

        if self._cur_isolator is not AffinityIsolator:
            pick_order = [True, False]
        elif self._cur_isolator is AffinityIsolator:
            pick_order = [False, True]

        logger.info(f'[choose_isolation_target] res_type: {res_type}')
        logger.info(f'[choose_isolation_target] sort_metric: {sort_metric}')
        logger.info(f'[choose_isolation_target] pick_order: {pick_order}')
        # FIXME: In this code, there is an assumption that two workloads are always selected.
        # FIXME: How about two workloads? -> deciding the direction of `strengthen` and `weaken`
        #logger.debug(f'[choose_isolation_target] self._leftover: {self._leftover}, action: {action}')
        logger.info(f'[choose_isolation_target] action: {action}')
        if not isinstance(self.cur_isolator, IdleIsolator):
            if action == 'strengthen':
            #if action is NextStep.STRENGTHEN:
                self.choosing_wl_for(objective="strengthen",
                                     sort_criteria=sort_metric[0],
                                     highest=pick_order[0])
            elif action == 'weaken':
            #elif action is NextStep.WEAKEN:
                self.choosing_wl_for(objective="weaken",
                                     sort_criteria=sort_metric[1],
                                     highest=pick_order[1])

        logger.debug(f"[choose_isolation_target] self._cur_isolator: {self._cur_isolator}")
        logger.debug(f"[choose_isolation_target] alloc_target_wl: {self._cur_isolator.alloc_target_wl}")
        logger.debug(f"[choose_isolation_target] dealloc_target_wl: {self._cur_isolator.dealloc_target_wl}")
        logger.debug(f"[choose_isolation_target] perf_target_wl: {self._cur_isolator.perf_target_wl}")
