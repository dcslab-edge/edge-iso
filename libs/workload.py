# coding: UTF-8

from collections import deque
from itertools import chain
from typing import Deque, Iterable, Optional, Set, Tuple, Dict, Any
from pathlib import Path

import psutil
import subprocess
import logging

from .metric_container.basic_metric import BasicMetric, MetricDiff
from .solorun_data.datas import data_map
from .utils import CPUDVFS, GPUDVFS  # , ResCtrl, numa_topology
from .utils.cgroup import Cpu, CpuSet


class Workload:
    """
    This class abstracts the process and contains the related metrics to represent its characteristics
    Controller schedules the groups of `Workload' instances to enforce their scheduling decisions
    """

    def __init__(self, name: str, wl_type: str, pid: int, wl_diff_slack: float,
                 perf_pid: int, perf_interval: int) -> None:
        #print("+++++++++++++++++++++++=WORKLOAD INITIATED+++++++++++++")
        self._name = name
        self._wl_type = wl_type # BE or LC
        self._is_gpu_task = 1  # if yes, set to 1, otherwise set to 0.

        self._pid = pid
        self._metrics: Deque[BasicMetric] = deque()
        self._perf_pid = perf_pid
        self._perf_interval = perf_interval

        self._proc_info = psutil.Process(pid)
        self._perf_info = psutil.Process(perf_pid)

        self._cgroup_cpuset = CpuSet(self.group_name)
        self._cgroup_cpu = Cpu(self.group_name)
        #self._resctrl = ResCtrl(self.group_name)
        self._cpu_dvfs = CPUDVFS(self.group_name)
        self._gpu_dvfs = GPUDVFS(self.group_name)

        # This variable is used to contain the recent avg. status
        self._avg_solorun_data: Optional[BasicMetric] = None

        if wl_type == 'BE':
            self._avg_solorun_data = data_map[name]

        # metric used to various isolation
        self._calc_metrics: Dict[str, Any] = dict()

        self._orig_bound_cores: Tuple[int, ...] = tuple(self._cgroup_cpuset.read_cpus())
        self._orig_bound_mems: Set[int] = self._cgroup_cpuset.read_mems()
        self._excess_cpu_flag = False   # flag indicating unused cpu cores exist
        self._diff_slack = wl_diff_slack    # slack for expressing diverse SLOs
                                            # (by controlling resource contention diff)
        self._need_profiling: bool = True
        self._profile_metrics: Deque[BasicMetric] = deque()

    def __repr__(self) -> str:
        return f'{self._name} (pid: {self._pid})'

    def __hash__(self) -> int:
        return self._pid

    @property
    def cgroup_cpuset(self) -> CpuSet:
        return self._cgroup_cpuset

    @property
    def cgroup_cpu(self) -> Cpu:
        return self._cgroup_cpu

    # @property
    # def resctrl(self) -> ResCtrl:
    #     return self._resctrl

    @property
    def is_gpu_task(self) -> int:
        return self._is_gpu_task

    @property
    def cpu_dvfs(self) -> CPUDVFS:
        return self._cpu_dvfs

    @property
    def gpu_dvfs(self) -> GPUDVFS:
        return self._gpu_dvfs

    @property
    def name(self) -> str:
        return self._name

    @property
    def pid(self) -> int:
        return self._pid

    @property
    def wl_type(self) -> str:
        return self._wl_type

    @property
    def metrics(self) -> Deque[BasicMetric]:
        return self._metrics

    @property
    def num_cores(self) -> int:
        return len(self.bound_cores)

    @property
    def bound_cores(self) -> Tuple[int, ...]:
        return tuple(self._cgroup_cpuset.read_cpus())

    @bound_cores.setter
    def bound_cores(self, core_ids: Iterable[int]):
        self._cgroup_cpuset.assign_cpus(core_ids)

    @property
    def orig_bound_cores(self) -> Tuple[int, ...]:
        return self._orig_bound_cores

    @orig_bound_cores.setter
    def orig_bound_cores(self, orig_bound_cores: Tuple[int, ...]) -> None:
        self._orig_bound_cores = orig_bound_cores

    @property
    def bound_mems(self) -> Tuple[int, ...]:
        return tuple(self._cgroup_cpuset.read_mems())

    @bound_mems.setter
    def bound_mems(self, affinity: Iterable[int]):
        self._cgroup_cpuset.assign_mems(affinity)

    @property
    def orig_bound_mems(self) -> Set[int]:
        return self._orig_bound_mems

    @orig_bound_mems.setter
    def orig_bound_mems(self, orig_bound_mems: Set[int]) -> None:
        self._orig_bound_mems = orig_bound_mems

    @property
    def perf_interval(self):
        return self._perf_interval

    @property
    def is_running(self) -> bool:
        return self._proc_info.is_running()

    @property
    def group_name(self) -> str:
        return f'{self.name}_{self.pid}'

    @property
    def number_of_threads(self) -> int:
        try:
            return self._proc_info.num_threads()
        except psutil.NoSuchProcess:
            return 0

    @property
    def excess_cpu_flag(self) -> bool:
        return self._excess_cpu_flag

    @excess_cpu_flag.setter
    def excess_cpu_flag(self, flag: bool):
        self._excess_cpu_flag = flag

    @property
    def diff_slack(self) -> float:
        return self._diff_slack

    @diff_slack.setter
    def diff_slack(self, new_diff_slack:float):
        self._diff_slack = new_diff_slack

    @property
    def avg_solorun_data(self) -> Optional[BasicMetric]:
        return self._avg_solorun_data

    @avg_solorun_data.setter
    def avg_solorun_data(self, new_data: BasicMetric) -> None:
        self._avg_solorun_data = new_data

    @property
    def calc_metrics(self) -> Dict[str, Any]:
        return self._calc_metrics

    @property
    def need_profiling(self) -> bool:
        return self._need_profiling

    @need_profiling.setter
    def need_profiling(self, new_value) -> None:
        self._need_profiling = new_value

    @property
    def profile_metrics(self) -> Deque[BasicMetric]:
        return self._profile_metrics

    def collect_metrics(self) -> None:
        logger = logging.getLogger(__name__)
        dst_metric_queue: Deque[BasicMetric] = self._profile_metrics
        src_metric_queue: Deque[BasicMetric] = self._metrics
        logger.debug(f'[collect_metrics] moving metrics from {src_metric_queue} to {dst_metric_queue}')
        while src_metric_queue:
            try:
                item = src_metric_queue.pop()
                dst_metric_queue.appendleft(item)
            except IndexError:
                logger.debug(f'[collect_metrics] {src_metric_queue} is Empty!, len: {len(src_metric_queue)}')
                break

    def update_calc_metrics(self) -> None:
        # TODO: cur_metric can be extended to maintain various resource contention for workloads
        curr_metric_diff = self.calc_metric_diff()
        self._calc_metrics['mem_bw'] = BasicMetric.calc_avg(self._metrics, 30).llc_miss_ps
        self._calc_metrics['mem_bw_diff'] = curr_metric_diff.local_mem_util_ps
        self._calc_metrics['instr_diff'] = curr_metric_diff.instruction_ps
        self._calc_metrics['llc_hr_diff'] = curr_metric_diff.llc_hit_ratio

    def calc_metric_diff(self, core_norm: float = 1) -> MetricDiff:
        logger = logging.getLogger(__name__)
        curr_metric: BasicMetric = self._metrics[0]
        logger.debug(f'[calc_metric_diff] curr_metric: {curr_metric}')
        logger.debug(f'[calc_metric_diff] self._metric: {self._metrics}')
        logger.debug(f'[calc_metric_diff] self._avg_solorun_data: {self._avg_solorun_data}')
        return MetricDiff(curr_metric, self._avg_solorun_data, core_norm, self.diff_slack)

    def all_child_tid(self) -> Tuple[int, ...]:
        try:
            return tuple(chain(
                    (t.id for t in self._proc_info.threads()),
                    *((t.id for t in proc.threads()) for proc in self._proc_info.children(recursive=True))
            ))
        except psutil.NoSuchProcess:
            return tuple()

    def check_gpu_task(self) -> None:
        # gpu_mem_path = '/sys/kernel/debug/nvmap/iovmm/clients'
        try:
            lines = subprocess.check_output("sudo cat /sys/kernel/debug/nvmap/iovmm/clients | awk \'{print $3}\'",
                                            shell=True)
            pids = lines.split().decode().split()
            for pid in pids:
                if pid == str(self.pid):
                    self._is_gpu_task = 1
                    break
            self._is_gpu_task = 0
        except (ValueError, IndexError, AttributeError):
            self._is_gpu_task = 0

    """
    def check_gpu_task(self) -> None:
        gpu_mem_path = Path('/sys/kernel/debug/nvmap/iovmm/clients')
        try:
            with gpu_mem_path.open() as fp:
                line = fp.readline()
                while line is not None:
                    pid = line.split()[2]
                    line = fp.readline()
                    if pid == self.pid:
                        self._is_gpu_task = 1
                        break
        except (ValueError, IndexError):
            self._is_gpu_task = 0
    """

    """
    def cur_socket_id(self) -> int:
        sockets = frozenset(numa_topology.core_to_node[core_id] for core_id in self.bound_cores)
    
        # FIXME: hard coded
        if len(sockets) is not 1:
            raise NotImplementedError(f'Workload spans multiple sockets. {sockets}')
        else:
            return next(iter(sockets))
    """
    def pause(self) -> None:
        self._proc_info.suspend()
        self._perf_info.suspend()

    def resume(self) -> None:
        self._proc_info.resume()
        self._perf_info.resume()
