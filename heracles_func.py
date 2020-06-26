
import logging


import paramiko
from enum import IntEnum
from libs.workload import Workload
from libs.isolation.isolators import Isolator, CacheIsolator, CPUFreqThrottleIsolator, SchedIsolator
from libs.isolation.policies import IsolationPolicy
#from isolation_thread import IsolationThread

from typing import Dict, List, Set, Type


class State(IntEnum):
    GROW_CORES = 1
    GROW_LLC = 2
    STOP_GROWTH = 3


class HeraclesFunc:
    """
    Class related to Hearcles controller function
    """

    def __init__(self, interval: float, tail_latency: float, load: float, slo_target: float, file_path: str):
        self._interval: float = interval
        self._tail_latency: float = tail_latency   # 99%tile latency (15secs)
        self._load: float = load                   # QPS (Queries per second during 15 secs.)
        self._slo_target: float = slo_target       # SLO latency in milliseconds (ms)
        self._file_path: str = file_path
        self._last_num_line: int = -1            #
        self._latency_data = None
        self._be_growth = State.STOP_GROWTH
        self._group = None                                              # IsolationPolicy
        self._sub_controllers = []               # IsolationThread
        """
        self._isolator_map: Dict[Type[Isolator], Isolator] = dict((
            (CacheIsolator, CacheIsolator(self._lc_wls, self._be_wls)),
            (CPUFreqThrottleIsolator, CPUFreqThrottleIsolator(self._lc_wls, self._be_wls)),
            (SchedIsolator, SchedIsolator(self._lc_wls, self._be_wls)),
        ))
        """
    @property
    def sub_controllers(self):
        return self._sub_controllers

    def start_sub_controllers(self) -> None:
        """
        Initializing and running sub controllers (CPU-memory sub-controller, DVFS Controller)
        :return:
        """
        logger = logging.getLogger(__name__)

        logger.critical(f'[start_sub_controllers] self._sub_controllers: {self._sub_controllers}')
        for sub_controller in self._sub_controllers:
            logger.critical(f'[start_sub_controllers] sub_controller: {sub_controller}')
            #sub_controller.group = self._group
            sub_controller.heracles = self
            logger.critical(f'[start_sub_controllers] sub_controller.group: {sub_controller.group}, '
                            f'self._group: {self._group}, '
                            f'sub_controller.heracles: {sub_controller.heracles}')
            sub_controller.start()

    def poll_lc_app_latency(self) -> None:
        """
        Polling LC App Latency (read latency information from heracles_latency.txt)
        Calculating that 99 percentile latency
        :return: None
        """
        logger = logging.getLogger(__name__)
        # FIXME: hard-coded for connecting remote host (currently using ip of bc3)
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect('147.46.240.242', username='dcslab', password='dcs%%*#')
        read_cmd = 'cat '+self._file_path
        stdin, stdout, stderr = ssh.exec_command(read_cmd)
        logger.critical(f'[poll_lc_app_latency] ssh: {ssh}')

        read_list = stdout.read().splitlines()

        latency_data: List[float] = []
        start_line = self._last_num_line + 1
        for num_line, line in enumerate(read_list):
            if num_line >= start_line:
                latency_data.append(float(line))

        end_line = len(read_list)
        #logger.critical(f'[poll_lc_app_latency] latency_data: {latency_data}')

        self._latency_data = latency_data
        self._last_num_line = end_line
        self._tail_latency = self.calc_tail(0.99)    # 99 percentile
        logger.critical(f'[poll_lc_app_latency] self._last_num_line: {self._last_num_line}')
        logger.critical(f'[poll_lc_app_latency] self._tail_latency: {self._tail_latency}')

    def calc_tail(self, percentile: float) -> float:
        sorted_lat = sorted(self._latency_data, reverse=True)
        total_reqs = len(self._latency_data)
        req_lat_idx = int(total_reqs*(float(1-percentile)))
        tail_val = sorted_lat[req_lat_idx]
        return tail_val

    def poll_lc_app_load(self) -> float:
        """
        Polling LC App Load (get load information from rabbit MQ)
        Calculating How many requests processed
        :return:
        """
        logger = logging.getLogger(__name__)
        # peak_qps = ????
        total_reqs = len(self._latency_data)
        qps: float = float(total_reqs/self._interval)
        logger.critical(f'[poll_lc_app_load] total_reqs: {total_reqs}, qps: {qps}')
        # FIXME: qps should be adjusted!
        self._load = qps
        return qps

    @staticmethod
    def disable_be_wls(be_workloads: Set[Workload]) -> None:
        """
        Not allowing Best Effort Workloads to run
        Signaling BE App to suspend (SIGSTOP)
        :return:
        """
        for be_wl in be_workloads:
            be_wl.pause()

    @staticmethod
    def enable_be_wls(be_workloads: Set[Workload]) -> None:
        """
        Allowing Best Effort Workloads to run
        Signaling BE App to suspend (SIGCONT)
        :return:
        """
        for be_wl in be_workloads:
            be_wl.resume()

    def disallow_be_growth(self) -> None:
        """
        Setting status not to grow BE Workloads
        :return:
        """
        self._be_growth = State.STOP_GROWTH

    def allow_be_growth(self) -> None:
        """
        Setting status not to grow BE Workloads
        :return:
        """
        self._be_growth = State.GROW_CORES

    @property
    def tail_latency(self) -> float:
        return self._tail_latency

    @property
    def load(self) -> float:
        return self._load

    @property
    def slo_taget(self) -> float:
        return self._slo_target

    @property
    def state(self) -> State:
        return self._be_growth

    @state.setter
    def state(self, new_state: State):
        self._be_growth = new_state

    @property
    def group(self) -> IsolationPolicy:
        return self._group

    @group.setter
    def group(self, new_group):
        self._group = new_group
