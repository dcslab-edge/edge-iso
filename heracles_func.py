
import logging
from typing import Dict, List, Set


class HeraclesFunc:
    """
    Class related to Hearcles controller function
    """

    def __init__(self, tail_latency: float, load: float, slo_target: float):
        self._tail_latency = tail_latency   # 99%tile latency (15secs)
        self._load = load                   # QPS (Queries per second during 15 secs.)
        self._slo_target = slo_target       # SLO latency in milliseconds (ms)

    def PollLCAppLatency(self) -> None:
        """
        Polling LC App Latency (get latency information from rabbit MQ)
        Calculating that 99 percentile latency
        :return: None
        """

    def PollLCAppLoad(self) -> None:
        """
        Polling LC App Load (get load information from rabbit MQ)
        Calculating How many requests processed
        :return:
        """

    def DisableBE(self) -> None:
        """
        Not allowing Best Effort Workloads to run
        Signaling BE App to suspend (SIGSTOP)
        :return:
        """

    def EnableBE(self) -> None:
        """
        Allowing Best Effort Workloads to run
        Signaling BE App to suspend (SIGCONT)
        :return:
        """

    def DisallowBEGrowth(self) -> None:
        """
        Setting status not to grow BE Workloads
        :return:
        """

