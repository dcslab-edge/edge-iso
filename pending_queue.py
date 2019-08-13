# coding: UTF-8

import logging
from collections import defaultdict
from typing import DefaultDict, Dict, List, Sized, Tuple, Type

from libs.isolation.policies import IsolationPolicy
from libs.workload import Workload


class PendingQueue(Sized):
    def __init__(self, policy_type: Type[IsolationPolicy]) -> None:
        self._policy_type: Type[IsolationPolicy] = policy_type

        # self._bg_q: Dict[Tuple[int, ...], Workload] = dict()
        self._be_q_list: List[Dict[Tuple[int, ...], Workload]] = list(dict())
        self._lc_q_list: List[Dict[Tuple[int, ...], Workload]] = list(dict())
        self._ready_queues: DefaultDict[int, List[Workload]] = defaultdict(list)    # ready_queue per socket
        self._pending_list: List[IsolationPolicy] = list()                          # pending workload pairs
        self._first_pending: bool = True

    def __len__(self) -> int:
        pending_queue_len = len(tuple(
            filter(lambda x: x.check_lc_wls_metrics() and x.check_be_wls_metrics(), self._pending_list)))
        #print(f'pending_queue_len: {pending_queue_len}, self._pending_list: {self._pending_list}')
        return pending_queue_len
        #return len(tuple(
        #        filter(lambda x: len(x.foreground_workload.metrics) > 0 and x.check_bg_wls_metrics(),
        #               self._pending_list)))

    def add(self, workload: Workload, max_workloads: int) -> None:
        logger = logging.getLogger('monitoring.pending_queue')
        logger.info(f'{workload} is ready for active')

        # FIXME: hard coded : Workload always locate on the socket 0
        # ready_queue = self._ready_queue[workload.cur_socket_id()]
        ready_queue = self._ready_queues[0]
        ready_queue.append(workload)

        # Here, max_workloads is the `bench` numbers specified in bench_launcher's experiments config.json
        print(f'ready_queue: {ready_queue}, len_ready_queue: {len(ready_queue)}, max_workloads: {max_workloads}')
        if len(ready_queue) == max_workloads and self._first_pending:
            lc_wls = set()
            be_wls = set()
            for workload in ready_queue:
                if workload.wl_type == 'LC':
                    lc_wls.add(workload)
                else:
                    be_wls.add(workload)
            new_group = self._policy_type(lc_wls, be_wls)
            self._pending_list.append(new_group)
            self._ready_queues[0] = list()
            self._first_pending = False
        # elif not self._first_pending:
            # Add additional workload(s) into current group after

    def pop(self) -> IsolationPolicy:
        if len(self) is 0:
            raise IndexError(f'{self} is empty')
        return self._pending_list.pop()
