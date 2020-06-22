# coding: UTF-8

import json
from pathlib import Path
from typing import Dict

from ..metric_container.basic_metric import BasicMetric
from ..utils.machine_type import MachineChecker, NodeType

data_map: Dict[str, BasicMetric] = dict()

CUR_NODE_TYPE = MachineChecker.get_node_type()


def _init() -> None:
    for data in Path(__file__).parent.iterdir():  # type: Path
        if data.match('*.json'):
            metric = json.loads(data.read_text())
            """
            if CUR_NODE_TYPE == NodeType.IntegratedGPU:
                item = BasicMetric(metric['llc_references'],
                                   metric['llc_misses'],
                                   metric['instructions'],
                                   metric['cycles'],
                                   metric['gpu_core_util'],
                                   metric['gpu_core_freq'],
                                   metric['gpu_emc_util'],
                                   metric['gpu_emc_freq'],
                                   1000)
            """
            #if CUR_NODE_TYPE == NodeType.CPU:
            item = BasicMetric(metric['llc_references'],
                               metric['llc_misses'],
                               metric['instructions'],
                               metric['cycles'],
                               metric['stall_cycles'],
                               metric['intra_coh'],
                               metric['inter_coh'],
                               metric['wall_cycles'],
                               metric['llc_size'],
                               metric['local_mem'],
                               metric['remote_mem'],
                               1000)

            data_map[metric['name']] = item


_init()
