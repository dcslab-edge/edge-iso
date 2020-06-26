
import time
import logging
from heracles_func import State


class CPUMemController:
    """
    input:
    output: core and LLC allocation to the LC and BE workloads
    """
    _MAX_MEM_BANDWIDTH_PS = 68 * 1024 * 1024 * 1024     # MemBW of Xeon Server
    _DRAM_LIMIT = _MAX_MEM_BANDWIDTH_PS * 0.9           # 90% of peak dram bw

    def __init__(self):
        self._state = State.STOP_GROWTH
        self._bw_derivative = 0
        self._cur_total_dram_bw = 0

    def measure_dram_bw(self):
        # update
        #self._cur_dram_bw =
        return

    def predicted_total_bw(self) -> float:
        return self.lc_bw_model() + self.be_bw() + self._bw_derivative

    def can_grow_be(self) -> bool:
        if self._state is not State.STOP_GROWTH:
            return True
        else:
            return False

    def grow_cache_for_be(self):
        return

    def be_benefit(self):
        return

    def rollback(self):
        return

    def lc_bw_model(self):
        return

    def be_bw(self):
        return

    def be_bw_per_core(self):
        return

    def run(self) -> None:
        while True:
            self.measure_dram_bw()
            total_bw = self._cur_total_dram_bw
            if total_bw > self._DRAM_LIMIT:
                overage = total_bw - self._DRAM_LIMIT
                # be_cores.Remove(overage/BeBwPerCore())
                continue
            if not self.can_grow_be():
                continue
            if self._state is State.GROW_LLC:
                if self.predicted_total_bw() > self._DRAM_LIMIT:
                    self._state = State.GROW_CORES
                else:
                    self.grow_cache_for_be()
                    self.measure_dram_bw()
                    if self._bw_derivative >= 0:
                        self.rollback()
                        self._state = State.GROW_CORES
                    if not self.be_benefit():
                        self._state = State.GROW_CORES
            elif self._state is State.GROW_CORES:
                needed = self.lc_bw_model() + self.be_bw() + self.be_bw_per_core()
                if needed > self._DRAM_LIMIT:
                    self._state = State.GROW_LLC
                else:
                    #be_cores.Add(1)
            time.sleep(2)
