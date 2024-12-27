import threading
import psutil

class TimingStruct:
    def __init__(self):
        self.start = None
        self.end = None
        self.elapsed = None

class Stopwatch:
    def __init__(self):
        self.timers = {}

    def _get_thread_cpu_times(self, thread_id):
        # Get CPU times for the thread with the specified ID
        for thread_info in psutil.Process().threads():
            if thread_info.id == thread_id:
                return thread_info.user_time, thread_info.system_time
        return None, None

    def start_timer(self, name) -> None:
        time_struc = TimingStruct()
        thread_id = threading.get_ident()
        time_struc.start = sum(self._get_thread_cpu_times(thread_id))
        self.timers[name] = time_struc

    def stop_timer(self, name: str) -> None:
        cur_struct = self.timers[name]
        cur_struct.end = sum(self._get_thread_cpu_times(threading.get_ident()))
        cur_struct.elapsed = (
            cur_struct.end - cur_struct.start
            if cur_struct.start is not None and cur_struct.end is not None
            else None
        )

    def merge(self, other: 'Stopwatch') -> None:
        for name, ts in other.timers.items():
            if name in self.timers:
                self.timers[name].elapsed += ts.elapsed
            else:
                self.timers[name] = ts

    def get_time(self, name):
        return self.timers[name].elapsed


