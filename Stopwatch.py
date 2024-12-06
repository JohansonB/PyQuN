import threading
import uuid
import time
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
        return None, None  # Thread not found

    def start_timer(self, name) -> None:
        time_struc = TimingStruct()
        thread_id = threading.get_ident()  # Get the calling thread ID
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



class StopWatchManager:
    _instance = None  # Class variable to store the singleton instance
    _creation_lock = threading.Lock()  # A lock object to ensure thread-safe singleton creation
    _dic_lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._creation_lock:  # Acquire the lock before checking/creating the instance
            if cls._instance is None:
                cls._instance = super(StopWatchManager, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        # Initialize the StopWatchManager instance (only the first time)
        if not hasattr(self, '_initialized'):  # To avoid re-initialization
            self._initialized = True
            # Initialization logic here
            self.timers = {}  #Dictionary which maps ids of Strategies to there dictionary of name, timer assosicaitons

    def start_timer(self, id:uuid.UUID, dataset:str, name:str) -> None:
        time_struc = TimingStruct()
        time_struc.start = time.process_time()
        with self._dic_lock:
            if id not in self.timers:
                self.timers[id] = {}
            if dataset not in self.timers[id]:
                self.timers[id][dataset] = {}

            self.timers[id][dataset][name] = time_struc


    def stop_timer(self, id:uuid.UUID, dataset:str, name:str) -> None:
        stop_time = time.process_time()
        with self._dic_lock:
            cur_struct = self.timers[id][dataset][name]
            cur_struct.end = stop_time
            cur_struct.elapsed = stop_time - cur_struct.start
if __name__ == "__main__":
    # Usage example:
    manager1 = StopWatchManager()
    manager2 = StopWatchManager()

    print(manager1 is manager2)
