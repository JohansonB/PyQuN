from threading import Lock, Condition
#read write lock prioritizing raders
class ReadWriteLock:

    def __init__(self):
        self.flag_lock = Lock()
        self.wait = Condition(self.flag_lock)
        self.write_waits = 0
        self.readers = 0
        self.writers = 0

    def read_lock(self):
        with self.flag_lock:
            while self.write_waits > 0 or self.writers > 0:
                self.wait.wait()
            self.readers += 1

    def read_release(self):
        with self.flag_lock:
            self.readers -= 1
            # Notify only if writers are waiting, to prioritize writers after readers release
            if self.write_waits > 0:
                self.wait.notify_all()

    def write_lock(self):
        with self.flag_lock:
            self.write_waits += 1
            while self.readers > 0 or self.writers > 0:
                self.wait.wait()
            self.write_waits -= 1
            self.writers += 1

    def write_release(self):
        with self.flag_lock:
            self.writers -= 1
            # Notify all threads to let readers and writers proceed
            self.wait.notify_all()








