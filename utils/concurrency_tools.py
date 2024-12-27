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
            self.wait.notify_all()


class ThreadSafeHashMap:
    def __init__(self):
        self.map = {}
        self.lock = ReadWriteLock()

    def __setitem__(self, key, value):
        self.lock.write_lock()
        try:
            self.map[key] = value
        finally:
            self.lock.write_release()

    def __getitem__(self, key):
        self.lock.read_lock()
        try:
            if key in self.map:
                return self.map[key]
            else:
                raise KeyError(f"Key '{key}' not found")
        finally:
            self.lock.read_release()

    def __delitem__(self, key):
        self.lock.write_lock()
        try:
            if key in self.map:
                del self.map[key]
            else:
                raise KeyError(f"Key '{key}' not found")
        finally:
            self.lock.write_release()

    def __contains__(self, key):
        self.lock.read_lock()
        try:
            return key in self.map
        finally:
            self.lock.read_release()

    def __len__(self):
        self.lock.read_lock()
        try:
            return len(self.map)
        finally:
            self.lock.read_release()

    def clear(self):
        self.lock.write_lock()
        try:
            self.map.clear()
        finally:
            self.lock.write_release()

    def keys(self):
        self.lock.read_lock()
        try:
            return list(self.map.keys())
        finally:
            self.lock.read_release()

    def values(self):
        self.lock.read_lock()
        try:
            return list(self.map.values())
        finally:
            self.lock.read_release()

    def items(self):
        self.lock.read_lock()
        try:
            return list(self.map.items())
        finally:
            self.lock.read_release()

    def setdefault(self, key, default):
        self.lock.read_lock()
        try:
            if key in self.map:
                return self.map[key]
        finally:
            self.lock.read_release()

        self.lock.write_lock()
        try:
            if key not in self.map:
                self.map[key] = default(key)
            return self.map[key]
        finally:
            self.lock.write_release()




