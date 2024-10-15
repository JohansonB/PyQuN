import threading
import uuid


class StopWatchManager:
    _instance = None  # Class variable to store the singleton instance
    _creation_lock = threading.Lock()  # A lock object to ensure thread-safe singleton creation

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
            self.tTimers = {}  # Example: Dictionary to store timers

    def start_timer(self, id:uuid.UUID, name:str) -> None:
        pass

    def stop_timer(self, id:uuid.UUID, name:str) -> None:
        pass
if __name__ == "__main__":
    # Usage example:
    manager1 = StopWatchManager()
    manager2 = StopWatchManager()

    print(manager1 is manager2)
