import uuid


class IDFactory:
    def __init__(self, seed: str = None):
        """Initialize the IDFactory with a seed and counters."""
        self._seed = seed
        self._uuid_counter = 0
        self._id_counter = 0

    def generate_uuid(self) -> uuid.UUID:
        """Generate a new deterministic UUID based on the seed and an increasing counter."""
        input_string = f"{self._seed}:{self._uuid_counter}"
        self._uuid_counter += 1
        return uuid.uuid5(uuid.NAMESPACE_DNS, input_string)

    def generate_id(self) -> int:
        """Generate a new incremental integer ID."""
        ret = self._id_counter
        self._id_counter += 1
        return ret
