import numpy as np

class BaseProcessor:
    """Base class for all feature processors."""

    name: str = ""
    description: str = ""

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Process a single BGR frame and return processed BGR frame."""
        raise NotImplementedError

    def release(self):
        """Release any resources."""
        pass
