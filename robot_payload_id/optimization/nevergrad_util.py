from pathlib import Path
from typing import List, Union

import nevergrad as ng


class NevergradLossLogger:
    """Logs losses into a file during optimization.
    NOTE: This is a minimal version of nevergrad's `ParametersLogger` that only logs
    losses to minimize overhead.

    Parameters
    ----------
    filepath: str or pathlib.Path
        the path to dump data to

    Example
    -------

    .. code-block:: python

        logger = NevergradLossLogger(filepath)
        optimizer.register_callback("tell",  logger)
        optimizer.minimize()
        list_of_dict_of_data = logger.load()

    Note
    ----
    Arrays are converted to lists
    """

    def __init__(self, filepath: Union[str, Path]) -> None:
        self._filepath = Path(filepath)

    def __call__(
        self,
        optimizer: ng.optimization.Optimizer,
        candidate: ng.parametrization.parameter.Parameter,
        loss: ng.typing.FloatLoss,
    ) -> None:
        with self._filepath.open("a") as f:
            f.write(str(loss) + "\n")

    def load(self) -> List[float]:
        """Loads data from the log file"""
        losses: List[float] = []
        if self._filepath.exists():
            with self._filepath.open("r") as f:
                for line in f.readlines():
                    losses.append(float(line))
        return losses
