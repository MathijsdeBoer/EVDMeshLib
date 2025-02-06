"""
Poly learning rate scheduler

This is taken from the nnUNet V2 repository:
nnunetv2/training/lr_scheduler/polylr.py
"""

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class PolyLRScheduler(LRScheduler):
    """
    Poly learning rate scheduler
    """

    def __init__(
        self,
        optimizer: Optimizer,
        initial_lr: float,
        max_steps: int,
        exponent: float = 0.9,
        current_step: int = None,
    ) -> None:
        """
        Initialize the poly learning rate scheduler.

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            The optimizer.
        initial_lr : float
            The initial learning rate.
        max_steps : int
            The maximum number of steps.
        exponent : float, optional
            The exponent. The default is 0.9.
        current_step : int, optional
            The current step. The default is None.
        """
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.max_steps = max_steps
        self.exponent = exponent
        self.ctr = 0
        super().__init__(optimizer, current_step if current_step is not None else -1, False)

    def step(self, current_step: int | None = None) -> None:
        """
        Step the learning rate.

        Parameters
        ----------
        current_step : int, optional
            The current step. The default is None.

        Returns
        -------
        None
        """
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1

        new_lr = self.initial_lr * (1 - current_step / self.max_steps) ** self.exponent
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr
