# Modified from OpenAI's diffusion repos
#     GLIDE: https://github.com/openai/glide-text2im/blob/main/glide_text2im/gaussian_diffusion.py
#     ADM:   https://github.com/openai/guided-diffusion/blob/main/guided_diffusion
#     IDDPM: https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py

from abc import ABC, abstractmethod

import numpy as np
import torch as th
import torch.distributed as dist


def create_named_schedule_sampler(name, diffusion):
    """    Create a ScheduleSampler from a library of pre-defined samplers.

    Args:
        name (str): The name of the sampler.
        diffusion (object): The diffusion object to sample for.

    Returns:
        ScheduleSampler: An instance of ScheduleSampler based on the input name.

    Raises:
        NotImplementedError: If the input name does not match any pre-defined samplers.
    """
    if name == "uniform":
        return UniformSampler(diffusion)
    elif name == "loss-second-moment":
        return LossSecondMomentResampler(diffusion)
    else:
        raise NotImplementedError(f"unknown schedule sampler: {name}")


class ScheduleSampler(ABC):
    """
    A distribution over timesteps in the diffusion process, intended to reduce
    variance of the objective.
    By default, samplers perform unbiased importance sampling, in which the
    objective's mean is unchanged.
    However, subclasses may override sample() to change how the resampled
    terms are reweighted, allowing for actual changes in the objective.
    """

    @abstractmethod
    def weights(self):
        """        Get a numpy array of weights, one per diffusion step.

        The weights needn't be normalized, but must be positive.

        Returns:
            numpy.ndarray: An array of weights, one per diffusion step.
        """

    def sample(self, batch_size, device):
        """        Importance-sample timesteps for a batch.

        This function importance-samples timesteps for a batch using the given batch size and torch device.

        Args:
            batch_size (int): The number of timesteps.
            device (torch.device): The torch device to save to.

        Returns:
            tuple: A tuple containing:
                - timesteps (torch.Tensor): A tensor of timestep indices.
                - weights (torch.Tensor): A tensor of weights to scale the resulting losses.
        """
        w = self.weights()
        p = w / np.sum(w)
        indices_np = np.random.choice(len(p), size=(batch_size,), p=p)
        indices = th.from_numpy(indices_np).long().to(device)
        weights_np = 1 / (len(p) * p[indices_np])
        weights = th.from_numpy(weights_np).float().to(device)
        return indices, weights


class UniformSampler(ScheduleSampler):
    def __init__(self, diffusion):
        """        Initialize the object with the given diffusion.

        Args:
            diffusion: The diffusion to be initialized.
        """

        self.diffusion = diffusion
        self._weights = np.ones([diffusion.num_timesteps])

    def weights(self):
        """        Return the weights associated with the object.

        Returns:
            object: The weights associated with the object.
        """

        return self._weights


class LossAwareSampler(ScheduleSampler):
    def update_with_local_losses(self, local_ts, local_losses):
        """        Update the reweighting using losses from a model.

        Call this method from each rank with a batch of timesteps and the corresponding losses for each of those timesteps.
        This method will perform synchronization to make sure all of the ranks maintain the exact same reweighting.

        Args:
            local_ts (Tensor): An integer Tensor of timesteps.
            local_losses (Tensor): A 1D Tensor of losses.
        """
        batch_sizes = [
            th.tensor([0], dtype=th.int32, device=local_ts.device)
            for _ in range(dist.get_world_size())
        ]
        dist.all_gather(
            batch_sizes,
            th.tensor([len(local_ts)], dtype=th.int32, device=local_ts.device),
        )

        # Pad all_gather batches to be the maximum batch size.
        batch_sizes = [x.item() for x in batch_sizes]
        max_bs = max(batch_sizes)

        timestep_batches = [th.zeros(max_bs).to(local_ts) for bs in batch_sizes]
        loss_batches = [th.zeros(max_bs).to(local_losses) for bs in batch_sizes]
        dist.all_gather(timestep_batches, local_ts)
        dist.all_gather(loss_batches, local_losses)
        timesteps = [
            x.item() for y, bs in zip(timestep_batches, batch_sizes) for x in y[:bs]
        ]
        losses = [x.item() for y, bs in zip(loss_batches, batch_sizes) for x in y[:bs]]
        self.update_with_all_losses(timesteps, losses)

    @abstractmethod
    def update_with_all_losses(self, ts, losses):
        """        Update the reweighting using losses from a model.

        Sub-classes should override this method to update the reweighting
        using losses from the model. This method directly updates the reweighting
        without synchronizing between workers. It is called by update_with_local_losses from all
        ranks with identical arguments. Thus, it should have deterministic behavior to maintain state across workers.

        Args:
            ts (list): A list of int timesteps.
            losses (list): A list of float losses, one per timestep.
        """


class LossSecondMomentResampler(LossAwareSampler):
    def __init__(self, diffusion, history_per_term=10, uniform_prob=0.001):
        """        Initialize the object with diffusion, history_per_term, and uniform_prob.

        Args:
            diffusion: The diffusion object.
            history_per_term (int): The number of history per term.
            uniform_prob (float): The uniform probability.
        """

        self.diffusion = diffusion
        self.history_per_term = history_per_term
        self.uniform_prob = uniform_prob
        self._loss_history = np.zeros(
            [diffusion.num_timesteps, history_per_term], dtype=np.float64
        )
        self._loss_counts = np.zeros([diffusion.num_timesteps], dtype=np.int)

    def weights(self):
        """        Calculate the weights for the loss history.

        If the model is not warmed up, it returns an array of ones with the same length as the number of timesteps.
        Otherwise, it calculates the weights based on the root mean square of the loss history and normalizes them.

        Returns:
            numpy.ndarray: An array of weights for the loss history.
        """

        if not self._warmed_up():
            return np.ones([self.diffusion.num_timesteps], dtype=np.float64)
        weights = np.sqrt(np.mean(self._loss_history ** 2, axis=-1))
        weights /= np.sum(weights)
        weights *= 1 - self.uniform_prob
        weights += self.uniform_prob / len(weights)
        return weights

    def update_with_all_losses(self, ts, losses):
        """        Update the loss history with all losses for given time steps.

        This method updates the loss history with all the losses for the given time steps. If the number of losses for a
        specific time step exceeds the history_per_term, the oldest loss term is shifted out.

        Args:
            ts (list): A list of time steps.
            losses (list): A list of loss values corresponding to the time steps.
        """

        for t, loss in zip(ts, losses):
            if self._loss_counts[t] == self.history_per_term:
                # Shift out the oldest loss term.
                self._loss_history[t, :-1] = self._loss_history[t, 1:]
                self._loss_history[t, -1] = loss
            else:
                self._loss_history[t, self._loss_counts[t]] = loss
                self._loss_counts[t] += 1

    def _warmed_up(self):
        """        Check if the model has been warmed up.

        This method checks if the model has been warmed up by comparing the loss counts with the history per term.

        Returns:
            bool: True if the model has been warmed up, False otherwise.
        """

        return (self._loss_counts == self.history_per_term).all()
