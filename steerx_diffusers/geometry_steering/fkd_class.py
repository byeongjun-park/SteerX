"""
Feynman-Kac Diffusion (FKD) steering mechanism implementation.
"""

import torch
from enum import Enum
import numpy as np
from typing import Callable, Optional, Tuple

class PotentialType(Enum):
    DIFF = "diff"
    MAX = "max"
    IS = "is"

class SteerType(Enum):
    Linear = "linear"
    Early = "early"
    Later = "later"


class FKD:
    """
    Implements the FKD steering mechanism. Should be initialized along the diffusion process. .resample() should be invoked at each diffusion timestep.
    See FKD fkd_pipeline_sdxl
    Args:
        potential_type: Type of potential function must be one of PotentialType.
        num_particles: Number of particles to maintain in the population.
        num_steering: Mumber of steering to resample.
        time_steps: Total number of timesteps in the sampling process.
        reward_fn: Function to compute rewards from decoded latents.
        reward_min_value: Minimum value for rewards (default: 0.0). Important for the Max potential type.
        latent_to_decode_fn: Function to decode latents to images, relevant for latent diffusion models (default: identity function).
        device: Device on which computations will be performed (default: CUDA).
        **kwargs: Additional keyword arguments, unused.
    """

    def __init__(
        self,
        *,
        num_particles: int,
        num_steering: int,
        time_steps: int,
        reward_fn: Callable[[torch.Tensor], torch.Tensor],
        reward_min_value: float = 0.001,
        device: torch.device = torch.device('cuda'),
        **kwargs,
    ) -> None:
        # Initialize hyperparameters and functions
        self.num_particles = num_particles
        self.num_steering = num_steering
        self.time_steps = time_steps
        self.steer_type = SteerType('early')
        self.potential_type = PotentialType('max')
        self.lmbda = 10.0
        self.reward_fn = reward_fn

        # Initialize device and population reward state
        self.device = device

        # initial rewards
        self.population_rs = torch.ones(self.num_particles, device=self.device) * reward_min_value
        self.reward_min_value = reward_min_value

    @torch.no_grad()
    def resample(
        self, *, sampling_idx: int, latents: torch.Tensor, x0_preds: torch.Tensor, use_dpm: Optional[bool] = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        resampling_interval = np.linspace(0, 1, self.num_steering + 2)[1:]

        if self.steer_type == SteerType.Linear:
            resampling_interval = np.linspace(0, 1, self.num_steering + 2)[1:]
        elif self.steer_type == SteerType.Early:
            resampling_interval = np.linspace(0.2, 0.4, self.num_steering)
            resampling_interval = np.append(resampling_interval, 1.0)
        elif self.steer_type == SteerType.Later:
            resampling_interval = np.linspace(0.6, 0.8, self.num_steering)
            resampling_interval = np.append(resampling_interval, 1.0)
        else:
            raise ValueError(f"steer_type {self.steer_type} not recognized")

        resampling_interval = (resampling_interval * (self.time_steps - 1)).astype(int)

        if sampling_idx not in resampling_interval:
            return latents, None, None

        rs_candidates, scene_candidates = [], []
        for x0_pred in x0_preds:
            rs, scene = self.reward_fn(x0_pred[None])
            rs_candidates.append(rs)
            scene_candidates.append(scene)

        rs_candidates = torch.stack(rs_candidates, dim=0)

        if sampling_idx == self.time_steps - 1:
            argmax_rewards = torch.argsort(rs_candidates, descending=True)[0]  # We use the same reward_fn in fkds

            return (latents[argmax_rewards].unsqueeze(0),
                    scene_candidates[argmax_rewards],
                    rs_candidates[argmax_rewards],
                )

        # Compute importance weights
        if self.potential_type == PotentialType.MAX:
            before_w = torch.max(rs_candidates, self.population_rs)
            w = torch.exp(self.lmbda * before_w)  # use default lambda value
        elif self.potential_type == PotentialType.DIFF:
            before_w = rs_candidates - self.population_rs
            w = torch.exp(self.lmbda * before_w)  # use default lambda value
        elif self.potential_type == PotentialType.IS:
            before_w = rs_candidates
            w = torch.exp(self.lmbda * before_w)  # use default lambda value
        else:
            raise ValueError(f"potential_type {self.potential_type} not recognized")

        w = torch.clamp(w, 0, 1e10)
        w[torch.isnan(w)] = 0.0

        # Resample indices based on weights
        indices = torch.multinomial(w, num_samples=self.num_particles, replacement=True)

        if use_dpm:
            resampled_latents = latents[indices]
        else:
            resampled_latents = x0_preds[indices]  # Note: we have to forward this to the exact manifold.

        self.population_rs = rs_candidates[indices].clamp(min=self.reward_min_value)

        return resampled_latents, scene_candidates, rs_candidates
