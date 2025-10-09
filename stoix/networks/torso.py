from typing import Sequence

import chex
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.linen.initializers import Initializer, orthogonal

from stoix.networks.layers import NoisyLinear
from stoix.networks.utils import parse_activation_fn


class MLPTorso(nn.Module):
    """MLP torso."""

    layer_sizes: Sequence[int]
    activation: str = "relu"
    use_layer_norm: bool = False
    kernel_init: Initializer = orthogonal(np.sqrt(2.0))
    activate_final: bool = True

    @nn.compact
    def __call__(self, observation: chex.Array) -> chex.Array:
        """Forward pass."""
        x = observation
        for layer_size in self.layer_sizes:
            x = nn.Dense(
                layer_size, kernel_init=self.kernel_init, use_bias=not self.use_layer_norm
            )(x)
            if self.use_layer_norm:
                x = nn.LayerNorm()(x)
            if self.activate_final or layer_size != self.layer_sizes[-1]:
                x = parse_activation_fn(self.activation)(x)
        return x


class NoisyMLPTorso(nn.Module):
    """MLP torso using NoisyLinear layers instead of standard Dense layers."""

    layer_sizes: Sequence[int]
    activation: str = "relu"
    use_layer_norm: bool = False
    kernel_init: Initializer = orthogonal(np.sqrt(2.0))
    activate_final: bool = True
    sigma_zero: float = 0.5

    @nn.compact
    def __call__(self, observation: chex.Array) -> chex.Array:
        x = observation
        for layer_size in self.layer_sizes:
            x = NoisyLinear(
                layer_size, sigma_zero=self.sigma_zero, use_bias=not self.use_layer_norm
            )(x)
            if self.use_layer_norm:
                x = nn.LayerNorm()(x)
            if self.activate_final or layer_size != self.layer_sizes[-1]:
                x = parse_activation_fn(self.activation)(x)
        return x


class CNNTorso(nn.Module):
    """2D CNN torso. Expects input of shape (batch, height, width, channels).
    After this torso, the output is flattened and put through an MLP of
    hidden_sizes."""

    channel_sizes: Sequence[int]
    kernel_sizes: Sequence[int]
    strides: Sequence[int]
    activation: str = "relu"
    use_layer_norm: bool = False
    kernel_init: Initializer = orthogonal(np.sqrt(2.0))
    channel_first: bool = False
    hidden_sizes: Sequence[int] = (256,)

    @nn.compact
    def __call__(self, observation: chex.Array) -> chex.Array:
        """Forward pass."""
        x = observation

        # If there is a batch of sequences of images
        if observation.ndim > 4:
            return nn.batch_apply.BatchApply(self.__call__)(observation)

        # If the input is in the form of [B, C, H, W], we need to transpose it to [B, H, W, C]
        if self.channel_first:
            x = x.transpose((0, 2, 3, 1))

        # Convolutional layers
        for channel, kernel, stride in zip(self.channel_sizes, self.kernel_sizes, self.strides):
            x = nn.Conv(
                channel, (kernel, kernel), (stride, stride), use_bias=not self.use_layer_norm
            )(x)
            if self.use_layer_norm:
                x = nn.LayerNorm(reduction_axes=(-3, -2, -1))(x)
            x = parse_activation_fn(self.activation)(x)

        # Flatten
        x = x.reshape(*observation.shape[:-3], -1)

        # MLP layers
        x = MLPTorso(
            layer_sizes=self.hidden_sizes,
            activation=self.activation,
            use_layer_norm=self.use_layer_norm,
            kernel_init=self.kernel_init,
            activate_final=True,
        )(x)

        return x


class MazeTorso(nn.Module):
    """Torso for JaxMaze environment."""

    num_objects: int
    num_actions: int
    num_features: int

    @nn.compact
    def __call__(self, observation: dict[str, jnp.ndarray]) -> jnp.ndarray:
        """Forward pass."""
        image_shape = observation["image"].shape
        batch_shape = image_shape[:-2]
        values, num_values = zip(
            *[
                (observation["image"], self.num_objects + 2),
                (observation["position"][..., 0], image_shape[-2]),
                (observation["position"][..., 1], image_shape[-1]),
                (observation["task_w"].argmax(axis=-1), self.num_objects),
                (observation["direction"], 4),
                (observation["prev_action"], self.num_actions + 1),
            ]
        )
        offsets = np.cumsum((0,) + num_values)
        values = [
            x.reshape(*batch_shape, -1) + offset
            for x, offset in zip(values, offsets[:-1], strict=True)
        ]
        x = jnp.concat(values, axis=-1)
        x = nn.Embed(num_embeddings=offsets[-1].item(), features=self.num_features)(x)
        return x.reshape(*batch_shape, -1)
