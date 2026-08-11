from __future__ import annotations

import logging

import torch

import comfy.samplers
from comfy.utils import model_trange


LOG = logging.getLogger("h3_cab_sampler")


def _safe_scalar(value, epsilon=1.0e-12):
    return torch.where(
        value.abs() < epsilon,
        value.sign() * epsilon + (value == 0).to(value.dtype) * epsilon,
        value,
    )


@torch.no_grad()
def sample_h3_cab(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    order=2,
    theta=0.2,
):
    """CAB-2/3 adapted to Comfy's denoised-output and H3 NestedTensor API."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    sample_dtype = x.dtype

    last_sample = None
    last_velocity = None
    previous_epsilon = None
    previous_previous_epsilon = None
    previous_h_lambda = None
    previous_previous_h_lambda = None

    for index in model_trange(len(sigmas) - 1, disable=disable):
        sigma = sigmas[index].to(device=x.device, dtype=torch.float32)
        sigma_next = sigmas[index + 1].to(device=x.device, dtype=torch.float32)
        current = x.float()
        denoised = model(current, sigma * s_in, **extra_args).float()
        velocity = (current - denoised) / _safe_scalar(sigma)
        if callback is not None:
            callback(
                {
                    "x": current,
                    "i": index,
                    "sigma": sigma,
                    "sigma_hat": sigma,
                    "denoised": denoised,
                }
            )

        delta = sigma_next - sigma
        alpha = 1.0 - sigma
        alpha_next = 1.0 - sigma_next
        alpha_safe = _safe_scalar(alpha)
        alpha_next_safe = _safe_scalar(alpha_next)
        lambda_current = sigma / alpha_safe
        lambda_next = sigma_next / alpha_next_safe
        h_lambda = lambda_next - lambda_current
        epsilon_current = current + velocity * alpha
        y_current = current / alpha_safe

        if index == 0:
            next_sample = current + velocity * delta
        elif index == 1:
            # CAB's flow bootstrap is Euler followed by Adams-Bashforth 2.
            next_sample = current + (velocity * 1.5 - last_velocity * 0.5) * delta

            sigma_previous = sigmas[index - 1].to(
                device=x.device, dtype=torch.float32
            )
            alpha_previous = 1.0 - sigma_previous
            lambda_previous = sigma_previous / _safe_scalar(alpha_previous)
            epsilon_previous = last_sample + last_velocity * alpha_previous
            previous_previous_epsilon = epsilon_previous
            previous_epsilon = epsilon_current
            previous_previous_h_lambda = lambda_current - lambda_previous
            previous_h_lambda = h_lambda
            # These two bootstrap buffers are never read after CAB history is
            # initialized. Drop them before the third model call.
            last_sample = None
            last_velocity = None
        else:
            h_previous_safe = _safe_scalar(previous_h_lambda)
            if order == 2:
                step_ratio = h_lambda / h_previous_safe
                direction = (
                    epsilon_current * (1.0 + 0.5 * step_ratio)
                    - previous_epsilon * (0.5 * step_ratio)
                )
            else:
                h_previous_previous_safe = _safe_scalar(
                    previous_previous_h_lambda
                )
                h_sum_safe = _safe_scalar(
                    previous_h_lambda + previous_previous_h_lambda
                )
                beta0 = (
                    h_lambda.square() / 3.0
                    + 0.5
                    * h_lambda
                    * (2.0 * previous_h_lambda + previous_previous_h_lambda)
                    + previous_h_lambda
                    * (previous_h_lambda + previous_previous_h_lambda)
                ) / (h_previous_safe * h_sum_safe)
                beta1 = -h_lambda * (
                    2.0 * h_lambda
                    + 3.0 * previous_h_lambda
                    + 3.0 * previous_previous_h_lambda
                ) / (6.0 * h_previous_safe * h_previous_previous_safe)
                beta2 = h_lambda * (
                    2.0 * h_lambda + 3.0 * previous_h_lambda
                ) / (6.0 * h_previous_previous_safe * h_sum_safe)
                direction = (
                    epsilon_current * beta0
                    + previous_epsilon * beta1
                    + previous_previous_epsilon * beta2
                )

            y_predictor = y_current + direction * h_lambda
            history_ratio = previous_h_lambda / _safe_scalar(
                previous_previous_h_lambda
            )
            epsilon_extrapolated = (
                previous_epsilon * (1.0 + history_ratio)
                - previous_previous_epsilon * history_ratio
            )
            defect = epsilon_current - epsilon_extrapolated
            y_next = y_predictor + defect * (float(theta) * h_lambda)
            next_sample = y_next * alpha_next_safe

            previous_previous_epsilon = previous_epsilon
            previous_epsilon = epsilon_current
            previous_previous_h_lambda = previous_h_lambda
            previous_h_lambda = h_lambda

        if index == 0:
            last_velocity = velocity
            last_sample = current
        x = next_sample

    return x.to(dtype=sample_dtype)


class H3CABSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "order": (["CAB-2", "CAB-3"], {"default": "CAB-2"}),
                "theta": (
                    "FLOAT",
                    {
                        "default": 0.20,
                        "min": 0.0,
                        "max": 1.5,
                        "step": 0.05,
                    },
                ),
            }
        }

    RETURN_TYPES = ("SAMPLER",)
    FUNCTION = "get_sampler"
    CATEGORY = "sampling/minimax_h3/optimization"
    EXPERIMENTAL = True
    DESCRIPTION = (
        "Training-free Corrected Adams-Bashforth sampler adapted for MiniMax "
        "H3's packed audio/video NestedTensor. Reuses past velocity estimates "
        "without extra model calls. Start with CAB-2, theta 0.20, 6-10 stock "
        "simple sigmas."
    )

    def get_sampler(self, order="CAB-2", theta=0.20):
        numeric_order = 2 if order == "CAB-2" else 3
        sampler = comfy.samplers.KSAMPLER(
            sample_h3_cab,
            extra_options={"order": numeric_order, "theta": float(theta)},
        )
        LOG.info("H3 CAB sampler configured: order=%d theta=%.3f", numeric_order, theta)
        return (sampler,)
