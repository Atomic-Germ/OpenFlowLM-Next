# Traces: OPEN-DIFFUSION-RESOLUTIONS, OPEN-DIFFUSION-SCHEDULE (canonical spec: specs/open-diffusion/spec.md)
"""The host setup of a generation: which sizes run, and the scheduler's sigmas."""
from __future__ import annotations

import numpy as np
import pytest

import klein_pipeline as kp

# FlowMatchEulerDiscreteScheduler.set_timesteps(sigmas=linspace(1, 1/4, 4), mu=...) in
# diffusers 0.37 with klein's scheduler_config.json (utilities/dit-ref/capture_pipeline_inputs.py)
DIFFUSERS_SIGMAS = {
    512: [1.0, 0.9580853581428528, 0.8839818835258484, 0.7174965739250183, 0.0],
    1024: [1.0, 0.9673840403556824, 0.908143937587738, 0.7671999335289001, 0.0],
}


@pytest.mark.parametrize("R", [512, 1024])
def test_supported_sizes(R):
    assert kp.check_resolution(R) is None


@pytest.mark.parametrize("R, why", [
    (768, "2304 image tokens is not a multiple of 512"),
    (520, "must be a positive multiple of 16"),
    (0, "must be a positive multiple of 16"),
])
def test_other_sizes_are_refused_by_name(R, why):
    assert why in kp.check_resolution(R)


def test_the_plan_refuses_an_unsupported_size():
    with pytest.raises(AssertionError, match="not a multiple of 512"):
        kp.plan(768)


@pytest.mark.parametrize("R", [512, 1024])
def test_sigmas_match_diffusers_bit_for_bit(R):
    got = kp.sigmas(R)
    assert got.dtype == np.float32
    assert got.tolist() == np.asarray(DIFFUSERS_SIGMAS[R], np.float32).tolist()


def test_euler_dt_is_stored_as_fp32_in_the_parameter_run():
    sig = kp.sigmas(512)
    p = kp.dt_params(sig).view(np.uint16)
    for s in range(kp.STEPS):
        o = (s * kp.DT_SLOT + 1) * kp.EL
        assert p[o:o + 2].view(np.float32)[0] == np.float32(sig[s + 1] - sig[s])
