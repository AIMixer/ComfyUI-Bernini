import torch

from engine.wanvideo.schedulers.ksampler_bridge import (
    attach_scheduler_timesteps,
    flowmatch_sgm_uniform_sigmas,
    normalize_wanvideo_schedulerv2_inputs,
    resolve_bernini_scheduler,
    resolve_sigma_schedule,
)


def test_resolve_bernini_scheduler_maps_ksampler_names():
    assert resolve_bernini_scheduler("dpmpp_2m_sde") == "dpm++_sde"
    assert resolve_bernini_scheduler("uni_pc") == "unipc"


def test_resolve_bernini_scheduler_rejects_unknown_sampler():
    try:
        resolve_bernini_scheduler("heun")
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "no Bernini/Wan equivalent" in str(exc)


def test_flowmatch_sgm_uniform_sigmas_shape():
    sigmas = flowmatch_sgm_uniform_sigmas(4, torch.device("cpu"))
    assert sigmas.shape == (5,)
    assert sigmas[-1].item() == 0.0


def test_resolve_sigma_schedule_sgm_uniform_without_model():
    sigmas = resolve_sigma_schedule(
        sigmas=None,
        scheduler="sgm_uniform",
        steps=4,
        device=torch.device("cpu"),
    )
    assert sigmas is not None
    assert len(sigmas) == 5


def test_normalize_legacy_wanvideo_scheduler_widgets_high():
    out = normalize_wanvideo_schedulerv2_inputs("euler", 8, 5, 0, 4, False)
    assert out == ("euler", "simple", 8, 5.0, 0, 4, False)


def test_normalize_legacy_wanvideo_scheduler_widgets_low():
    out = normalize_wanvideo_schedulerv2_inputs("euler", 8, 5, 4, 1000, False)
    assert out == ("euler", "simple", 8, 5.0, 4, 1000, False)


def test_normalize_keeps_v2_widgets():
    out = normalize_wanvideo_schedulerv2_inputs("euler", "simple", 8, 5, 0, 4, False)
    assert out == ("euler", "simple", 8, 5.0, 0, 4, False)


def test_normalize_coerces_invalid_shift():
    out = normalize_wanvideo_schedulerv2_inputs("euler", "simple", 4, 0, 0, 2, False)
    assert out == ("euler", "simple", 4, 5.0, 0, 2, False)


def test_normalize_partial_legacy_with_simple_combo():
    out = normalize_wanvideo_schedulerv2_inputs("euler", "simple", 5, 0, 4, False, False)
    assert out == ("euler", "simple", 8, 5.0, 0, 4, False)


def test_attach_scheduler_timesteps_preserves_sgm_uniform_high_sigmas():
    import copy

    from engine.wanvideo.schedulers import get_scheduler
    from engine.wanvideo.schedulers.ksampler_bridge import resolve_sigma_schedule

    device = torch.device("cpu")
    sig = resolve_sigma_schedule(sigmas=None, scheduler="sgm_uniform", steps=4, device=device)
    ss, ts, _, _ = get_scheduler("dpm++_sde", 4, 0, 2, 5.0, device, sigmas=sig)
    expected_sigmas = ss.sigmas.clone()
    dc = copy.deepcopy(ss)
    repaired, _ = attach_scheduler_timesteps(dc, ts, device=device, stored_sigmas=expected_sigmas)
    assert repaired.sigmas.tolist() == expected_sigmas.tolist()
    assert repaired.sigmas.tolist() == [1.0, 0.75, 0.5]
