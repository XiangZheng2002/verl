# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CPU tests for the OPSD loss knobs on the FSDP forward-KL top-k path:
``clip_tau`` (per-vocab pointwise KL clip), ``loss_temperature`` (top-k
renormalization), the ``student_logits_temperature`` sampling-temperature undo,
and the ``clamp_negative_to_zero`` gate."""

from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from verl.trainer.distillation.fsdp.losses import compute_forward_kl_topk, kl_divergence
from verl.trainer.distillation.losses import compute_forward_kl_topk as collect_forward_kl_topk_metrics


def _nested_from_rows(rows):
    values = torch.tensor(rows)
    offsets = torch.tensor([0, len(rows)], dtype=torch.int64)
    return torch.nested.nested_tensor_from_jagged(values, offsets=offsets)


def _make_inputs():
    """Small (1, 3, 6) student logits + (3, 2) teacher top-k, as in
    test_distillation_topk_symmetry_on_cpu."""
    logits = torch.tensor(
        [
            [0.0, 9.0, 8.0, 1.0, 0.0, 0.0],
            [8.0, 7.0, 0.0, 0.0, 9.0, 0.0],
            [9.0, 8.0, 7.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    ).unsqueeze(0)
    teacher_ids = _nested_from_rows([[1, 2], [4, 5], [3, 4]]).to(torch.int64)
    teacher_logprobs = _nested_from_rows([[-0.4, -1.6], [-0.5, -1.2], [-0.7, -0.9]]).to(torch.float32)
    return logits, teacher_ids, teacher_logprobs


def _config(**loss_kwargs):
    loss_kwargs.setdefault("log_prob_min_clamp", None)
    return SimpleNamespace(distillation_loss=SimpleNamespace(**loss_kwargs))


def test_kl_divergence_clip_tau_clamps_before_vocab_sum():
    log_p = torch.log(torch.tensor([[0.8, 0.2]]))
    log_q = torch.log(torch.tensor([[0.1, 0.9]]))
    per_vocab = log_p.exp() * (log_p - log_q)  # first entry > 0.5, second < 0

    torch.testing.assert_close(kl_divergence(log_q, log_p), per_vocab.sum(-1))
    torch.testing.assert_close(kl_divergence(log_q, log_p, clip_tau=0.5), per_vocab.clamp_max(0.5).sum(-1))


def test_student_logits_temperature_undoes_engine_division():
    logits, teacher_ids, teacher_logprobs = _make_inputs()
    temperature = torch.tensor([1.1, 0.9, 1.3]).view(1, -1, 1)

    raw = compute_forward_kl_topk(logits, teacher_logprobs, teacher_ids, _config(), "thd")
    undone = compute_forward_kl_topk(
        logits / temperature, teacher_logprobs, teacher_ids, _config(), "thd",
        student_logits_temperature=temperature,
    )

    for key in ("distillation_losses", "student_mass"):
        torch.testing.assert_close(undone[key], raw[key])


def test_loss_temperature_renormalizes_over_topk():
    logits, teacher_ids, teacher_logprobs = _make_inputs()
    T = 2.0

    out = compute_forward_kl_topk(logits, teacher_logprobs, teacher_ids, _config(loss_temperature=T), "thd")

    ids = teacher_ids.values().unsqueeze(0)
    log_q = torch.log_softmax(torch.gather(logits, -1, ids) / T, dim=-1)
    log_p = torch.log_softmax(teacher_logprobs.values().unsqueeze(0) / T, dim=-1)
    torch.testing.assert_close(out["distillation_losses"], kl_divergence(log_q, log_p))
    # diagnostic masses stay at T=1: renormalized top-k probs would sum to exactly 1
    assert not torch.allclose(out["student_mass"], torch.ones_like(out["student_mass"]))


def test_loss_temperature_rejects_chunked_topk():
    logits, teacher_ids, teacher_logprobs = _make_inputs()
    config = _config(loss_temperature=2.0, use_chunked_topk=True)
    with pytest.raises(NotImplementedError, match="loss_temperature"):
        compute_forward_kl_topk(logits, teacher_logprobs, teacher_ids, config, "thd")


@pytest.mark.parametrize("clamp_negative_to_zero", [True, False])
def test_clamp_negative_to_zero_gate(clamp_negative_to_zero):
    # setup mirrors test_forward_kl_topk_metric_aggregation_for_overlap_outputs
    data = TensorDict(
        {
            "prompts": torch.tensor([[101]]),
            "responses": torch.tensor([[11, 12, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 0]]),
            "response_mask": torch.tensor([[1, 1, 0]], dtype=torch.bool),
        },
        batch_size=[1],
    )
    model_output = {
        "distillation_losses": torch.tensor([-0.5, 0.2, 0.3]),
        "student_mass": torch.tensor([0.9, 0.8, 0.7]),
        "teacher_mass": torch.tensor([0.95, 0.85, 0.75]),
        "overlap_count": torch.tensor([2, 1, 0]),
        "overlap_token_advantage": torch.tensor([-0.2, -0.4, 0.0]),
    }
    distillation_config = SimpleNamespace(
        distillation_loss=SimpleNamespace(topk=2, clamp_negative_to_zero=clamp_negative_to_zero)
    )

    losses, _ = collect_forward_kl_topk_metrics(
        config=SimpleNamespace(), distillation_config=distillation_config, model_output=model_output, data=data
    )

    expected_first = 0.0 if clamp_negative_to_zero else -0.5
    assert losses.flatten()[0].item() == pytest.approx(expected_first)
