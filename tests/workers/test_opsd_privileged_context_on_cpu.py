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
"""CPU tests for the OPSD privileged-context helpers and config sizing.

Covers the two pure helpers in ``teacher_manager`` (privileged-sequence build,
privileged->student layout remap for both the top-k 2-D and the non-top-k 1-D
tensor shapes) and the teacher context-budget arithmetic
(``SelfDistillationConfig.privileged_extra`` /
``DistillationTeacherModelConfig.validate_and_prepare_for_distillation``).
"""

import pytest
import torch

from verl.experimental.teacher_loop.teacher_manager import (
    build_privileged_sequence,
    remap_privileged_to_student_layout,
)
from verl.workers.config.distillation import (
    DistillationTeacherModelConfig,
    SelfDistillationConfig,
)
from verl.workers.config.rollout import RolloutConfig


class _CharTokenizer:
    """Deterministic char-level stub: encode/decode round-trip exactly."""

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, ids):
        return "".join(chr(i) for i in ids)

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True, enable_thinking=None):
        return self.encode(f"<u>{messages[0]['content']}</u><a>")


_SAMPLE_KWARGS = {"problem": "1+1=?", "reward_model": {"ground_truth": "ABCDEF"}}


def test_build_privileged_sequence_layout():
    tok = _CharTokenizer()
    cfg = SelfDistillationConfig(enabled=True)
    response_ids = [7, 8, 9]

    seq, prefix_len = build_privileged_sequence(response_ids, _SAMPLE_KWARGS, cfg, tok)

    # response is appended verbatim after the teacher prompt, and prefix_len marks the boundary
    assert seq[prefix_len:] == response_ids
    prompt_text = tok.decode(seq[:prefix_len])
    # dotted reference_key resolved the nested dict; both fields land in the rendered template
    assert "1+1=?" in prompt_text
    assert "ABCDEF" in prompt_text


def test_build_privileged_sequence_truncates_solution_to_budget():
    tok = _CharTokenizer()
    cfg = SelfDistillationConfig(enabled=True, max_reference_length=3)

    seq, prefix_len = build_privileged_sequence([7], _SAMPLE_KWARGS, cfg, tok)

    prompt_text = tok.decode(seq[:prefix_len])
    assert "ABC" in prompt_text
    assert "ABCD" not in prompt_text


@pytest.mark.parametrize("k_shape", [(), (2,)], ids=["1d_no_topk", "2d_topk"])
def test_remap_privileged_to_student_layout(k_shape):
    prompt_len, resp_len, priv_prefix_len = 3, 2, 5
    priv_total = priv_prefix_len + resp_len
    priv_ids = torch.arange(priv_total * max(1, *k_shape, 1)).reshape(priv_total, *k_shape)
    priv_logprobs = torch.randn(priv_total, *k_shape)

    ids, logprobs = remap_privileged_to_student_layout(
        priv_ids, priv_logprobs, prompt_len=prompt_len, resp_len=resp_len, priv_prefix_len=priv_prefix_len
    )

    # student layout: prompt_len filler rows + the response rows from the privileged positions
    assert ids.shape == (prompt_len + resp_len, *k_shape)
    assert logprobs.shape == (prompt_len + resp_len, *k_shape)
    torch.testing.assert_close(ids[prompt_len:], priv_ids[priv_prefix_len : priv_prefix_len + resp_len])
    torch.testing.assert_close(logprobs[prompt_len:], priv_logprobs[priv_prefix_len : priv_prefix_len + resp_len])


def test_privileged_extra_is_zero_when_disabled():
    assert SelfDistillationConfig(enabled=False).privileged_extra == 0
    cfg = SelfDistillationConfig(enabled=True, max_reference_length=100, max_bridge_length=20)
    assert cfg.privileged_extra == 120


def test_teacher_sizing_reserves_privileged_extra():
    tm = DistillationTeacherModelConfig(
        inference=RolloutConfig(prompt_length=4, response_length=2, max_model_len=17)
    )

    # required = prompt(4) + privileged_extra(10) + response(2) + 1 == 17 -> fits exactly
    tm.validate_and_prepare_for_distillation(use_topk=False, topk=None, privileged_extra=10)
    assert tm.inference.prompt_length == 4 + 10 + 2
    assert tm.inference.response_length == 1


def test_teacher_sizing_rejects_overflow():
    tm = DistillationTeacherModelConfig(
        inference=RolloutConfig(prompt_length=4, response_length=2, max_model_len=16)
    )
    with pytest.raises(ValueError, match="privileged"):
        tm.validate_and_prepare_for_distillation(use_topk=False, topk=None, privileged_extra=10)
