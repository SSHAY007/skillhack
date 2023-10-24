# This file taken from
#     https://github.com/deepmind/scalable_agent/blob/
#         cd66d00914d56c8ba2f0615d9cdeefcb169a8d70/vtrace.py
# and modified.

# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Functions to compute V-trace off-policy actor critic targets.

For details and theory see:

"IMPALA: Scalable Distributed Deep-RL with
Importance Weighted Actor-Learner Architectures"
by Espeholt, Soyer, Munos et al.

See https://arxiv.org/abs/1802.01561 for the full paper.
"""

import collections

import torch
import torch.nn.functional as F


import logging

VTraceFromLogitsReturns = collections.namedtuple(
    "VTraceFromLogitsReturns",
    [
        "vs",
        "pg_advantages",
        "log_rhos",
        "behavior_action_log_probs",
        "target_action_log_probs",
    ],
)

VTraceReturns = collections.namedtuple("VTraceReturns", "vs pg_advantages")

def action_log_probs2(policy_logits, actions):
    policy_logits = torch.flatten(policy_logits, 0, 1)
    actions = torch.flatten(actions, 0, 1)
    #logging.critical(policy_logits.shape)
    #logging.critical(actions.shape)
    return -F.nll_loss(
        F.log_softmax(torch.flatten(policy_logits, 0, 1), dim=-1),
        torch.flatten(actions, 0, 1),
        reduction="none",
    ).view_as(actions)

def action_log_probs(policy_logits, actions):
#torch.Size([2560, 5])
#torch.Size([2560])
    logging.critical(policy_logits.shape)
    logging.critical(actions.shape)
    return -F.nll_loss(
        F.log_softmax(torch.flatten(policy_logits, 0, 1), dim=-1),
        torch.flatten(actions, 0, 1),
        reduction="none",
    ).view_as(actions)


def from_logits(
    behavior_policy_logits,
    target_policy_logits,
    actions,
    discounts,
    rewards,
    values,
    bootstrap_value,
    clip_rho_threshold=1.0,
    clip_pg_rho_threshold=1.0,
):
    """V-trace for softmax policies."""

    #target_action_log_probs = action_log_probs(target_policy_logits, actions)#the probleeem
    #behavior_action_log_probs = action_log_probs(behavior_policy_logits, actions)
    target_action_log_probs = target_policy_logits
    behavior_action_log_probs = behavior_policy_logits


    log_rhos = target_action_log_probs - behavior_action_log_probs
    log_rhos = log_rhos.sum(2)
    vtrace_returns = from_importance_weights(
        log_rhos=log_rhos,
        discounts=discounts,
        rewards=rewards,
        values=values,
        bootstrap_value=bootstrap_value,
        clip_rho_threshold=clip_rho_threshold,
        clip_pg_rho_threshold=clip_pg_rho_threshold,
    )
    return VTraceFromLogitsReturns(
        log_rhos=log_rhos,
        behavior_action_log_probs=behavior_action_log_probs,
        target_action_log_probs=target_action_log_probs,
        **vtrace_returns._asdict(),
    )


@torch.no_grad()
def from_importance_weights(
    log_rhos,
    discounts,
    rewards,
    values,
    bootstrap_value,
    clip_rho_threshold=1.0,
    clip_pg_rho_threshold=1.0,
):
    """V-trace from log importance weights."""
    with torch.no_grad():
        rhos = torch.exp(log_rhos)
        if clip_rho_threshold is not None:
            clipped_rhos = torch.clamp(rhos, max=clip_rho_threshold)
        else:
            clipped_rhos = rhos

        cs = torch.clamp(rhos, max=1.0)
        # Append bootstrapped value to get [v1, ..., v_t+1]
        values_t_plus_1 = torch.cat(
            [values[1:], torch.unsqueeze(bootstrap_value, 0)], dim=0
        )


        #torch.Size([80, 32, 6]) clipped ratios of pi/mu
        #torch.Size([80, 32]) values at 80 timesteps of 32 batchsizes
        #torch.Size([80, 32])

        #clipped_rhos = clipped_rhos.sum(2)
        # to get policy at different actions we have to take the sum
        # since sum(log(policy)) = product(sum(policy))
        #logging.info(clipped_rhos.shape)
        #logging.info(values_t_plus_1.shape)
        #logging.info(values.shape)

        deltas = clipped_rhos * (
            rewards + discounts * values_t_plus_1 - values
        ) # temporal difference

        acc = torch.zeros_like(bootstrap_value)
        result = []
        #logging.info(acc.shape)
        #logging.info(discounts.shape)
        #logging.info(cs.shape)
        for t in range(discounts.shape[0] - 1, -1, -1):
            acc = deltas[t] + discounts[t] * cs[t] * acc
            result.append(acc)
        result.reverse()
        vs_minus_v_xs = torch.stack(result)

        # Add V(x_s) to get v_s.
        vs = torch.add(vs_minus_v_xs, values)

        # Advantage for policy gradient.
        vs_t_plus_1 = torch.cat(
            [vs[1:], torch.unsqueeze(bootstrap_value, 0)], dim=0
        )
        if clip_pg_rho_threshold is not None:
            clipped_pg_rhos = torch.clamp(rhos, max=clip_pg_rho_threshold)
        else:
            clipped_pg_rhos = rhos
        pg_advantages = clipped_pg_rhos * (
            rewards + discounts * vs_t_plus_1 - values
        )

        # Make sure no gradients backpropagated through the returned values.
        return VTraceReturns(vs=vs, pg_advantages=pg_advantages)
