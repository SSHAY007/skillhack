# Copyright (c) Facebook, Inc. and its affiliates.
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
import logging

from agent.common.envs import tasks
from agent.polybeast.models.base import BaseNet, RandomNet, HalfCheetahAgent
from agent.polybeast.models.intrinsic import RNDNet, RIDENet
from nle.env.base import DUNGEON_SHAPE
from envs.skills_all import HalfCheetah



from agent.polybeast.models.frozen_optioncritic import FOCNet
from agent.polybeast.models.hks import HKSNet
from agent.polybeast.models.kickstarting import KSNet


def create_model(flags, device):
    model_string = flags.model
    if model_string == "random":
        model_cls = RandomNet
    elif model_string == "baseline":
        model_cls = BaseNet
    elif model_string == "rnd":
        model_cls = RNDNet
    elif model_string == "HalfCheetah":
        model_cls = HalfCheetahAgent
    elif model_string == "ride":
        model_cls = RIDENet
    elif model_string == "foc":
        model_cls = FOCNet
    elif model_string == "ks":
        model_cls = KSNet
    elif model_string == "hks":
        model_cls = HKSNet
    elif model_string == "cnn" or model_string == "transformer":
        raise RuntimeError(
            "model=%s deprecated, use model=baseline crop_model=%s instead"
            % (model_string, model_string)
        )
    else:
        raise NotImplementedError("model=%s" % model_string)
    #env = tasks.ENVS[flags.env]()()
    #env = tasks.ENVS[flags.env]()().actions
    #num_actions = tasks.ENVS[flags.env]().actions
    num_actions = 5


    env = HalfCheetah()()#have a functor, first brackets is constructor and second is a function call
    #model = model_cls(DUNGEON_SHAPE, num_actions, flags, device)
    model = model_cls(env.observation_space,env.action_space,flags,device)
    model.to(device=device)
    return model
