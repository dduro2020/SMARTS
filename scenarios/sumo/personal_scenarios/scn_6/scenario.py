# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import random
from itertools import combinations
from pathlib import Path

from smarts.sstudio import gen_scenario
from smarts.sstudio.sstypes import (
    Mission,
    EndlessMission,
    Route,
    Scenario,
    Traffic,
    Trip,
    TrafficActor,
    TrapEntryTactic,
)

from smarts.sstudio import types as t

vertical_routes = [
    ("E2", 0, "E7", 0),
    ("E8", 0, "E1", 1),
]

horizontal_routes = [
    ("E3", 0, "E5", 0),
    ("E3", 1, "E5", 1),
    ("E3", 2, "E5", 2),
    ("E6", 1, "E4", 1),
    ("E6", 0, "E4", 0),
]

turn_left_routes = [
    ("E8", 0, "E5", 2),
    ("E6", 1, "E1", 1),
    ("E2", 1, "E4", 1),
    ("E3", 2, "E7", 0),
]

turn_right_routes = [
    ("E6", 0, "E7", 0),
    ("E3", 0, "E1", 0),
    ("E2", 0, "E5", 0),
    ("E8", 0, "E4", 0),
]

normal = TrafficActor(name="car", depart_speed=0)

traffic = {}
traffic["0"] = Traffic(
    engine="SUMO",
    trips=[
        Trip(
            vehicle_name=f"car_{i+1}",
            route=Route(
                begin=("E6", 1, (i+2) * 12),
                end=("E6", 1, 40)
            ),
            depart=0,
            actor=normal
        )
        for i in range(2)
    ],
    flows=[]
)

route = Route(begin=("E6", 1, 28), end=("E6", 1, 40))
ego_missions = [
    Mission(
        # route=route,
        route=t.RandomRoute(),
        entry_tactic=TrapEntryTactic(
            start_time=0
        ),
    )
]

gen_scenario(
    scenario=Scenario(
        traffic=traffic,
        ego_missions=ego_missions,
    ),
    output_dir=Path(__file__).parent,
)
