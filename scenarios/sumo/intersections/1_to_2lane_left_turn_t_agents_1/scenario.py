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
    Flow,
    Mission,
    Route,
    Scenario,
    Traffic,
    TrafficActor,
    TrapEntryTactic,
)

normal = TrafficActor(
    name="car",
)

horizontal_routes = [
    ("E4", 0, "E1", 0),
    ("E4", 1, "E1", 1),
    ("-E1", 0, "-E4", 0),
    ("-E1", 1, "-E4", 1),
]

turn_left_routes = [
    ("E0", 0, "E1", 1),
    ("E4", 1, "-E0", 0),
]

turn_right_routes = [
    ("E0", 0, "-E4", 0),
    ("-E1", 0, "-E0", 0),
]

# Total route combinations = 8C1 + 8C2 + 8C3 + 8C4 + 8C5 = 218
# Repeated route combinations = 218 * 2 = 436
all_routes = horizontal_routes + turn_left_routes + turn_right_routes
route_comb = [
    com for elems in range(1, 6) for com in combinations(all_routes, elems)
] * 2
traffic = {}
for name, routes in enumerate(route_comb):
    traffic[str(name)] = Traffic(
        flows=[
            Flow(
                route=Route(
                    begin=(start_edge, start_lane, 0),
                    end=(end_edge, end_lane, "max"),
                ),
                # Random flow rate, between x and y vehicles per minute.
                rate=60 * random.uniform(5, 10),
                # Random flow start time, between x and y seconds.
                begin=random.uniform(0, 3),
                # For an episode with maximum_episode_steps=3000 and step
                # time=0.1s, the maximum episode time=300s. Hence, traffic is
                # set to end at 900s, which is greater than maximum episode
                # time of 300s.
                end=60 * 15,
                actors={normal: 1},
            )
            for start_edge, start_lane, end_edge, end_lane in routes
        ]
    )

route = Route(begin=("E0", 0, 5), end=("E1", 0, "max"))
ego_missions = [
    Mission(
        route=route,
        entry_tactic=TrapEntryTactic(
            start_time=4,
            wait_to_hijack_limit_s=1,
        ),
    ),
]

gen_scenario(
    scenario=Scenario(
        traffic=traffic,
        ego_missions=ego_missions,
    ),
    output_dir=Path(__file__).parent,
)
