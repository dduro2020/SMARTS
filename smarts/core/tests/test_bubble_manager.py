# MIT License
#
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
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
import math
from functools import partial
from typing import Any, Generator, Sequence, Tuple

import pytest
from helpers.scenario import temp_scenario

import smarts.sstudio.sstypes as t
from smarts.core.controllers import ActionSpaceType
from smarts.core.coordinates import Heading, Pose
from smarts.core.scenario import Scenario
from smarts.core.smarts import SMARTS
from smarts.core.tests.helpers.providers import MockProvider, MockTrafficProvider
from smarts.core.vehicle_index import VehicleIndex
from smarts.sstudio import gen_scenario

HEADING_CONSTANT = Heading(-math.pi / 2)

# TODO: Add test for traveling bubbles


@pytest.fixture
def time_resolution(request):
    tr = getattr(request, "param", 0.1)
    assert tr >= 1e-10, "Should be a non-negative non-zero real number"
    return tr


@pytest.fixture
def bubble_limits(request):
    return getattr(request, "param", t.BubbleLimits(10, 11))


@pytest.fixture
def social_actor(transition_cases):
    _, _, arg = transition_cases
    social_actor_part = partial(t.SocialAgentActor, name="zoo-car")
    if arg == "keep-lane":
        return social_actor_part(agent_locator="zoo.policies:keep-lane-agent-v0")
    elif arg == "keep-pose":
        return social_actor_part(
            agent_locator="smarts.core.tests.helpers.agent_prefabs:keep-pose-v0"
        )
    elif isinstance(arg, (int, float)):
        return social_actor_part(
            agent_locator="smarts.core.tests.helpers.agent_prefabs:move-to-target-pose-v0",
            policy_kwargs=dict(
                target_pose=Pose.from_center([arg, 0, 0], HEADING_CONSTANT),
            ),
        )


@pytest.fixture
def bubble(
    bubble_limits: t.BubbleLimits, social_actor: t.SocialAgentActor, transition_cases
):
    """
    |(93)  |(95)     (100)     (105)|  (107)|
    """
    _, margin, _ = transition_cases
    return t.Bubble(
        zone=t.PositionalZone(pos=(100, 0), size=(10, 10)),
        margin=margin,
        limit=bubble_limits,
        actor=social_actor,
    )


@pytest.fixture
def scenarios(bubble: t.Bubble):
    with temp_scenario(name="straight", map="maps/straight.net.xml") as scenario_root:
        gen_scenario(
            t.Scenario(traffic={}, bubbles=[bubble]),
            output_dir=scenario_root,
        )
        yield Scenario.variations_for_all_scenario_roots([str(scenario_root)], [])


@pytest.fixture
def mock_provider(request):
    provider_type = getattr(request, "param", "mock_provider")
    if provider_type == "mock_provider":
        return MockProvider()
    elif provider_type == "mock_traffic_provider":
        return MockTrafficProvider()


@pytest.fixture
def smarts(
    scenarios: Generator[Scenario, None, None],
    mock_provider: MockProvider,
    time_resolution: float,
):
    smarts_partial = partial(SMARTS, agent_interfaces={})
    if isinstance(mock_provider, MockTrafficProvider):
        smarts_ = smarts_partial(traffic_sims=[mock_provider])
    elif isinstance(mock_provider, MockProvider):
        smarts_ = smarts_partial()

        smarts_.add_provider(mock_provider)
    smarts_.reset(next(scenarios))
    yield smarts_
    smarts_.destroy()


@pytest.fixture()
def transition_cases(request):
    # o outside
    # a airlock
    # b bubble
    p = getattr(request, "param", "oabao")
    margin = 2 if "a" in p else 0  # margin not needed if no airlock

    if p == "oabao":
        return (
            (
                # Outside airlock and bubble
                ((92, 0, 0), (False, False)),
                # Inside airlock, begin collecting experiences, but don't hijack
                ((94, 0, 0), (True, False)),
                # Entered bubble, now hijack
                ((100, 0, 0), (False, True)),
                # Leave bubble into exiting airlock
                ((105.01, 0, 0), (False, True)),
                # Exit bubble and airlock, now relinquish
                ((107.01, 0, 0), (False, False)),
            ),
            margin,
            "keep-pose",
        )
    elif p == "obbao":
        return (
            (
                # Outside airlock and bubble
                ((92, 0, 0), (False, False)),
                # Dropped into middle of bubble shadow
                ((100, 0, 0), (True, False)),
                # Step has been taken, now hijack
                ((100, 0, 0), (False, True)),
                # Exit bubble and airlock, now relinquish
                ((108, 0, 0), (False, False)),
            ),
            margin,
            108,
        )
    elif p == "oaoao":
        return (
            (
                # Outside airlock and bubble
                ((92, 0, 0), (False, False)),
                # Dropped into airlock
                ((94, 0, 0), (True, False)),  # inside airlock
                # Outside airlock and bubble
                ((92, 0, 0), (False, False)),
                # Dropped into airlock
                ((94, 0, 0), (True, False)),  # inside airlock
                # Outside airlock and bubble
                ((92, 0, 0), (False, False)),
            ),
            margin,
            92,
        )
    elif p == "aa":
        return (
            (
                ((94, 0, 0), (False, False)),  # inside airlock
                ((94, 0, 0), (True, False)),  # inside airlock
            ),
            margin,
            94,
        )
    elif p == "oa":
        return (
            (
                ((94, 0, 0), (False, False)),  # inside airlock
                ((94, 0, 0), (True, False)),  # inside airlock
            ),
            margin,
            94,
        )
    elif p == "bbb":
        return (
            (
                ((100, 0, 0), (False, False)),  # inside bubble
                ((100, 0, 0), (True, False)),  # inside bubble
                ((100, 0, 0), (False, True)),  # inside bubble
            ),
            margin,
            100,
        )
    elif p == "obb":
        return (
            (
                ((92, 0, 0), (False, False)),  # inside bubble
                ((100, 0, 0), (True, False)),  # inside bubble
                ((100, 0, 0), (False, True)),  # inside bubble
            ),
            margin,
            100,
        )
    elif p == "obbob":
        return (
            (
                ((105.001, 0, 0), (False, False)),  # outside bubble
                ((104.9, 0, 0), (True, False)),  # inside
                ((104.9, 0, 0), (False, True)),  # inside
                ((105.001, 0, 0), (False, False)),  # outside
                ((104.9, 0, 0), (True, False)),  # inside
            ),
            margin,
            105.01,
        )
    elif p == "obobo":
        return (
            (
                ((105.001, 0, 0), (False, False)),  # outside bubble
                ((104.9, 0, 0), (True, False)),  # inside
                ((105.001, 0, 0), (False, False)),  # outside
                ((104.9, 0, 0), (True, False)),  # inside
            ),
            margin,
            105.01,
        )


@pytest.mark.parametrize(
    "mock_provider", ["mock_provider", "mock_traffic_provider"], indirect=True
)
@pytest.mark.parametrize(
    "transition_cases",
    [
        "oa",
        "aa",  # behavior for first step entry
        "obb",
        "bbb",  # behavior for first step entry
        "oabao",
        "oaoao",
        "obbao",
        "obbob",
        "obobo",
    ],
    indirect=True,
)
def test_bubble_manager_state_change(
    smarts: SMARTS,
    mock_provider: MockProvider,
    transition_cases: Tuple[
        Sequence[Tuple[Tuple[float, float, float], Tuple[bool, bool]]], int, Any
    ],
):
    state_at_position, _, _ = transition_cases
    index: VehicleIndex = smarts.vehicle_index

    vehicle_id = "vehicle"

    for (next_position, (shadowed, hijacked)) in state_at_position:
        mock_provider.override_next_provider_state(
            vehicles=[
                (vehicle_id, Pose.from_center(next_position, HEADING_CONSTANT), 10)
            ]
        )

        # Providers must be disjoint
        if index.vehicle_is_hijacked(vehicle_id):
            mock_provider.clear_next_provider_state()
            agent_id = index.owner_id_from_vehicle_id(vehicle_id)
            interface = smarts.agent_manager.agent_interface_for_agent_id(agent_id)
            while (
                index.vehicle_is_hijacked(vehicle_id)
                and index.vehicle_position(vehicle_id)[0] < next_position[0]
            ):
                if interface.action == ActionSpaceType.TargetPose:
                    smarts.agent_manager.reserve_social_agent_action(
                        agent_id,
                        (next_position[0], next_position[1], HEADING_CONSTANT, 0.1),
                    )
                smarts.step({})

        else:
            smarts.step({})

        got_shadowed = index.vehicle_is_shadowed(vehicle_id)
        got_hijacked = index.vehicle_is_hijacked(vehicle_id)
        assert_msg = (
            f"position={next_position}\n"
            f"\tvehicle_position={index.vehicle_position(vehicle_id)}\n"
            f"\t(expected: shadowed={shadowed}, hijacked={hijacked})\n"
            f"\t(received: shadowed={got_shadowed}, hijacked={got_hijacked})"
        )
        assert got_shadowed == shadowed, assert_msg
        assert got_hijacked == hijacked, assert_msg


@pytest.mark.parametrize("bubble_limits", [t.BubbleLimits(1, 1)], indirect=True)
def test_bubble_manager_limit(
    smarts: SMARTS,
    mock_provider: MockProvider,
    time_resolution: float,
):
    vehicle_ids = ["vehicle-1", "vehicle-2", "vehicle-3"]
    current_vehicle_ids = [*vehicle_ids]
    step_vehicle_ids = [(y, id_) for y, id_ in enumerate(vehicle_ids)]
    speed = 2.5
    distance_per_step = speed * time_resolution

    for x in range(59, 69):
        current_vehicle_ids = {
            v_id
            for v_id in vehicle_ids
            if not smarts.vehicle_index.vehicle_is_hijacked(v_id)
        }

        vehicles = [
            (
                v_id,
                Pose.from_center(
                    (80 + x * distance_per_step, y * 4 - 4, 0),
                    HEADING_CONSTANT,
                ),
                speed,  # speed
            )
            for y, v_id in step_vehicle_ids
            if v_id in current_vehicle_ids
        ]
        mock_provider.override_next_provider_state(vehicles=vehicles)
        smarts.step({})

    # 3 total vehicles, 1 hijacked and removed according to limit, 2 remaining
    assert (
        len(current_vehicle_ids) == 2
    ), "Only 1 vehicle should have been hijacked according to the limit"


def test_vehicle_spawned_outside_bubble_is_captured(
    smarts: SMARTS, mock_provider: MockProvider
):
    # Spawned vehicle drove through airlock so _should_ get captured
    vehicle_id = "vehicle"
    got_hijacked = False
    for x in range(20):
        mock_provider.override_next_provider_state(
            vehicles=[
                (
                    vehicle_id,
                    Pose.from_center((90 + x, 0, 0), HEADING_CONSTANT),
                    10,  # speed
                )
            ]
        )
        smarts.step({})
        if smarts.vehicle_index.vehicle_is_hijacked(vehicle_id):
            got_hijacked = True
            break

    assert got_hijacked
