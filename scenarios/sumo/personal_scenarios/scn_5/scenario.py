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
from smarts.sstudio.sstypes.actor.social_agent_actor import SocialAgentActor

normal = TrafficActor(name="car", depart_speed=0)
social_agent2 = SocialAgentActor(
        name="car1",
        agent_locator='scenarios.sumo.zoo_intersection.agent_prefabs:zoo-agent2-v0',
        initial_speed=0.0
        )

social_agent1 = SocialAgentActor(
        name="car2",
        agent_locator='scenarios.sumo.zoo_intersection.agent_prefabs:zoo-agent2-v0',
        initial_speed=0.0
        )

actor_missions = [
    Mission(
        route=Route(begin=("gneE3", 1, 35), end=("gneE3", 1, 100)),
        entry_tactic=TrapEntryTactic(
            start_time=0
        ),
    ),
]

traffic = {}
traffic["0"] = Traffic(
    engine="SUMO",
    trips=[
        Trip(
            vehicle_name=f"car_{i+1}",
            route=Route(
                begin=("gneE3", 1, (i+2) * 12),
                end=("gneE3", 1, (i+2) * 12)
            ),
            depart=0,
            actor=normal
        )
        for i in range(4)
    ],
    flows=[]
)

    
route = Route(begin=("gneE3", 1, 29), end=("gneE3", 1, 100))
ego_missions = [
    Mission(
        route=route,
        entry_tactic=TrapEntryTactic(
            start_time=0
        ),
    )
]

gen_scenario(
    scenario=Scenario(
        # traffic=traffic,
        ego_missions=ego_missions,
        social_agent_missions={
            f"s-agent-{social_agent2.name}": (
                [social_agent2],
                [
                    EndlessMission(
                        begin=("gneE3", 1, 24),
                        entry_tactic=TrapEntryTactic(start_time=0),
                    ),
                ],
            ),
            f"s-agent-{social_agent1.name}": (
                [social_agent1],
                [
                    EndlessMission(
                        begin=("gneE3", 1, 36),
                        entry_tactic=TrapEntryTactic(start_time=0),
                    ),
                ],
            ),
        },
    ),
    output_dir=Path(__file__).parent,
)

