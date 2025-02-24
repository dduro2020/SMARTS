.. _agent:

Agent
=====

An agent is built by specifying its desired (i) `interface` and (ii) `policy`. 
The `interface` and `policy` are contained inside a :class:`~smarts.zoo.agent_spec.AgentSpec` class. 
A snippet of :class:`~smarts.zoo.agent_spec.AgentSpec` class is shown here.

.. code-block:: python

    class AgentSpec:
        interface: AgentInterface
        agent_builder: Callable[..., Agent] = None
        agent_params: Optional[Any] = None

Next, a minimal example of how to create and register an agent is illustrated.

.. _minimal_agent:
.. code-block:: python

    from smarts.core.agent import Agent
    from smarts.core.agent_interface import AgentInterface, AgentType
    from smarts.core.controllers import ActionSpaceType
    from smarts.zoo.agent_spec import AgentSpec
    from smarts.zoo.registry import register

    # A policy which simply follows the waypoint paths and drives at the road's
    # speed limit.
    class FollowWaypoints(Agent):
        def __init__(self):
            """Any policy initialization matters, including loading of model,
            may be performed here.
            """
            pass

        def act(self, obs):
            speed_limit = obs.waypoint_paths[0][0].speed_limit

            return (speed_limit, 0)

    # AgentSpec specifying the agent's interface and policy.
    agent_spec = AgentSpec(
        # Agent's interface.
        interface=AgentInterface.from_type(
            requested_type = AgentType.LanerWithSpeed,
            max_episode_steps=500,
        ),
        # Agent's policy.
        agent_builder=FollowWaypoints,
        agent_params=None, # Optional parameters passed to agent_builder during building. 
    )

    def entry_point(**kwargs):
       """An entrypoint for the agent, which takes any number of optional keyword
       arguments, and returns an :class:`~smarts.zoo.agent_spec.AgentSpec`.
       """
       return agent_spec

    # Register the agent.
    register(
        "follow-waypoints-v0",
        entry_point=entry_point,
    )

A registered agent can, at a later time, be built (i.e., instantiated) using its agent locator string.

.. code-block:: python

    from smarts.zoo.registry import make_agent

    # Builds the agent, by instantiatng the agent's policy.
    follow_waypoints_agent = make_agent("smarts.zoo:follow-waypoints-v0")

| The syntax of an agent locator is:
| ``"`` ``module.importable.in.python`` ``:`` ``registered_name_of_agent`` ``-v`` ``X`` ``"``

-  ``module.importable.in.python`` : Denotes the module in which the agent was 
   registered. For example, if the agent was registered in 
   ``smarts/zoo/__init__.py``, the module would be ``smarts.zoo``. The module
   must be importable from within python. An easy test to see if the module is
   importable, is to try importing the module within interactive python or a 
   script (e.g., ``import module.importable.in.python``)
- ``:`` : A separator, which separates the module and name sections of the
  locator.
-  ``registered_name_of_agent`` : The registered name of the agent.
-  ``-v`` : A version separator, which separates the name and version
   sections of the locator.
-  ``X`` : The version of the agent. This is required to register
   an agent. The version can be any positive integer.


Sections below elaborate on the agent's `interface` and `policy` design.

Interface
---------

The :class:`~smarts.core.agent_interface.AgentInterface` regulates information flow between the agent and SMARTS environment. 

+ It specifies the observation from the environment to the agent, by selecting the sensors to enable in the vehicle. 
+ It specifies the action from the agent to the environment. Attribute :attr:`~smarts.core.agent_interface.AgentInterface.action` controls the action type used. There are multiple action types to choose from :class:`~smarts.core.controllers.action_space_type.ActionSpaceType`.

Pre-configured interface
^^^^^^^^^^^^^^^^^^^^^^^^

SMARTS provides several pre-configured `interfaces` for ease of use. Namely,

+ `AgentType.Full`
+ `AgentType.StandardWithAbsoluteSteering`
+ `AgentType.Standard`
+ `AgentType.Laner`
+ `AgentType.LanerWithSpeed`
+ `AgentType.Tracker`
+ `AgentType.TrajectoryInterpolator`
+ `AgentType.MPCTracker`
+ `AgentType.Boid`

The attributes enabled for each pre-configured `interface` is shown in the table below.


.. list-table::
   :header-rows: 1

   * - **Interface**
     - :attr:`~smarts.core.agent_interface.AgentType.Full`
     - :attr:`~smarts.core.agent_interface.AgentType.StandardWithAbsoluteSteering`
     - :attr:`~smarts.core.agent_interface.AgentType.Standard`
     - :attr:`~smarts.core.agent_interface.AgentType.Laner`
     - :attr:`~smarts.core.agent_interface.AgentType.LanerWithSpeed`
     - :attr:`~smarts.core.agent_interface.AgentType.Tracker`
     - :attr:`~smarts.core.agent_interface.AgentType.TrajectoryInterpolator`
     - :attr:`~smarts.core.agent_interface.AgentType.MPCTracker`
     - :attr:`~smarts.core.agent_interface.AgentType.Boid`
     - :attr:`~smarts.core.agent_interface.AgentType.Loner`
     - :attr:`~smarts.core.agent_interface.AgentType.Tagger`
     - :attr:`~smarts.core.agent_interface.AgentType.Direct`
   * - **action**
     - :attr:`~smarts.core.controllers.action_space_type.ActionSpaceType.Continuous`
     - :attr:`~smarts.core.controllers.action_space_type.ActionSpaceType.Continuous`
     - :attr:`~smarts.core.controllers.action_space_type.ActionSpaceType.ActuatorDynamic`
     - :attr:`~smarts.core.controllers.action_space_type.ActionSpaceType.Lane`
     - :attr:`~smarts.core.controllers.action_space_type.ActionSpaceType.LaneWithContinuousSpeed`
     - :attr:`~smarts.core.controllers.action_space_type.ActionSpaceType.Trajectory`
     - :attr:`~smarts.core.controllers.action_space_type.ActionSpaceType.TrajectoryWithTime`
     - :attr:`~smarts.core.controllers.action_space_type.ActionSpaceType.MPC`
     - :attr:`~smarts.core.controllers.action_space_type.ActionSpaceType.MultiTargetPose`
     - :attr:`~smarts.core.controllers.action_space_type.ActionSpaceType.Continuous`
     - :attr:`~smarts.core.controllers.action_space_type.ActionSpaceType.Continuous`
     - :attr:`~smarts.core.controllers.action_space_type.ActionSpaceType.Direct`
   * - **max_episode_steps**
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - **neighborhood_vehicles**
     - ✓
     - ✓
     - ✓
     - 
     -
     -
     -
     -
     - ✓
     -
     - ✓
     - ✓
   * - **waypoint_paths** 
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     -
     - ✓
     - ✓
     - ✓
     - ✓
     -
   * - **drivable_area_grid_map** 
     - ✓
     -
     -
     -
     -
     -
     -
     -
     -
     -
     -
     -
   * - **occupancy_grid_map**
     - ✓
     -
     -
     -
     -
     -
     -
     -
     -
     -
     -
     -
   * - **top_down_rgb**           
     - ✓
     -
     -
     -
     -
     -
     -
     -
     -
     -
     -
     -
   * - **occlusion_map**
     -
     -
     -
     -
     -
     -
     -
     -
     -
     -
     -
     -
   * - **custom_renders**
     -
     -
     -
     -
     -
     -
     -
     -
     -
     -
     -
     -
   * - **lidar_point_cloud**
     - ✓
     -
     -
     -
     -
     -
     -
     -
     -
     -
     -
     -
   * - **accelerometer**
     - ✓ 
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - **signals**
     - ✓
     -
     -
     -
     -
     -
     -
     -
     -
     -
     -
     - ✓
   * - **debug** 
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓

Here, ``max_episode_steps`` controls the max steps allowed for the agent in an episode. Defaults to ``None``, implies agent has no step limit.

.. note:: 

    While using ``RLlib``, the ``max_episode_steps`` control authority may be ceded to ``RLlib`` through their config option ``horizon``, but doing so 
    removes the ability to customize different max episode steps for each agent.

A pre-configured `interface` can be extended by supplying extra `kwargs`. For example the following extends `AgentType.Standard` pre-configured interface to include lidar observation.

.. code-block:: python

    agent_interface = AgentInterface.from_type(
        requested_type = AgentType.Standard,
        lidar_point_cloud = True, 
    )

Custom interface
^^^^^^^^^^^^^^^^

Alternatively, users may customize their agent `interface` from scratch, like:

.. code-block:: python

    from smarts.core.agent_interface import AgentInterface
    from smarts.core.controllers import ActionSpaceType

    agent_interface = AgentInterface(
        max_episode_steps=1000,
        waypoint_paths=True,
        neighborhood_vehicle_states=True,
        drivable_area_grid_map=True,
        occupancy_grid_map=True,
        top_down_rgb=True,
        lidar_point_cloud=False,
        action=ActionSpaceType.Continuous,
    )

Further customization of individual `interface` options of :class:`~smarts.core.agent_interface` is also possible.

.. code-block:: python

    from smarts.core.agent_interface import AgentInterface, NeighborhoodVehicles, RGB, Waypoints
    from smarts.core.controllers import ActionSpaceType

    agent_interface = AgentInterface(
        max_episode_steps=1000,
        waypoint_paths=Waypoints(lookahead=50), # lookahead 50 meters
        neighborhood_vehicle_states=NeighborhoodVehicles(radius=50), # only get neighborhood info with 50 meters.
        drivable_area_grid_map=True,
        occupancy_grid_map=True,
        top_down_rgb=RGB(height=128,width=128,resolution=100/128), # 128x128 pixels RGB image representing a 100x100 meters area.
        lidar_point_cloud=False,
        action=ActionSpaceType.Continuous,
    )

.. important::

    Generation of a drivable area grid map (``drivable_area_grid_map=True``), occupancy grid map (``occupancy_grid_map=True``), and RGB (``top_down_rgb=True``) images, may significantly slow down the environment ``step()``. 
    It is recommended to set these image renderings to ``False`` if the agent `policy` does not require such observations.

Spaces
^^^^^^

Spaces provide samples for variation. For reference on spaces, see `gymnasium <https://gymnasium.farama.org/api/spaces/>`_ .
SMARTS environments contains (i) ``observation_space`` and (ii) ``action_space`` attributes, which are dictionaries mapping agent ids to their corresponding observation or action spaces, respectively.

Consider a SMARTS env with an agent named `Agent_001`. If `Agent_001`'s `interface` is customized, then the agent's corresponding observation space (i.e., ``env.observation_space["Agent_001"]``) and action space (i.e., ``env.action_space["Agent_001"]``) from the environment would be changed accordingly. 

Policy
------

A `policy` dictates the actions that the agent takes as a function of the observation received from the environment.

All `policies` must inherit the base class of :class:`~~smarts.core.agent.Agent` and must contain a ``def act(self, obs)`` method.

The received ``obs`` argument in ``def act(self, obs)`` is controlled by the selected agent `interface`.

The ``act()`` method should return an action complying to the agent's chosen action type in its agent `interface`. 
For example, if action type :attr:`~smarts.core.controllers.action_space_type.ActionSpaceType.LaneWithContinuousSpeed` was chosen, then ``act()`` should return an action ``(speed, lane_change)`` with type ``(float, int)``. See the :ref:`example <minimal_agent>` above.

Custom camera rendering
-----------------------

The agent interface provides for additional configurable rendering cameras that can be used to augment what the agent can see.
This generally involves providing a fragment shader to generate or modify existing or uninitialized pixels.


Configuring custom rendering
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These custom renders can be configured through the agent interface.

.. code:: python

    # adding a variable
    # uniform vec2 position;
    crcd = CustomRenderVariableDependency(
        value=(4, 4),
        variable_name="position"
    )

    # adding a buffer
    # uniform float time;
    crbd = CustomRenderBufferDependency(
        buffer_dependency_name=BufferID.ELAPSED_SIM_TIME,
        variable_name="time",
    )

    # referencing an inbuilt camera to
    # uniform sampler2D iChannel0;
    # uniform vec2 iChannel0Resolution;
    crd0 = CustomRenderCameraDependency(
        camera_dependency_name=CameraSensorID.OCCLUSION,
        variable_name="iChannel0",
    )
    # fragment shader
    fshader0 = "warp.frag"
    cr0 = CustomRender(
        name="step0",
        fragment_shader_path=fshader0, 
        dependencies=(crd0, crcd, crbd),
        ...,
    )
    # referencing a previous shader step
    # uniform sampler2D step0_texture;
    crd1 = CustomRenderCameraDependency(
        camera_dependency_name="step0",
        variable_name="step0_texture",
    )
    # fragment shader
    fshader1 = "ftl.frag"
    cr1 = CustomRender(
        name="step1",
        fragment_shader_path=fshader1, # fragment shader
        dependencies=(crd1,),
        ...,
    )

    AgentInterface(
        ...,
        custom_renders=(
            cr0,
            cr1,
        )
    )

.. note::

    For camera dependencies a variable name will translate into both the texture sampler and resolution of the target camera.


Available internal render buffers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The render optionally has access to certain buffers. Note the following code:

.. code:: python

    # adding a buffer
    crbd = CustomRenderBufferDependency(
        buffer_dependency_name=BufferID.DELTA_TIME,
        variable_name="dt",
    )


These values are as follows:

.. list-table::
   :header-rows: 1

   * - **configuration**
     - DELTA_TIME
     - STEP_COUNT
     - STEPS_COMPLETED
     - ELAPSED_SIM_TIME
     - EVENTS_COLLISIONS
     - EVENTS_OFF_ROAD
     - EVENTS_OFF_ROUTE
     - EVENTS_ON_SHOULDER
     - EVENTS_WRONG_WAY
     - EVENTS_NOT_MOVING
     - EVENTS_REACHED_GOAL
     - EVENTS_REACHED_MAX_EPISODE_STEPS
     - EVENTS_AGENTS_ALIVE_DONE
     - EVENTS_INTEREST_DONE
     - EGO_VEHICLE_STATE_POSITION
     - EGO_VEHICLE_STATE_BOUNDING_BOX
     - EGO_VEHICLE_STATE_HEADING
     - EGO_VEHICLE_STATE_SPEED
     - EGO_VEHICLE_STATE_STEERING
     - EGO_VEHICLE_STATE_YAW_RATE
     - EGO_VEHICLE_STATE_ROAD_ID
     - EGO_VEHICLE_STATE_LANE_ID
     - EGO_VEHICLE_STATE_LANE_INDEX
     - EGO_VEHICLE_STATE_LINEAR_VELOCITY
     - EGO_VEHICLE_STATE_ANGULAR_VELOCITY
     - EGO_VEHICLE_STATE_LINEAR_ACCELERATION
     - EGO_VEHICLE_STATE_ANGULAR_ACCELERATION
     - EGO_VEHICLE_STATE_LINEAR_JERK
     - EGO_VEHICLE_STATE_ANGULAR_JERK
     - EGO_VEHICLE_STATE_LANE_POSITION
     - UNDER_THIS_VEHICLE_CONTROL
     - NEIGHBORHOOD_VEHICLE_STATES_POSITION
     - NEIGHBORHOOD_VEHICLE_STATES_BOUNDING_BOX
     - NEIGHBORHOOD_VEHICLE_STATES_HEADING
     - NEIGHBORHOOD_VEHICLE_STATES_SPEED
     - NEIGHBORHOOD_VEHICLE_STATES_ROAD_ID
     - NEIGHBORHOOD_VEHICLE_STATES_LANE_ID
     - NEIGHBORHOOD_VEHICLE_STATES_LANE_INDEX
     - NEIGHBORHOOD_VEHICLE_STATES_LANE_POSITION
     - NEIGHBORHOOD_VEHICLE_STATES_INTEREST
     - WAYPOINT_PATHS_POSITION
     - WAYPOINT_PATHS_HEADING
     - WAYPOINT_PATHS_LANE_ID
     - WAYPOINT_PATHS_LANE_WIDTH
     - WAYPOINT_PATHS_SPEED_LIMIT
     - WAYPOINT_PATHS_LANE_INDEX
     - WAYPOINT_PATHS_LANE_OFFSET
     - DISTANCE_TRAVELLED
     - ROAD_WAYPOINTS_POSITION
     - ROAD_WAYPOINTS_HEADING
     - ROAD_WAYPOINTS_LANE_ID
     - ROAD_WAYPOINTS_LANE_WIDTH
     - ROAD_WAYPOINTS_SPEED_LIMIT
     - ROAD_WAYPOINTS_LANE_INDEX
     - ROAD_WAYPOINTS_LANE_OFFSET
     - VIA_DATA_NEAR_VIA_POINTS_POSITION
     - VIA_DATA_NEAR_VIA_POINTS_LANE_INDEX
     - VIA_DATA_NEAR_VIA_POINTS_ROAD_ID
     - VIA_DATA_NEAR_VIA_POINTS_REQUIRED_SPEED
     - VIA_DATA_NEAR_VIA_POINTS_HIT
     - LIDAR_POINT_CLOUD_POINTS
     - LIDAR_POINT_CLOUD_HITS
     - LIDAR_POINT_CLOUD_ORIGIN
     - LIDAR_POINT_CLOUD_DIRECTION
     - VEHICLE_TYPE
     - SIGNALS_LIGHT_STATE
     - SIGNALS_STOP_POINT
     - SIGNALS_LAST_CHANGED
   * - **type**
     - uniform float
     - uniform int
     - uniform int
     - uniform float
     - uniform int
     - uniform int
     - uniform int
     - uniform int
     - uniform int
     - uniform int
     - uniform int
     - uniform int
     - uniform int
     - uniform int
     - uniform vec3
     - uniform vec3
     - uniform float
     - uniform float
     - uniform float
     - uniform float
     - uniform int
     - uniform int
     - uniform int
     - uniform vec2
     - uniform vec2
     - uniform vec2
     - uniform vec2
     - uniform vec2
     - uniform vec2
     - uniform vec3
     - uniform int
     - uniform vec3
     - uniform vec3
     - uniform float
     - uniform float
     - uniform int
     - uniform int
     - uniform int
     - uniform vec3
     - uniform int
     - uniform vec2
     - uniform float
     - uniform int
     - uniform float
     - uniform float
     - uniform int
     - uniform float
     - uniform float
     - uniform vec2
     - uniform float
     - uniform int
     - uniform float
     - uniform float
     - uniform int
     - uniform float
     - uniform vec2
     - uniform int
     - uniform int
     - uniform float
     - uniform int
     - uniform vec3
     - uniform int
     - uniform vec3
     - uniform vec3
     - uniform int
     - uniform int
     - uniform vec2
     - uniform float


.. note::

    See :ref:`observation information <obs_action_reward>` for more information.