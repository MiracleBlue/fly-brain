# Locomotion

An object of the `Fly` class is an instance of a simulated fly. There can be multiple flies in a `Simulation`. The `Fly` object contains parameters related to the fly but not the whole simulation: for example, the set of actuated joints, spawn position, joint actuator parameters (stiffness, damping, etc.), and initial pose.

## `flygym.Fly`

```python
class flygym.Fly(
    name: str | None = None,
    actuated_joints: list = preprogrammed.all_leg_dofs,
    monitored_joints: list = preprogrammed.all_leg_dofs,
    contact_sensor_placements: list = preprogrammed.all_tarsi_links,
    xml_variant: str | Path = 'seqik',
    spawn_pos: tuple[float, float, float] = (0.0, 0.0, 0.5),
    spawn_orientation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    control: str = 'position',
    init_pose: str | KinematicPose = 'stretch',
    floor_collisions: str | list[str] = 'legs',
    self_collisions: str | list[str] = 'legs',
    detect_flip: bool = False,
    joint_stiffness: float = 0.05,
    joint_damping: float = 0.06,
    non_actuated_joint_stiffness: float = 1.0,
    non_actuated_joint_damping: float = 1.0,
    neck_stiffness: float | None = 10.0,
    actuator_gain: float | list = 45.0,
    actuator_forcerange: float | tuple[float, float] | list = 65.0,
    tarsus_stiffness: float = 7.5,
    tarsus_damping: float = 1e-2,
    friction: tuple[float, float, float] = (1.0, 0.005, 0.0001),
    contact_solref: tuple[float, float] = (2e-4, 1e3),
    contact_solimp: tuple[float, float, float, float, float] = (0.999, 0.9999, 0.001, 0.5, 2.0),
    enable_olfaction: bool = False,
    enable_vision: bool = False,
    render_raw_vision: bool = False,
    vision_refresh_rate: int = 500,
    enable_adhesion: bool = False,
    adhesion_force: float = 40,
    draw_adhesion: bool = False,
    draw_sensor_markers: bool = False,
    neck_kp: float | None = None,
    head_stabilisation_model: Callable | str | None = None
)
```

A NeuroMechFly environment using MuJoCo as the physics engine.

### Parameters

- `name` (`str`, optional): 
    The name of the fly model. Will be automatically generated if not provided.
- `actuated_joints` (`list[str]`, optional): 
    List of names of actuated joints. By default all active leg DoFs.
- `monitored_joints` (`list[str]`, optional): 
    List of names of joints to monitor with sensors. By default all active leg DoFs.
- `contact_sensor_placements` (`list[str]`, optional): 
    List of body segments where contact sensors are placed. By default all tarsus segments.
- `xml_variant` (`str` or `Path`, optional): 
    The variant of the fly model to use. Multiple variants exist because when replaying experimentally recorded behaviour, the ordering of DoF angles in multi-DoF joints depends on how they are configured in the upstream inverse kinematics program. Two variants are provided: "seqik" (default) and "deepfly3d" (for legacy data produced by DeepFly3D, Gunel et al., eLife, 2019). The ordering of DoFs can be seen from the XML files under `flygym/data/mjcf/`.
- `spawn_pos` (`tuple[float, float, float]`, optional): 
    The (x, y, z) position in the arena defining where the fly will be spawn, in mm. By default (0, 0, 0.5).
- `spawn_orientation` (`tuple[float, float, float]`, optional): 
    The spawn orientation of the fly in the Euler angle format: (x, y, z), where x, y, z define the rotation around x, y and z in radian. By default (0.0, 0.0, 0.0). In the default configuration, the fly is spawned facing the +x direction; the +y direction is on the fly's left.
- `control` (`str`, optional): 
    The joint controller type. Can be "position", "velocity", or "motor", by default "position".
- `init_pose` (`BaseState`, optional): 
    Which initial pose to start the simulation from. By default "stretch" kinematic pose with all legs fully stretched.
- `floor_collisions` (`str` or `list[str]`, optional): 
    Which set of collisions should collide with the floor. Can be "all", "legs", "tarsi" or a list of body names. By default "legs".
- `self_collisions` (`str` or `list[str]`, optional): 
    Which set of collisions should collide with each other. Can be "all", "legs", "legs-no-coxa", "tarsi", "none", or a list of body names. By default "legs".
- `detect_flip` (`bool`): 
    [Deprecated] Fly flips are now detected regardless of this parameter. This will be removed in future releases.
- `joint_stiffness` (`float`): 
    Stiffness of actuated joints, by default 0.05.
- `joint_damping` (`float`): 
    Damping coefficient of actuated joints, by default 0.06.
- `non_actuated_joint_stiffness` (`float`): 
    Stiffness of non-actuated joints, by default 1.0. If set to 0, the DoF would passively drift over time. Therefore it is set explicitly here for better stability.
- `non_actuated_joint_damping` (`float`): 
    Damping coefficient of non-actuated joints, by default 1.0. Similar to `non_actuated_joint_stiffness`, it is set explicitly here for better stability.
- `neck_stiffness` (`Union[float, None]`): 
    Stiffness of the neck joints (`joint_Head`, `joint_Head_roll`, and `joint_Head_yaw`), by default 10.0. The head joints have their stiffness set separately, typically to a higher value than the other non-actuated joints, to ensure that the visual input is not perturbed by unintended passive head movements. If set, this value overrides `non_actuated_joint_stiffness`.
- `actuator_gain` (`Union[float, list[float]]`): 
    Gain of the actuator: If `control` is "position", it is the position gain of the actuators. If `control` is "velocity", it is the velocity gain of the actuators. If `control` is "motor", it is not used. If the actuator gain is a list, it needs to be of same length as the number of actuated joints and will be applied to every joint.
- `actuator_forcerange` (`Union[float, tuple[float, float], list]`): 
    The force limit of the actuators. If a single value is provided, it will be symmetrically applied to all actuators (-a, a). If a tuple is provided, the first value is the lower limit and the second value is the upper limit. If a list is provided, it should have the same length as the number of actuators. By default 65.0.
- `tarsus_stiffness` (`float`): 
    Stiffness of the passive, compliant tarsus joints, by default 7.5.
- `tarsus_damping` (`float`): 
    Damping coefficient of the passive, compliant tarsus joints, by default 1e-2.
- `friction` (`float`): 
    Sliding, torsional, and rolling friction coefficients, by default (1, 0.005, 0.0001).
- `contact_solref` (`tuple[float, float]`): 
    MuJoCo contact reference parameters (see MuJoCo documentation for details). By default (9.99e-01, 9.999e-01, 1.0e-03, 5.0e-01, 2.0e+00). Under the default configuration, contacts are very stiff. This is to avoid penetration of the leg tips into the ground when leg adhesion is enabled. The user might want to decrease the stiffness if the stability becomes an issue.
- `contact_solimp` (`tuple[float, float, float, float, float]`): 
    MuJoCo contact reference parameters (see MuJoCo docs for details). By default (9.99e-01, 9.999e-01, 1.0e-03, 5.0e-01, 2.0e+00). Under the default configuration, contacts are very stiff. This is to avoid penetration of the leg tips into the ground when leg adhesion is enabled. The user might want to decrease the stiffness if the stability becomes an issue.
- `enable_olfaction` (`bool`): 
    Whether to enable olfaction, by default `False`.
- `enable_vision` (`bool`): 
    Whether to enable vision, by default `False`.
- `render_raw_vision` (`bool`): 
    If `enable_vision` is `True`, whether to render the raw vision (raw pixel values before binning by ommatidia), by default `False`.
- `vision_refresh_rate` (`int`): 
    The rate at which the vision sensor is updated, in Hz, by default 500.
- `enable_adhesion` (`bool`): 
    Whether to enable adhesion. By default `False`.
- `adhesion_force` (`float`): 
    The magnitude of the adhesion force. By default 20.
- `draw_adhesion` (`bool`): 
    Whether to signal that adhesion is on by changing the colour of the concerned leg. By default `False`.
- `draw_sensor_markers` (`bool`): 
    If `True`, coloured spheres will be added to the model to indicate the positions of the cameras (for vision) and odour sensors. By default `False`.
- `neck_kp` (`float`, optional): 
    Position gain of the neck position actuators. If supplied, this will overwrite `actuator_kp` for the neck actuators. Otherwise, `actuator_kp` will be used.
- `head_stabilisation_model` (`Callable` or `str`, optional): 
    If callable, it should be a function that, given the observation, predicts signals that need to be applied to the neck DoFs to stabilises the head of the fly. If "thorax", the rotation (roll and pitch) of the thorax is inverted and applied to the head by the neck actuators. If `None` (default), no head stabilisation is applied.

### Attributes

- `name` (`str`): 
    The name of the fly model.
- `actuated_joints` (`list[str]`): 
    List of names of actuated joints.
- `contact_sensor_placements` (`list[str]`): 
    List of body segments where contact sensors are placed.
- `spawn_pos` (`tuple[float, float, float]`): 
    The (x, y, z) position in the arena defining where the fly will be spawn, in mm.
- `spawn_orientation` (`tuple[float, float, float, float]`): 
    The spawn orientation of the fly in the Euler angle format: (x, y, z), where x, y, z define the rotation around x, y and z in radian. If the spawn orientation is (0, 0, 0), the fly is spawned facing the +x direction; the +y direction is on the fly's left.
- `control` (`str`): 
    The joint controller type. Can be "position", "velocity", or "torque".
- `init_pose` (`flygym.state.BaseState`): 
    Which initial pose to start the simulation from.
- `floor_collisions` (`str`): 
    Which set of collisions should collide with the floor. Can be "all", "legs", "tarsi" or a list of body names.
- `self_collisions` (`str`): 
    Which set of collisions should collide with each other. Can be "all", "legs", "legs-no-coxa", "tarsi", "none", or a list of body names.
- `detect_flip` (`bool`): 
    [Deprecated] Fly flips are now detected regardless of this parameter. This will be removed in future releases.
- `joint_stiffness` (`float`): 
    Stiffness of actuated joints.
- `joint_damping` (`float`): 
    Damping coefficient of actuated joints.
- `non_actuated_joint_stiffness` (`float`): 
    Stiffness of non-actuated joints.
- `non_actuated_joint_damping` (`float`): 
    Damping coefficient of non-actuated joints.
- `neck_stiffness` (`Union[float, None]`): 
    Stiffness of the neck joints (`joint_Head`, `joint_Head_roll`, and `joint_Head_yaw`), by default 10.0. The head joints have their stiffness set separately, typically to a higher value than the other non-actuated joints, to ensure that the visual input is not perturbed by unintended passive head movements. If set, this value overrides `non_actuated_joint_stiffness`.
- `tarsus_stiffness` (`float`): 
    Stiffness of the passive, compliant tarsus joints.
- `tarsus_damping` (`float`): 
    Damping coefficient of the passive, compliant tarsus joints.
- `friction` (`float`): 
    Sliding, torsional, and rolling friction coefficients.
- `contact_solref` (`tuple[float, float]`): 
    MuJoCo contact reference parameters (see MuJoCo documentation for details). Under the default configuration, contacts are very stiff. This is to avoid penetration of the leg tips into the ground when leg adhesion is enabled. The user might want to decrease the stiffness if the stability becomes an issue.
- `contact_solimp` (`tuple[float, float, float, float, float]`): 
    MuJoCo contact reference parameters (see MuJoCo docs for details). Under the default configuration, contacts are very stiff. This is to avoid penetration of the leg tips into the ground when leg adhesion is enabled. The user might want to decrease the stiffness if the stability becomes an issue.
- `enable_olfaction` (`bool`): 
    Whether to enable olfaction.
- `enable_vision` (`bool`): 
    Whether to enable vision.
- `render_raw_vision` (`bool`): 
    If `enable_vision` is `True`, whether to render the raw vision (raw pixel values before binning by ommatidia).
- `vision_refresh_rate` (`int`): 
    The rate at which the vision sensor is updated, in Hz.
- `enable_adhesion` (`bool`): 
    Whether to enable adhesion.
- `adhesion_force` (`float`): 
    The magnitude of the adhesion force.
- `draw_adhesion` (`bool`): 
    Whether to signal that adhesion is on by changing the colour of the concerned leg.
- `draw_sensor_markers` (`bool`): 
    If `True`, coloured spheres will be added to the model to indicate the positions of the cameras (for vision) and odour sensors.
- `head_stabilisation_model` (`Callable` or `str`, optional): 
    If callable, it should be a function that, given the observation, predicts signals that need to be applied to the neck DoFs to stabilises the head of the fly. If "thorax", the rotation (roll and pitch) of the thorax is inverted and applied to the head by the neck actuators. If `None`, no head stabilisation is applied.
- `retina` (`flygym.vision.Retina`): 
    The retina simulation object used to render the fly's visual inputs.
- `arena_root` (`dm_control.mjcf.RootElement`): 
    The root element of the arena.
- `action_space` (`gymnasium.core.ObsType`): 
    Definition of the fly's action space.
- `observation_space` (`gymnasium.core.ObsType`): 
    Definition of the fly's observation space.
- `model` (`dm_control.mjcf.RootElement`): 
    The MuJoCo model.
- `vision_update_mask` (`np.ndarray`): 
    The refresh frequency of the visual system is often lower than the physics simulation time step.

### Methods

#### `change_segment_colour(physics: dm_control.mjcf.Physics, segment: str, colour)`

Change the colour of a segment of the fly.

**Parameters:**

- `physics` (`mjcf.Physics`): The physics object of the simulation.
- `segment` (`str`): The name of the segment to change the colour of.
- `colour` (`tuple[float, float, float, float]`): Target colour as RGBA values normalised to [0, 1].

#### `close()`

Release resources allocated by the environment.

#### `get_info()`

Any additional information that is not part of the observation. This method always returns an empty dictionary unless extended by the user.

**Returns:**

- `dict[str, Any]`: The dictionary containing additional information.

#### `get_observation(sim: Simulation) -> ObsType`

Get observation without stepping the physics simulation.

**Returns:**

- `ObsType`: The observation as defined by the environment.

#### `get_reward()`

Get the reward for the current state of the environment. This method always returns 0 unless extended by the user.

**Returns:**

- `float`: The reward.

#### `init_floor_contacts(arena: BaseArena)`

Initialise contacts between the fly and the floor. This is called by the `Simulation` after the fly is placed in the arena and before setting up the physics engine.

**Parameters:**

- `arena` (`BaseArena`): The arena in which the fly is placed.

#### `is_terminated()`

Whether the episode has terminated due to factors that are defined within the Markov Decision Process (e.g. task completion/ failure, etc.). This method always returns `False` unless extended by the user.

**Returns:**

- `bool`: Whether the simulation is terminated.

#### `is_truncated()`

Whether the episode has terminated due to factors beyond the Markov Decision Process (e.g. time limit, etc.). This method always returns `False` unless extended by the user.

**Returns:**

- `bool`: Whether the simulation is truncated.

#### `post_init(sim: Simulation)`

Initialise attributes that depend on the arena or physics of the simulation.

**Parameters:**

- `sim` (`Simulation`): Simulation object.

#### `update_colours(physics: dm_control.mjcf.Physics)`

Update the colours of the fly's body segments. This is typically called by `Simulation.render` to update the colours of the fly before the cameras do the rendering.

**Parameters:**

- `physics` (`mjcf.Physics`): The physics object of the simulation.

#### `@property vision_update_mask`

The refresh frequency of the visual system is often lower than the physics simulation time step. This 1D mask, whose size is the same as the number of simulation time steps, indicates in which time steps the visual inputs have been refreshed. In other words, the visual input frames where this mask is `False` are repetitions of the previous updated visual input frames.

