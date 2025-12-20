import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg


Go2W_TRANSPORT_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,  # 基座不固定，受力学影响可移动
        merge_fixed_joints=True,  # 合并固定关节
        replace_cylinders_with_capsules=False,  # 不将圆柱体替换为胶囊体
        asset_path=f"locotouch/assets/go2w_transport/urdf/go2w_transport.urdf",
        activate_contact_sensors=True,  # 激活接触传感器
        rigid_props=sim_utils.RigidBodyPropertiesCfg(  # 刚体属性配置
            disable_gravity=False,  # 启用重力
            retain_accelerations=False,  # 不保留上一帧加速度
            linear_damping=0.0,  #
            angular_damping=0.0, # 0阻尼
            max_linear_velocity=1000.0,  # 速度很大防止被裁剪
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0, # 最大穿透速度
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, # TODO 自碰撞, go1中设置为True, robot_lab中设置为False
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
            contact_offset=1.0e-9,
            rest_offset=-0.004,
        ),  # TODO robot_lab 中未显式配置该属性
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),  # 用于 URDF->Isaac 转换时的关节驱动初始增益，这里把 PD 增益置为 0（即让后端驱动配置负责控制）
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.45),  # 基座初始位置
        joint_pos={  # 各关节初始位置
            ".*L_hip_joint": 0.0,
            ".*R_hip_joint": -0.0,
            "F.*_thigh_joint": 0.8,
            "R.*_thigh_joint": 0.8,
            ".*_calf_joint": -1.5,
            ".*_foot_joint": 0.0,
        },
        joint_vel={".*": 0.0},  # 各关节初始速度
    ),
    soft_joint_pos_limit_factor=0.9,  # 软限位系数（与 go1 的 0.95 不同），会影响关节限制的松紧
    actuators={
        "legs": DCMotorCfg(
            joint_names_expr=["^(?!.*_foot_joint).*"],
            effort_limit=23.5,
            saturation_effort=23.5,
            velocity_limit=30.0,
            stiffness=25.0,
            damping=0.5,
            friction=0.0,
        ),
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=[".*_foot_joint"],
            effort_limit_sim=23.5,
            velocity_limit_sim=30.0,
            stiffness=0.0,
            damping=0.5,
            friction=0.0,
        ),
    },
)
"""Configuration of Unitree Go2W using DC motor.
"""


Go2W_TRANSPORT_PLAY_CFG = Go2W_TRANSPORT_CFG.copy()