from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from locotouch.config.locotouch.agents.rsl_rl_ppo_cfg import LocomotionPPORunnerCfg


@configclass
class LocomotionGo2WPPORunnerCfg(LocomotionPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "locotouch_go2w"
        self.wandb_project = "Go2W_Locomotion"
        self.max_iterations = 80000





































# # ------------------------ Encoder ------------------------
# @configclass
# class CylinderTransportTeacherObjectStateEncoderPPORunnerCfg(CylinderTransportTeacherPPORunnerCfg):
#     def __post_init__(self):
#         super().__post_init__()
#         self.policy.class_name = "ActorCriticEncoder"
#         self.policy.actor_flatten_obs_end_idx = 270
#         self.policy.actor_encoder_obs_start_idx = 270
#         self.policy.actor_encoder_hidden_dims = [256, 128, 64]
#         self.policy.actor_encoder_embedding_dim = 64
#         self.policy.critic_flatten_obs_end_idx = None
#         self.policy.critic_encoder_obs_start_idx = None
#         self.policy.critic_encoder_hidden_dims = None
#         self.policy.critic_encoder_embedding_dim = None
#         self.policy.encoder_activation = "elu"
#         self.policy.encoder_final_activation = "tanh"

# @configclass
# class CylinderTransportTeacherSingleObjectStateEncoderPPORunnerCfg(CylinderTransportTeacherObjectStateEncoderPPORunnerCfg):
#     def __post_init__(self):
#         super().__post_init__()
#         self.policy.actor_flatten_obs_end_idx = -13*1
#         self.policy.actor_encoder_obs_start_idx = -13*1


# # ------------------------ RNNEncoder ------------------------
# @configclass
# class CylinderTransportTeacherObjectStateRNNEncoderPPORunnerCfg(CylinderTransportTeacherObjectStateEncoderPPORunnerCfg):
#     def __post_init__(self):
#         super().__post_init__()
#         self.policy.class_name = "ActorCriticRNNEncoder"
#         self.policy.critic_flatten_obs_end_idx = self.policy.actor_flatten_obs_end_idx
#         self.policy.critic_encoder_obs_start_idx = self.policy.actor_encoder_obs_start_idx
#         self.policy.critic_encoder_hidden_dims = self.policy.actor_encoder_hidden_dims
#         self.policy.critic_encoder_embedding_dim = self.policy.actor_encoder_embedding_dim
#         self.policy.encoder_rnn_type="gru"
#         self.policy.encoder_rnn_hidden_size=256
#         self.policy.encoder_rnn_num_layers=1






# # ------------------------ Pretrained Transport ------------------------
# # still need to use --resume because the default args_cli.resume is False if not given
# @configclass
# class CylinderTransportTeacherFCPreTrainedPPORunnerCfg(CylinderTransportTeacherFCPPORunnerCfg):
#     def __post_init__(self):
#         super().__post_init__()
#         self.resume = True
#         self.resume_experiment = "locomotion_pretrain"
#         self.pretrained = True



# # ------------------------ Tactile Input ------------------------


# @configclass
# class CylinderTransportTactileInputPPORunnerCfg(CylinderTransportTeacherObjectStateEncoderPPORunnerCfg):
#     def __post_init__(self):
#         super().__post_init__()
#         self.experiment_name = "locotouch_object_transport_teacher_tactile"
#         self.policy.class_name = "ActorCriticPreEncoderRNNEncoder"
#         self.policy.actor_flatten_obs_end_idx = -17*13
#         self.policy.actor_encoder_obs_start_idx = -17*13
#         self.policy.actor_pre_encoder_hidden_dims = [128, 128, 64]
#         self.policy.actor_pre_encoder_embedding_dim = 64
#         self.policy.critic_flatten_obs_end_idx = self.policy.actor_flatten_obs_end_idx
#         self.policy.critic_encoder_obs_start_idx = self.policy.actor_encoder_obs_start_idx
#         self.policy.critic_pre_encoder_hidden_dims = self.policy.actor_pre_encoder_hidden_dims
#         self.policy.critic_pre_encoder_embedding_dim = self.policy.actor_pre_encoder_embedding_dim
#         self.policy.critic_encoder_hidden_dims = self.policy.actor_encoder_hidden_dims
#         self.policy.critic_encoder_embedding_dim = self.policy.actor_encoder_embedding_dim
#         self.policy.pre_encoder_activation="elu"
#         self.policy.encoder_rnn_type="gru"
#         self.policy.encoder_rnn_hidden_size=256
#         self.policy.encoder_rnn_num_layers=1


# @configclass
# class CylinderTransportTactileInputCNNPPORunnerCfg(CylinderTransportTeacherObjectStateEncoderPPORunnerCfg):
#     def __post_init__(self):
#         super().__post_init__()
#         self.experiment_name = "locotouch_object_transport_teacher_tactile"
#         self.policy.class_name = "ActorCriticPreEncoderRNNEncoder"
#         self.policy.actor_flatten_obs_end_idx = -2*17*13
#         self.policy.actor_encoder_obs_start_idx = -2*17*13
#         self.policy.actor_pre_encoder_hidden_dims = [128, 128, 64]
#         self.policy.actor_pre_encoder_embedding_dim = 64
#         self.policy.critic_flatten_obs_end_idx = self.policy.actor_flatten_obs_end_idx
#         self.policy.critic_encoder_obs_start_idx = self.policy.actor_encoder_obs_start_idx
#         self.policy.critic_pre_encoder_hidden_dims = self.policy.actor_pre_encoder_hidden_dims
#         self.policy.critic_pre_encoder_embedding_dim = self.policy.actor_pre_encoder_embedding_dim
#         self.policy.critic_encoder_hidden_dims = self.policy.actor_encoder_hidden_dims
#         self.policy.critic_encoder_embedding_dim = self.policy.actor_encoder_embedding_dim
#         self.policy.pre_encoder_activation="elu"
#         self.policy.encoder_rnn_type="gru"
#         self.policy.encoder_rnn_hidden_size=256
#         self.policy.encoder_rnn_num_layers=1



# @configclass
# class CylinderTransportTactileInputAsymmetricPPORunnerCfg(CylinderTransportTactileInputPPORunnerCfg):
#     def __post_init__(self):
#         super().__post_init__()
#         self.experiment_name = "locotouch_object_transport_teacher_tactile"
#         self.policy.class_name = "ActorCriticPreEncoderRNNEncoder"
#         self.policy.critic_flatten_obs_end_idx = None
#         self.policy.critic_encoder_obs_start_idx = None
#         self.policy.critic_pre_encoder_hidden_dims = None
#         self.policy.critic_pre_encoder_embedding_dim = None
#         self.policy.critic_encoder_hidden_dims = None
#         self.policy.critic_encoder_embedding_dim = None


# @configclass
# class CylinderTransportTactileDoubleChannelsPPORunnerCfg(CylinderTransportTactileInputPPORunnerCfg):
#     def __post_init__(self):
#         super().__post_init__()
#         self.policy.actor_flatten_obs_end_idx = -17*13*2
#         self.policy.actor_encoder_obs_start_idx = -17*13*2
#         self.policy.critic_flatten_obs_end_idx = self.policy.actor_flatten_obs_end_idx
#         self.policy.critic_encoder_obs_start_idx = self.policy.actor_encoder_obs_start_idx

# @configclass
# class CylinderTransportTactileDoubleChannelsAsymmetricPPORunnerCfg(CylinderTransportTactileInputAsymmetricPPORunnerCfg):
#     def __post_init__(self):
#         super().__post_init__()
#         self.policy.actor_flatten_obs_end_idx = -17*13*2
#         self.policy.actor_encoder_obs_start_idx = -17*13*2


# # ------------------------ Pretrained Locomotion ------------------------

# @configclass
# class ObjectTransportLocomotionPretrainPPORunnerCfg(LocomotionPPORunnerCfg):
#     def __post_init__(self):
#         super().__post_init__()
#         self.experiment_name = "locomotion_pretrain"
#         self.logger = "tensorboard"





# # ------------------------ Other Objects ------------------------


# @configclass
# class CuboidTransportTeacherPPORunnerCfg(ObjectTransportTeacherPPORunnerCfg):
#     def __post_init__(self):
#         super().__post_init__()
#         self.experiment_name = "locotouch_cuboid_transport_teacher"

# @configclass
# class SphereTransportTeacherPPORunnerCfg(ObjectTransportTeacherPPORunnerCfg):
#     def __post_init__(self):
#         super().__post_init__()
#         self.experiment_name = "locotouch_sphere_transport_teacher"

# @configclass
# class RandCuboidTransportTeacherPPORunnerCfg(ObjectTransportTeacherPPORunnerCfg):
#     def __post_init__(self):
#         super().__post_init__()
#         self.experiment_name = "locotouch_rand_cuboid_transport_teacher"



# @configclass
# class RandSphereTransportTeacherPPORunnerCfg(ObjectTransportTeacherPPORunnerCfg):
#     def __post_init__(self):
#         super().__post_init__()
#         self.experiment_name = "locotouch_rand_sphere_transport_teacher"





