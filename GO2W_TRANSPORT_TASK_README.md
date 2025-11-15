# Go2W 轮腿机器人运载任务说明

## 📋 任务概述

这是一个基于 **Go2W 轮腿机器人**的随机圆柱体运输任务，与 Go1 版本保持最大程度的一致性，仅适配机器人构型差异。

### 🎯 设计目标

- ✅ **保持奖励函数一致**：与 `rand_cylinder_transport_no_tactile_test_env_cfg` 完全相同
- ✅ **保持任务设置一致**：圆柱体尺寸、放置方式、终止条件等
- ✅ **只适配机器人差异**：关节数量、控制模式、观察空间

## 🤖 Go2W vs Go1 关键差异

| 特性 | Go1 | Go2W |
|------|-----|------|
| **关节数量** | 12 个（纯腿部） | 16 个（12腿 + 4轮） |
| **控制模式** | 位置控制 | 位置控制（腿）+ 速度控制（轮） |
| **动作维度** | 12D | 16D |
| **足端命名** | `a__FL_foot` (双下划线) | `FL_foot` (单下划线) |
| **观察处理** | 标准关节位置观察 | 轮子位置置零（`joint_pos_rel_without_wheel`） |
| **驱动器** | 全部 DCMotor | 腿部 DCMotor + 轮子 ImplicitActuator |

## 📁 创建的文件

### 1. 环境配置
```
locotouch/config/locotouch/rand_cylinder_transport_go2w_test_env_cfg.py
```
- `RandCylinderTransportGo2WTestEnvCfg`: 训练配置
- `RandCylinderTransportGo2WTestEnvCfg_PLAY`: 测试配置

### 2. PPO 训练配置
```
locotouch/config/locotouch/agents/rsl_rl_ppo_cfg.py
```
- 新增 `RandCylinderTransportGo2WTestPPORunnerCfg` 类

### 3. 任务注册
```
locotouch/config/locotouch/__init__.py
```
- 训练任务: `Isaac-RandCylinderTransportGo2WTest-LocoTouch-v1`
- 测试任务: `Isaac-RandCylinderTransportGo2WTest-LocoTouch-Play-v1`

## 🔧 关键实现细节

### 1. 关节配置

```python
leg_joint_names = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
]  # 12 个腿部关节

wheel_joint_names = [
    "FR_foot_joint", "FL_foot_joint", 
    "RR_foot_joint", "RL_foot_joint",
]  # 4 个轮子关节
```

### 2. 观察空间适配

```python
# 使用 Go2W 专用的观察函数
self.observations.policy.joint_pos.func = mdp_go2w.joint_pos_rel_without_wheel

# 原因：轮子可以无限旋转，位置没有意义，只有速度有意义
# 函数会将轮子的位置观察置零
```

### 3. 动作空间设计

```python
# 腿部：位置控制（12D）
self.actions.joint_pos.joint_names = leg_joint_names
self.actions.joint_pos.scale = {
    ".*_hip_joint": 0.125,      # hip 关节更小的动作幅度
    "^(?!.*_hip_joint).*": 0.25  # 其他腿部关节
}

# 轮子：速度控制（4D）
# 注意：需要确保父类 ActionsCfg 支持 joint_vel
```

### 4. Body 命名适配

```python
# Go2W 的 body 命名（注意：没有双下划线）
躯干: "base"
足端: "FL_foot", "FR_foot", "RL_foot", "RR_foot"

# 步态配对（对角线 Trot）
synced_feet_pair_names = (
    ("FR_foot", "RL_foot"),  # 右前 + 左后
    ("FL_foot", "RR_foot"),  # 左前 + 右后
)
```

## 🚀 使用方法

### 方案 A：直接训练（不推荐，难度高）

```bash
# 小规模测试
python locotouch/scripts/train.py \
    --task Isaac-RandCylinderTransportGo2WTest-LocoTouch-v1 \
    --num_envs=20 \
    --logger=tensorboard

# 大规模训练
python locotouch/scripts/train.py \
    --task Isaac-RandCylinderTransportGo2WTest-LocoTouch-v1 \
    --num_envs=4096 \
    --headless \
    --max_iterations=20000
```

### 方案 B：使用预训练（强烈推荐）

#### 步骤 1：训练 Go2W Locomotion（5k-10k 迭代）

```bash
# 检查是否已有 Go2W locomotion 任务
python locotouch/scripts/list_envs.py | grep Go2W

# 如果有，训练基础运动
python locotouch/scripts/train.py \
    --task <Go2W_Locomotion_Task> \
    --num_envs=4096 \
    --headless \
    --max_iterations=10000
```

#### 步骤 2：从预训练开始训练运输任务

```bash
python locotouch/scripts/train.py \
    --task Isaac-RandCylinderTransportGo2WTest-LocoTouch-v1 \
    --num_envs=4096 \
    --headless \
    --resume \
    --load_run=<locomotion_run_folder> \
    --load_checkpoint=model_10000.pt \
    --max_iterations=15000
```

### 测试训练好的模型

```bash
python locotouch/scripts/play.py \
    --task Isaac-RandCylinderTransportGo2WTest-LocoTouch-Play-v1 \
    --num_envs=20 \
    --load_run=<your_trained_model_folder>
```

## 📊 观察空间维度分析

```python
Go2W 观察空间（与 Go1 基本一致）:

本体感觉:
- 基座角速度: 3D × 6 历史 = 18D
- 重力投影: 3D × 6 历史 = 18D
- 速度命令: 3D × 6 历史 = 18D
- 关节位置: 16D × 6 历史 = 96D  ← 比 Go1 多 4×6=24D
- 关节速度: 16D × 6 历史 = 96D  ← 比 Go1 多 4×6=24D
- 上一步动作: 16D × 6 历史 = 96D  ← 比 Go1 多 4×6=24D

物体状态:
- 相对位置/速度/姿态/角速度: 13D × 6 历史 = 78D

总计: 18+18+18+96+96+96+78 = 420D
（Go1 是 348D，Go2W 多了 72D）
```

## ⚠️ 潜在问题和注意事项

### 1. 动作空间配置

**问题**：父类 `ObjectTransportTeacherEnvCfg` 可能不支持 `joint_vel` 动作。

**解决方案**：
- 检查父类是否定义了 `joint_vel`
- 如果没有，需要手动添加或修改动作配置

**检查方法**：
```bash
# 尝试运行，如果报错则需要修改
python locotouch/scripts/train.py \
    --task Isaac-RandCylinderTransportGo2WTest-LocoTouch-v1 \
    --num_envs=20
```

### 2. 网络容量

Go2W 的观察空间从 348D 增加到 420D，可能需要更大的网络：

```python
# 可选：在 PPO 配置中增加网络容量
self.policy.actor_hidden_dims = [768, 512, 256]  # 默认是 [512, 256, 128]
self.policy.critic_hidden_dims = [768, 512, 256]
```

### 3. 轮腿协调的挑战

- **轮式运动**会导致更大的加速度
- **物体更容易滑动**或倾倒
- 可能需要：
  - 更强的物体稳定性奖励
  - 更平滑的动作惩罚
  - 更严格的速度限制

### 4. 训练难度预期

| 难度因素 | Go1 | Go2W | 说明 |
|---------|-----|------|------|
| 控制复杂度 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 轮腿混合控制 |
| 动作空间 | 12D | 16D | +33% 维度 |
| 观察空间 | 348D | 420D | +20% 维度 |
| 物理稳定性 | ⭐⭐⭐ | ⭐⭐⭐⭐ | 轮子加速更剧烈 |
| 预期训练时间 | 15k 迭代 | 20k+ 迭代 | 约 +30% |

## 🔍 调试建议

### 1. 首次运行检查

```bash
# 检查任务是否正确注册
python -c "import gymnasium as gym; print('Isaac-RandCylinderTransportGo2WTest-LocoTouch-v1' in gym.envs.registry)"

# 小规模测试（观察初始化）
python locotouch/scripts/train.py \
    --task Isaac-RandCylinderTransportGo2WTest-LocoTouch-v1 \
    --num_envs=4 \
    --max_iterations=10
```

### 2. 常见错误排查

**错误 1**: `AttributeError: 'ActionsCfg' object has no attribute 'joint_vel'`
- **原因**: 父类不支持轮子速度控制
- **解决**: 需要从 Go2W 的 locomotion 基类继承

**错误 2**: Body 名称匹配错误
- **原因**: Go2W 的 body 命名与 Go1 不同
- **解决**: 已在配置中修复，检查错误信息中的具体 body 名称

**错误 3**: 观察空间维度不匹配
- **原因**: 网络输入期望的维度与实际观察不符
- **解决**: 检查 `joint_pos_rel_without_wheel` 函数是否正确调用

### 3. 可视化检查

```bash
# 不使用 headless，观察机器人行为
python locotouch/scripts/train.py \
    --task Isaac-RandCylinderTransportGo2WTest-LocoTouch-v1 \
    --num_envs=4 \
    --max_iterations=100
```

观察重点：
- [ ] 机器人是否能站立？
- [ ] 轮子是否在转动？
- [ ] 物体是否正确放置在背上？
- [ ] 是否有不合理的碰撞或穿透？

## 📚 参考文档

- Go1 无触觉任务: `rand_cylinder_transport_no_tactile_test_env_cfg.py`
- Go2W Locomotion: `locotouch/config/locomotion_go2w/`
- Go2W MDP: `locotouch/mdp_go2w/`
- Go2W 资产: `locotouch/assets/go2w.py`

## 🎯 下一步建议

1. **测试任务注册**
   ```bash
   python locotouch/scripts/list_envs.py | grep Go2W
   ```

2. **小规模验证**
   ```bash
   python locotouch/scripts/train.py \
       --task Isaac-RandCylinderTransportGo2WTest-LocoTouch-v1 \
       --num_envs=20
   ```

3. **根据错误信息调整配置**
   - 如果有动作空间问题 → 修改父类继承
   - 如果有观察空间问题 → 检查 mdp_go2w 函数
   - 如果有命名问题 → 检查 body_names

4. **训练 Locomotion 预训练**（如果直接训练失败）

5. **大规模训练并监控指标**

---

**创建时间**: 2025-11-12  
**作者**: AI Assistant  
**版本**: 1.0

