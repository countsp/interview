**GNSS Poser**

- 订阅话题 `/sensing/gnss/pose_with_covariance`，获取来自 GNSS 的粗略位姿及其协方差。
- 将该初始位姿用于后续初始化。 

**Pose Initializer**

- 接收 GNSS Poser 发布的粗略位姿，状态切换为 `INITIALIZING`
- 调用 NDT 扫描匹配服务（`init_srv`）对位姿做一次大致校准
- 将校准后的初始位姿通过 `/initialpose3d` 发布给 EKF 本地化器 

**Random Downsample Filter**

- 对传入的原始点云（LiDAR 扫描）做随机下采样，生成 `/localization/util/downsample/pointcloud`
- 降低 NDT 匹配的计算量和时延 

**NDT Scan Matcher**

- 订阅下采样后的点云和初始化位姿，调用 Hybrid NDT 算法：
  1. **align_pose()**：将点云与预先构建的地图做匹配
  2. 发布 **对齐后的点云**（`/points_aligned`，frame=map） rviz可视化
  3. 发布 **插值后的初始位姿**（`/initial_pose_with_covariance`，包含位姿与协方差）
- 当 NDT 初始化完成后，将 `is_completed=true`，供上层判断 

**EKF Localizer**

- 接收 `/initialpose3d`（或 `/initial_pose_with_covariance`）和来自 IMU/里程计的 **gyro_odometer**、**twist_estimator** 数据
- 以固定频率（由 `ekf_dt_` 决定）调用 EKF 的 **预测-更新** 回调（`timerCallback`）
- 融合多源信息后发布：
  - `/localization/pose_twist_fusion_filter/biased_pose_with_covariance`（滤波后带偏差校正的位姿与协方差）
  - `/localization/pose_twist_fusion_filter/twist`（速度/角速度）
  - 发布 `map → base_link` 的 TF 变换 

**Stop Filter**

- 简单检测当前融合的速度或加速度，小于阈值时认为车辆“停稳”
- 发布到后续 **Obstacle Stop Planner**，用于安全停车判断 

**Obstacle Stop Planner**

- 订阅融合后的运动状态（kinematic_state）以及来自**场景规划**的全局/局部轨迹

- 在检测到前方障碍物且距离过近时，发布停车命令，保障安全 

  ![aw.drawio](/home/office2004/Downloads/aw.drawio.png)