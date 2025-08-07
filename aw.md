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

# ekf



1. 用一个简化的一维 EKF 示例来说明整个“预测 → 测量更新”流程。状态向量取为
   $$
   \mathbf x =  \begin{bmatrix} x\\ v \end{bmatrix}
   $$
   

   分别表示“位置”和“速度”，并假设：

   - **初始状态**
     $$
     x_0 =  \begin{bmatrix}0\,\text{m}\\2\,\text{m/s}\end{bmatrix},\quad P_0 = \begin{bmatrix}1 & 0\\0 & 1\end{bmatrix}
     $$
     
   - **过程噪声**
     $$
     Q = \begin{bmatrix}0.1 & 0\\0 & 0.1\end{bmatrix}
     $$
     
   - **测量噪声**

     - 位置测量方差 R_x = 1
     - 速度测量方差 Rv=0.5
   
   - **采样间隔** Δt=1 s

   - **IMU 测得加速度** a=1 m/s2

   - **位置测量（如 NDT 或 GNSS）**  zx=2.6 m

   - **速度测量（如融合后的 Twist）**  zv=2.8 m/s

   ------

   ## 1. 预测（Predict）

   我们采用**匀加速模型**：
   $$
   \begin{aligned} x_{k+1|k} &= x_k + v_k\,\Delta t + \tfrac12\,a\,\Delta t^2,\\ v_{k+1|k} &= v_k + a\,\Delta t. \end{aligned}
   $$
   
   
   代入数值：
   $$
   \begin{aligned} x_{1|0} &= 0 + 2\cdot1 + 0.5\cdot1\cdot1^2 = 2.5\;\text{m},\\ v_{1|0} &= 2 + 1\cdot1 = 3\;\text{m/s}. \end{aligned}
   $$
   
   
   协方差预测（简化为 P1∣0=P0+QP_{1|0}=P_0+QP1∣0=P0+Q）：
   $$
   P_{1|0} =  \begin{bmatrix}1 & 0\\0 & 1\end{bmatrix} + \begin{bmatrix}0.1 & 0\\0 & 0.1\end{bmatrix} = \begin{bmatrix}1.1 & 0\\0 & 1.1\end{bmatrix}
   $$
   
   
   ## 2. 计算卡尔曼增益

   由于测量为“位置”和“速度”两条独立观测，增益可分开计算：
   $$
   \begin{aligned} K_x &= \frac{P_{1|0}(1,1)}{P_{1|0}(1,1) + R_x}   = \frac{1.1}{1.1 + 1} \approx 0.524,\\ K_v &= \frac{P_{1|0}(2,2)}{P_{1|0}(2,2) + R_v}      = \frac{1.1}{1.1 + 0.5} \approx 0.6875. \end{aligned}
   $$
   
   
   ## 3. 测量更新（Measurement Update）

   分别对位置和速度做校正：
   $$
   \begin{aligned} x_{1|1}  &= x_{1|0} + K_x\bigl(z_x - x_{1|0}\bigr) = 2.5 + 0.524\,(2.6 - 2.5) \approx 2.5 + 0.0524 = 2.5524,\\ v_{1|1}  &= v_{1|0} + K_v\bigl(z_v - v_{1|0}\bigr) = 3 + 0.6875\,(2.8 - 3) \approx 3 - 0.1375 = 2.8625. \end{aligned}
   $$
   
   
   更新后的协方差：
   $$
   \begin{aligned} P_{1|1}(1,1) &= (1 - K_x)\,P_{1|0}(1,1)              = (1 - 0.524)\times1.1 \approx 0.524,\\ P_{1|1}(2,2) &= (1 - K_v)\,P_{1|0}(2,2)              = (1 - 0.6875)\times1.1 \approx 0.3438. \end{aligned}
   $$