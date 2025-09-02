# ParkingE2E

### 整体结构

```
self.lss_bev_model = LssBevModel(self.cfg)      # LssBevModel.init() 输出多个相机图像融合后的特征.低层次的 BEV 特征，因为它主要聚焦在从图像 → BEV 空间的几何映射和对齐，不涉及太多 BEV 空间中的上下文语义建模。
        self.image_res_encoder = BevEncoder(in_channel=self.cfg.bev_encoder_in_channel) #对 BEV 图像特征提取高层语义特征 多通道

        # Target Encoder
        self.target_res_encoder = BevEncoder(in_channel=1) #变成一张具有空间语义信息的多通道目标特征图  这个 BEV 热点图本身只是个稀疏的点（像素上只有一小块是非零的），太原始了，不足以提供丰富的空间语义信息。

        # BEV Query
        self.bev_query = BevQuery(self.cfg) #将两个 BEV 特征进行 Transformer 融合，强调目标点区域 它的核心机制就是一个 Transformer Decoder ,cross-attention

        # Trajectory Decoder
        self.trajectory_decoder = self.get_trajectory_decoder() # 预测 token
```
# LSS

```
self.lss_bev_model(images, intrinsics, extrinsics)
        ↓
__call__(...)  # nn.Module 自动定义
        ↓
forward(images, intrinsics, extrinsics)
        ↓
calc_bev_feature(images, intrinsics, extrinsics)
        ├── get_geometry(...)       ← 利用相机内外参和 frustum 获取三维坐标
        ├── encoder_forward(...)    ← EfficientNet将图像编码为特征 + 深度（lift:**真正获取深度**）
        └── proj_bev_feature(...)   ← 将图像特征投影到 BEV 空间（splat:**把特征撒到 BEV 平面**）
        ↓
return bev_feature, pred_depth
```

#### 功能

显式地编码了像素->空间的映射关系，加速训练收敛

**bev_camera, pred_depth = self.lss_bev_model(images, intrinsics, extrinsics)**  #bev语义特征图 每个深度层的概率分布

**内部：**

1.使用 `EfficientNet` 提取每个相机图像的语义特征和（如果启用）像素级深度分布：

2.使用 `create_frustum()` 得到图像空间中的 3D 采样网格（u, v, d），即为每个像素采样多个深度层。

3.将每个像素点 + 深度点投影到世界坐标系下的 3D 点坐标（通过 `intrinsics`、`extrinsics`）

4.多个相机的所有深度层上每个像素位置对应的特征 `x_b` 被投影到 BEV 网格中。相同 BEV 网格（x, y）上多个相机 / 多个深度层落下来的特征会 **聚合（求和）**

### 具体：LssBevModel

self.frustum = self.create_frustum() # 创建视锥体

self.cam_encoder = CamEncoder(self.cfg, self.depth_channel)  #裁减EfficientNetB0

# Encoder

## Image Encoder

bev_camera_encoder = self.image_res_encoder(bev_camera, flatten=False) 

### BEV encoder

就是经过了resnet18

```
trunk = resnet18(weights=None, zero_init_residual=True)

self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
self.bn1 = trunk.bn1
self.relu = trunk.relu
self.max_pool = trunk.maxpool

self.layer1 = trunk.layer1
self.layer2 = trunk.layer2
self.layer3 = trunk.layer3
self.layer4 = trunk.layer4
```


## 为什么用Efficientnet

**大小：**

1.**EfficientNet** 用了 **MBConv + SE + Swish** 等结构，在**同等精度下计算量更低**，同时保留较强的感受野与特征表达能力

2.LSS 是一个**多相机输入、多尺度推理**的结构（6~8 路相机），如果 backbone 太重（如 ResNet101），GPU 显存和延迟都会炸掉

**功能**：

3.抑制无用通道，强化关键信号

- MBConv + SE 会让网络更关注有用通道，小目标、远距纹理细节会保留更多。MBConv 生成了很多种通道特征，SE 让网络能**根据当前图片的整体特征分布**，动态选择最有用的那几个通道来突出
- 这对 LSS 特别重要，因为远处目标在图像里只有几像素大，特征稍微被平滑掉就无法在 BEV 空间还原

---



### EfficientNet

**1.升维 Expansion：**

1×1 Conv（扩展）: Cin → Cin × t =Cexp

```
nn.Conv2d(16, 96, kernel_size=1, bias=False),
```

**提升特征维度：**提供更多中间特征组合，增加表达能力。

**增强 depthwise 卷积效果： **depthwise 不处理通道间信息，扩展后能处理更细致的局部空间特征

​		│
​       ▼

（BatchNorm + Swish）

```
nn.BatchNorm2d(96)
nn.SiLU()
```

​       │

​       ▼

2.**深度可分离卷积：**

每个通道单独使用一个卷积核进行卷积，**大大减少参数量和计算量**。

​		│
​       ▼

（BatchNorm + Swish）

​       │

​       ▼

3.**SE 模块（Squeeze-and-Excitation）**：

提炼 + 降低参数量 & 计算量 ( 16 * 16-> 4 * 16 * 2)

​			**Squeeze（压缩）**：全局平均池化，对每个通道压缩成一个标量

​			**Excitation（激励）**：通过一个两层 MLP 生成每个通道的权重

​			**Scale（重标定）**：用这些权重乘以原特征图，实现通道注意力

**用 `Sigmoid` 的输出（通道注意力权重）去“缩放”输入特征图的每一个通道**。

```
class SEModule(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)  # 输出 (B, C, 1, 1)
        self.fc1 = nn.Linear(channels, channels // reduction, bias=True) 16->4
        self.fc2 = nn.Linear(channels // reduction, channels, bias=True) 4->16
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.pool(x).view(b, c)           # (B, 16)
        y = self.relu(self.fc1(y))            # (B, 4)
        y = self.sigmoid(self.fc2(y))         # (B, 16)
        y = y.view(b, c, 1, 1)                # (B, C, 1, 1)
        return x * y                          # ✅ ← 关键：x * attention（通道加权）
```

**4.Project**

````
nn.Conv2d(96, 24, kernel_size=1, bias=False),
nn.BatchNorm2d(24)
````

**5.Residual**

保留原信息，提升梯度流动

## Trainer

设置训练器 Trainer，包括设备、分布式策略、日志记录、回调、验证频率等

```
def train(config_obj):
    parking_trainer = Trainer(callbacks=setup_callbacks(config_obj),
                              logger=TensorBoardLogger(save_dir=config_obj.log_dir, default_hp_metric=False),
                              accelerator='gpu',
                              strategy='ddp' if config_obj.num_gpus > 1 else None,
                              devices=config_obj.num_gpus,
                              max_epochs=config_obj.epochs,
                              log_every_n_steps=config_obj.log_every_n_steps,
                              check_val_every_n_epoch=config_obj.check_val_every_n_epoch,
                              profiler='simple')#设置训练器 Trainer，包括设备、分布式策略、日志记录、回调、验证频率等；
   
```

定义模型

```
model = ParkingTrainingModelModule(config_obj) # 实例化模型对象  
```

加载数据

```
data = ParkingDataloaderModule(config_obj)
```



## 自动调用

下面这几个方法是 PyTorch Lightning 在 `Trainer.fit(…)`／`.validate(…)` 流程中**自动**调用的钩子（hook）方法：

**pl.LightningModule（ParkingTrainingModuleReal）**

- **`__init__`**
   当你执行 `model = ParkingTrainingModuleReal(cfg)` 时，Python 会调用它来构造对象。
   Lightning 并不会在运行时再额外调用它。
- **`configure_optimizers(self)`**
   在 `Trainer.fit()` 一开始的时候，Lightning 会调用它来从你的模块里拿到：
  1. 优化器（`optimizer`）
  2. 学习率调度器（`lr_scheduler`，如果你返回的话）
- **`training_step(self, batch, batch_idx)`**
   在每个训练 epoch 中，对每个拿到的训练 batch，Lightning 会自动调用这个方法一次。
   你在这里实现了前向计算、loss 计算、`self.log_dict({...})` 和返回 loss。
- **`validation_step(self, batch, batch_idx)`**
   在每个验证 epoch 中，对每个拿到的验证 batch，Lightning 会自动调用这个方法一次。
   你在这里算了验证 loss、指标，并 `self.log_dict({...})`。

**pl.LightningDataModule（Dataloader）**

* setup()
* train_dataloader()
* val_dataloader()



## training step

```
def training_step(self, batch, batch_idx): # automatically processed 
        loss_dict = {}
        pred_traj_point, _, _ = self.parking_model(batch)

        train_loss = self.traj_point_loss_func(pred_traj_point, batch)        

        loss_dict.update({"train_loss": train_loss})

        self.log_dict(loss_dict)

        return train_loss
```

对batch进行推理，输出轨迹

与真值计算loss，返回loss


输入:  [BOS, 5, 9, 4, 2]  # L-1 个token

输出:  [p(5|BOS), p(9|BOS,5), p(4|BOS,5,9), p(2|BOS,5,9,4), p(EOS|BOS,5,9,4,2)]

GT:    [   5   ,     9     ,    4         ,   2           ,       EOS          ]

```
pred = pred[:, :-1, :]   #因为最后一个没有gt

gt   = data[:, 1:-1]  
```






### 数据处理流程图
```
1. 初始化 ParkingDataModuleReal(config, is_train)
   └─ 设定配置、BOS/EOS/PAD token、相机标签等
   └─ 调用 create_gt_data()
2. create_gt_data() 构建训练数据缓存
   ├─ 调用 get_all_tasks()
   │   └─ 根据 is_train 决定使用 training_dir / validation_dir
   │   └─ 遍历目录，收集所有任务路径 task_path
   ├─ 遍历所有 task_path:
   │   ├─ 初始化 CameraInfoParser 和 TrajectoryInfoParser
   │   ├─ 获取相机内参 + 外参 → intrinsic / extrinsic
   │   └─ 遍历每个时间帧（ego_index）:
   │       ├─ ego pose ← world 坐标系下
   │       ├─ 计算 world2ego_mat ← pose 的逆变换
   │       ├─ create_predict_point_gt()
   │       │   ├─ 获取多个 future trajectory pose（世界坐标）
   │       │   ├─ 转为 ego 坐标系
   │       │   ├─ 编码为 token + pad
   │       ├─ create_parking_goal_gt()
   │       │   ├─ 模糊目标点（随机）+ 精确目标点（最终）
   │       │   ├─ 转为 ego 坐标
   │       ├─ create_image_path_gt()
   │       │   └─ 构造图像路径字典：{image_tag: path}
   │       └─ 保存所有图像路径、位姿、token、目标点信息
   └─ 调用 format_transform() → 所有 list → numpy 格式缓存
3. __getitem__(index)
   ├─ 调用 process_camera(index)
   │   ├─ 加载 4 张图像（路径来自 self.images[image_tag]）
   │   ├─ resize → 归一化 → 拼接成一个 tensor（[4C,H,W]）
   │   ├─ 相机内参和外参也转为 tensor 并拼接
   └─ 返回字典：
       {
         "image": 图像拼接 Tensor,
         "intrinsics": 相机内参 Tensor,
         "extrinsics": 相机外参 Tensor,
         "gt_traj_point": 多步轨迹点（浮点坐标）,
         "gt_traj_point_token": 多步 token 编码, 
         "target_point": 精确目标点,
         "fuzzy_target_point": 模糊目标点
       }
4. DataLoader(batch_size, shuffle, num_workers)
   └─ 自动调用 __getitem__() 并打包为 batch，送入模型
5. 模型训练接收的 batch 数据：
   ├─ image: Tensor[B, C*4, H, W]
   ├─ gt_traj_point: Tensor[B, N*2]
   ├─ gt_traj_point_token: Tensor[B, token_len]
   ├─ target_point / fuzzy_target_point: Tensor[B, 2]
   ├─ intrinsics / extrinsics: Tensor[B, ...]

```

## Data

``` 
batch = next(iter(train_loader))
```

取出第一个 batch，并打印出结构

```
  'image': Tensor(shape=(1,4,3,256,256)),
  'extrinsics': Tensor(shape=(1,4,4,4)),
  'intrinsics': Tensor(shape=(1,4,3,3)),
  'target_point': Tensor(shape=(1,2)),
  'gt_traj_point': Tensor(shape=(1,60)),
  'gt_traj_point_token': Tensor(shape=(1,63)),
  'fuzzy_target_point': Tensor(shape=(1,2)),
```

**image**

- shape `(B, C_cams, C_img, H, W)` = `(1, 4, 3, 256, 256)`
- 这里 `B=1`，`4` 是四路摄像头（front/left/right/rear），每路 `3` 通道（RGB），分辨率 `256×256`。

**extrinsics**

- shape `(1, 4, 4, 4)`
- 四路摄像头的 **4×4 齐次外参矩阵**，第一个维度还是 batch。

**intrinsics**

- shape `(1, 4, 3, 3)`
- 四路摄像头的 **3×3 内参矩阵**。

**target_point**

- shape `(1, 2)`
- 当前帧的 **精确停车目标点**，格式 `[x, y]`。

**gt_traj_point**

- shape `(1, 60)`
- 未来 `autoregressive_points * item_number` 个点拼成的回归坐标向量（如 30×2 = 60）。

**gt_traj_point_token**

- shape `(1, 63)`
- 同样长度的 token 序列（含 BOS/EOS/PAD），长度 = `60 + append_token(3)`。

**fuzzy_target_point**

- shape `(1, 2)`
- 模糊停车目标点 `[x, y]`。



# Tokenize

将连续的轨迹点坐标target in ego（如 `(x, y)`）映射为离散的整数 Token。

```
x_normalize = (x + xy_max) / (2 * xy_max)# 将浮点坐标归一化
y_normalize = (y + xy_max) / (2 * xy_max)

return [int(x_normalize * valid_token), int(y_normalize * valid_token), int(progress_normalize * valid_token)]#生成整数 Token
```

valid_token:1200

xy_max:15m

1.25cm分度




## Target encoder

```
bev_target = self.get_target_bev(target_point, mode=mode)
```

计算 BEV 图像大小 h,w -> 初始化空白热力图 (B,1,h,w)  -> 把原点（车头）放到 BEV 图中心 ->除以分辨率 `res` 把米单位转换成网格单位  -> 训练时可加随机偏移（数据增强）->  对每个样本，在以 `(row,col)` 为中心、边长 `2r+1` 的小方块区域内置 `1`，其余保持 `0`。

```
bev_target_encoder = self.target_res_encoder(bev_target, flatten=False)
```

## BEV Query 

##### 将两个 BEV 特征进行 Transformer 融合

``` bev_feature = self.get_feature_fusion(bev_target_encoder, bev_camera_encoder)```

即

```bev_feature = self.bev_query(bev_target_encoder, bev_camera_encoder)```



```bev_feature = self.tf_query(tgt_feature, memory=img_feature)  # Transformer 融合```

执行了：

**自注意力（Self-Attention）**

- 让查询自己内部互相“看”一遍，学习序列内部的依赖。

**跨源注意力（Cross-Attention）**

- 把 `img_feature`（相机 BEV 流程编码的语义空间信息）注入到 `tgt_feature`（目标热力图编码）的表示里。

**前馈网络（Feed-Forward Network）**

- 操作：跨注意力后的每个位置再过两层全连接+激活 (`Linear → GELU/ReLU → Linear`)，增强非线性表达。

**残差 + LayerNorm**

- 每个子层（自注意力、跨注意力、前馈）都有 **残差连接**：输出 = 子层(输入) + 输入
- 紧跟一个 **LayerNorm**，保持梯度稳定。

**多层堆叠**

- `num_layers=self.cfg.query_en_layers`，就重复上面的流程若干次，让融合更深、更灵活。





tgt_feature.shape = (B, C, H, W)

​					|

→ view() 为 (B, C, H*W)

​					|

→ permute → (B, HW, C)

​					|

+self.pos_embed 加位置编码

​					|

self.tf_query(tgt_feature, memory=img_feature)   让 `target BEV` 对 `camera BEV` 做多头注意力

​					|

(B, HW, C) → permute → reshape → (B, C, H, W)



### Trajectory Decoder

```
def forward(self, encoder_out, tgt):
# train: (bev_feature, data['gt_traj_point_token'].cuda())

tgt = tgt[:, :-1]  # 去掉最后一个 token（如 EOS），做 teacher forcing
tgt_mask, tgt_padding_mask = self.create_mask(tgt)

tgt_embedding = self.embedding(tgt)

tgt_embedding = self.pos_drop(tgt_embedding + self.pos_embed)    **先做embedding**

pred_traj_points = self.decoder(encoder_out, tgt_embedding, tgt_mask, tgt_padding_mask)
```



对 `tgt_embedding` 做：

1. 自注意力（masked self-attention）
2. 和 `encoder_out` 做交叉注意力（cross-attention）
3. 前馈网络（MLP）
4. 每一层有残差连接和 LayerNorm



**Teacher Forcing** 是训练序列模型时用真实 token 作为 decoder 的输入（而不是自己预测的），提高训练稳定性。

**假设`gt_traj_point_token` 是：**

```
[BOS, token1, token2, ..., tokenN, EOS]
```

你输入 decoder 的是 `tgt[:, :-1] = [BOS, token1, ..., tokenN]`（用作每一步 decoder 输入）

你监督的是 `tgt[:, 1:] = [token1, ..., tokenN, EOS]`（用作目标输出）做 `CrossEntropyLoss`

**推理时不能用 Teacher Forcing**

> 在 `predict()` 阶段，是 auto-regressive 推理，**只能用模型自己前一步的预测作为当前输入**。

你代码中体现：

```
pred_traj_points = self.output(pred_traj_points)[:, length - offset, :]
...
pred_traj_points = torch.softmax(...).argmax(...)  # 自己选 token
...
tgt = torch.cat([tgt, pred_traj_points], dim=1)  
```

### Masking

```
tgt_mask, tgt_padding_mask = self.create_mask(tgt)
```

告诉 Transformer 在 attention 时忽略 PAD token 的影响（attention score 不计算）

```
tgt_padding_mask = [[False, False, True, True]]
```

告诉 Transformer 在 attention 时忽略 PAD token 的影响（attention score 不计算）



**Embedding**

embedding 是**一个可学习的、随机初始化的查表式 embedding 层**。不是预训练的 word embedding、也不是 positional embedding、也不是 learned codebook embedding

```
self.embedding = nn.Embedding(self.cfg.token_nums + self.cfg.append_token, self.cfg.tf_de_dim)
```

**`nn.Embedding(num_embeddings, embedding_dim)` 的含义：**

| 参数名           | 含义                                                         |
| ---------------- | ------------------------------------------------------------ |
| `num_embeddings` | 你希望能嵌入的 token 总数（即词表大小）                      |
| `embedding_dim`  | 每个 token 对应的嵌入向量维度（即词向量维度）                |
| 初始化           | 默认是 `torch.nn.init.normal_()` 初始化为正态分布（或可手动修改） |
| 学习方式         | 参与反向传播，**可训练**                                     |

**Pose embedding**

```
self.pos_embed = nn.Parameter(torch.randn(1, max_len, hidden_dim) * 0.02)
```

torch.randn(1, L, D) 对所有元素乘以 0.02，通常用于缩小初始值范围，使训练更稳定。

> shape 为 `[1, L, D]` 的张量会自动在 batch 维度上广播成 `[B, L, D]`，然后与 token embedding 对应位置元素相加。

**注意：**

token embedding 用 nn.Embedding 是因为它本质是“查表”（token id → 向量），而 position embedding 用 nn.Parameter 是因为它是一个固定 shape 的可训练矩阵，直接相加，无需查表。



## Transformer Decoder

PyTorch 的 `TransformerDecoder` 接收的维度要求是 `[seq_len, batch, dim]`，所以先转置：

```
encoder_out = encoder_out.transpose(0, 1) 
tgt_embedding = tgt_embedding.transpose(0, 1)
```

然后做decoder

```
pred_traj_points = self.tf_decoder(tgt=tgt_embedding,
                                        memory=encoder_out,
                                        tgt_mask=tgt_mask,
                                        tgt_key_padding_mask=tgt_padding_mask)
```

其中

```
self.tf_decoder = nn.TransformerDecoder(tf_layer, num_layers=self.cfg.tf_de_layers)
```

#### 对 tgt_embedding 进行：

1. Multi-head **self-attention**（带 causal mask）
2. Multi-head **cross-attention** with **encoder_out**(图像)
3. FeedForward + LayerNorm 等标准模块
4. 多层堆叠（`num_layers` 由 config 控制）

输出 shape 同 `tgt_embedding`：`[L, B, D]`



## 🔍 Learnable Positional Embedding vs Fixed

| 对比项               | Learnable PosEmbed                         | Sinusoidal PosEmbed (Fixed) |
| -------------------- | ------------------------------------------ | --------------------------- |
| 是否可训练           | ✅ 是                                       | ❌ 否                        |
| 表达能力             | 高，自由度大                               | 有一定规律性，适合泛化      |
| 能否外推长序列       | 不一定好（受限于训练时长度）               | ✅ 可外推（因函数有周期性）  |
| 用于 NLP/BERT        | ✅ 常见于 BERT、GPT                         | 曾在原始 Transformer 使用   |
| 用于 Trajectory Task | ✅ 常见（因为轨迹长度固定，不要求泛化长度） | 可选                        |

# 训练结构

在**train.py**中，有

```
ParkingTrainingModelModule = get_parking_model(data_mode=config_obj.data_mode, run_mode="train") 
```

在**model_interface.py**中，有

```
model_class = ParkingTrainingModuleReal
```

在**trainer_real.py**中，有 

```
class ParkingTrainingModuleReal(pl.LightningModule):
    self.parking_model = ParkingModelReal(self.cfg)
```

在**parking_model_real.py**中，实例化了使用了功能

```
class ParkingModelReal(nn.Module):
    self.lss_bev_model = LssBevModel(self.cfg)
    self.image_res_encoder = BevEncoder(in_channel=self.cfg.bev_encoder_in_channel)

    # Target Encoder
    self.target_res_encoder = BevEncoder(in_channel=1)

    # BEV Query
    self.bev_query = BevQuery(self.cfg)

    # Trajectory Decoder
    self.trajectory_decoder = self.get_trajectory_decoder()
```



## 优化

**1.过拟合**

做数据增强（加噪，截断，多场景）

**2.在逼近车位时，预测轨迹偏离**

对 target 通道做高斯分布处理， BEV 空间对“停车目标点”生成热图，作为 Encoder 的 Query。用 2D 高斯让网络学习不仅知道“在哪儿”，还知道“置信度随着距离衰减”的分布。

效果：有助于网络学到更精细的侧向定位



## 为什么用CE？

1. 与 Transformer 自回归框架天然契合

​		Transformer Decoder 原生就是**离散 token 的序列到序列建模**

2. 停车轨迹往往存在多种可行路径（多模态）；回归损失（MSE）假设输出是单峰高斯，容易平均化多种解，结果落在“多种轨迹中间”的不合理位置。

3. 回归 L₂ Loss 对于尺度、单位非常敏感，需要精心调整网络的输出范围、学习率；CE Loss 做分类，logits → softmax → log 概率，一般更容易收敛。
4. BEV 空间量化成固定网格，token 本身就代表某个格子索引；直接回归到连续坐标就绕过了网格结构，不容易对齐 BEV 特征和模型的输出。

CE LOSS？

l2 distance

hausdorff distance

