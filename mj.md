你正在为一个多模态感知网络准备输入并进行前向推理。请实现函数 `preprocess_and_forward`，完成以下任务：

------

**输入：**

- `image`: RGB 图像，形状为 `(H, W, 3)`，像素值范围为 `[0, 255]`
- `point_cloud`: 激光点云，形状为 `(N, 3)`
- `max_points`: 最大支持点数（采样或补零）

------

**步骤：**

1. **图像处理：**
   - 计算每通道的均值和标准差，对图像归一化；
   - 对归一化后的图像执行全局平均池化，得到图像全局特征 `(3,)`
2. **点云处理：**
   - 采样或补零至 `(max_points, 3)`
3. **多模态融合：**
   - 将图像全局特征复制为 `(max_points, 3)`；
   - 与点云 `(max_points, 3)` 在最后维度拼接，得到融合输入 `(max_points, 6)`
4. **前向网络模拟：**
   - 用 NumPy 构造线性层：
     - 权重 `W` 形状为 `(4, 6)`；
     - 偏置 `b` 形状为 `(4,)`；
     - 随机种子为 42；
   - 输出结果为 `(max_points, 4)`

```
import numpy as np

def preprocess_and_forward(image, point_cloud, max_points):
    # Step 1: 图像处理
    # 归一化
    image = image.astype(np.float32)
    mean = image.mean(axis=(0, 1), keepdims=True)  # (1, 1, 3)
    std = image.std(axis=(0, 1), keepdims=True) + 1e-6
    image_norm = (image - mean) / std

    # 全局平均池化，得到 (3,)
    image_feature = image_norm.mean(axis=(0, 1))  # shape: (3,)

    # Step 2: 点云处理
    N = point_cloud.shape[0]
    if N >= max_points:
        indices = np.random.choice(N, max_points, replace=False)
        point_cloud_processed = point_cloud[indices]
    else:
        padding = np.zeros((max_points - N, 3), dtype=point_cloud.dtype)
        point_cloud_processed = np.concatenate([point_cloud, padding], axis=0)

    # Step 3: 多模态融合
    image_feature_expand = np.tile(image_feature, (max_points, 1))  # shape: (max_points, 3)
    fused_input = np.concatenate([point_cloud_processed, image_feature_expand], axis=-1)  # (max_points, 6)

    # Step 4: 前向网络模拟
    np.random.seed(42)
    W = np.random.randn(4, 6)  # shape (4, 6)
    b = np.random.randn(4)     # shape (4,)
    output = fused_input @ W.T + b  # (max_points, 4)

    return output

```





```
import torch
import torch.nn.functional as F

def preprocess_and_forward(image, point_cloud, max_points):
    # Step 1: 图像处理
    image = image.float()  # 转换为 float32 类型
    mean = image.mean(dim=(0, 1), keepdim=True)  # (1, 1, 3)
    std = image.std(dim=(0, 1), keepdim=True) + 1e-6
    image_norm = (image - mean) / std  # 归一化

    # 全局平均池化，得到图像特征 (3,)
    image_feature = image_norm.mean(dim=(0, 1))  # shape: (3,)

    # Step 2: 点云处理
    N = point_cloud.shape[0]
    if N >= max_points:
        indices = torch.randperm(N)[:max_points]
        point_cloud_processed = point_cloud[indices]
    else:
        padding = torch.zeros((max_points - N, 3), dtype=point_cloud.dtype, device=point_cloud.device)
        point_cloud_processed = torch.cat([point_cloud, padding], dim=0)

    # Step 3: 多模态融合
    image_feature_expand = image_feature.unsqueeze(0).repeat(max_points, 1)  # (max_points, 3)
    fused_input = torch.cat([point_cloud_processed, image_feature_expand], dim=1)  # (max_points, 6)

    # Step 4: 前向网络模拟
    torch.manual_seed(42)
    W = torch.randn(4, 6)  # 权重 (out_features, in_features)
    b = torch.randn(4)     # 偏置
    output = F.linear(fused_input, W, b)  # shape: (max_points, 4)

    return output

```

## shape：

| 示例代码                  | 数据    | shape   | 含义说明      |
| ------------------------- | ------- | ------- | ------------- |
| `np.array(42)`            | 42      | `()`    | 0维标量       |
| `np.array([1, 2, 3])`     | [1,2,3] | `(3,)`  | 一维，3个元素 |
| `np.array([[1,2,3]])`     | 1×3矩阵 | `(1,3)` | 二维，1行3列  |
| `np.array([[1],[2],[3]])` | 3×1矩阵 | `(3,1)` | 二维，3行1列  |

shape = ()       → 标量         → 一个点
shape = (3,)     → 向量         → 一条线段（长度3）
shape = (2, 3)   → 矩阵         → 2行3列的表格
shape = (2, 3, 3)→ 图像块等高维张量 → 比如2张3×3图像



# np.tile()

### **`np.tile` 支持多维复制，因此参数必须是元组 `reps`，指定每个维度的重复次数。**

#### 

### 情况一：一维数组

```
a = np.array([1, 2, 3])
np.tile(a, 2)
```

输出：

```
array([1, 2, 3, 1, 2, 3])
```

这等价于：

```
np.tile(a, (2,))  # 沿着第一个维度重复2次
```

------

### 情况二：二维数组，重复成矩阵

```
a = np.array([[1, 2, 3]])
np.tile(a, (3, 1))  # 沿第0维重复3次，沿第1维重复1次
```

输出：

```
[[1 2 3]
 [1 2 3]
 [1 2 3]]
```

- `(3, 1)` 的意思是：
  - 在第 0 维复制 3 次（行重复）
  - 在第 1 维复制 1 次（列不变）

# np.concatenate

np.concatenate([array1, array2], axis=n)
array1 和 array2 必须在除了 axis 指定的维度外，其他维度形状一致

axis=0 表示在“行方向”拼接

axis=1 表示在“列方向”拼接

---
```
x.view(batch_size, -1)      # 重塑
x.permute(0, 2, 3, 1)       # 维度调换
x.squeeze()                 # 去除维度
x.unsqueeze(dim=1)         # 增加维度
x.transpose(1, 2)          # 交换维度
```

view 张量**重塑（reshape）**操作

```
x.view(-1)	拉平成一维张量
x.view(batch, -1)	批处理 reshape（常用于送入全连接层）
x.view(1, -1, 28, 28)	
```

上三角-inf
torch.triu(torch.ones(L, L) * float('-inf'), diagonal=1)

---

# bosch

**模仿学习的定义**

智能体（Agent）通过观察和模仿专家（Expert）的示范来学习任务策略，而不是通过自己大量试错（强化学习）。

**fast bev**

# 鱼眼模型

核心模型：Kannala–Brandt（OpenCV fisheye 模块）

---

# mlp 

```
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super().__init__()  

        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)
        self.relu1 = nn.ReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out
```

**self.relu1 = nn.ReLU()**

---

# IOU
```
import numpy as np

def iou_numpy(box1, box2):
    """
    box1: [x1, y1, x2, y2]
    box2: [x1, y1, x2, y2]
    """
    # 交集坐标
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # 交集面积
    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h

    # 各自面积
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # 并集
    union = area1 + area2 - inter_area

    # IoU
    iou = inter_area / union if union > 0 else 0.0
    return iou

# 测试
b1 = [0, 0, 2, 2]
b2 = [1, 1, 3, 3]
print("NumPy IoU:", iou_numpy(b1, b2))  # 1/7 ≈ 0.142
```
Torch
```
def iou_loss(pred, target, eps=1e-6):
    # pred, target: [N, 4], (x1,y1,x2,y2)
    inter_x1 = torch.max(pred[:,0], target[:,0])
    inter_y1 = torch.max(pred[:,1], target[:,1])
    inter_x2 = torch.min(pred[:,2], target[:,2])
    inter_y2 = torch.min(pred[:,3], target[:,3])

    inter_area = (inter_x2 - inter_x1).clamp(min=0) * \
                 (inter_y2 - inter_y1).clamp(min=0)
    area_pred = (pred[:,2]-pred[:,0]) * (pred[:,3]-pred[:,1])
    area_target = (target[:,2]-target[:,0]) * (target[:,3]-target[:,1])

    union = area_pred + area_target - inter_area
    iou = inter_area / union.clamp(min=eps)

    return 1 - iou  # IoU 越大，loss 越小

```
# GIOU

**问题：**IoU=0 时无法优化（框不相交）

**改进：**引入最小闭包矩形 𝐶，再加一个惩罚项：

```
import torch

def giou_loss(pred, target, eps=1e-7):
    """
    pred:   Tensor [N, 4] (x1, y1, x2, y2)
    target: Tensor [N, 4] (x1, y1, x2, y2)
    return: Tensor [N] GIoU loss for each pair
    """
    # 交集
    inter_x1 = torch.max(pred[:, 0], target[:, 0])
    inter_y1 = torch.max(pred[:, 1], target[:, 1])
    inter_x2 = torch.min(pred[:, 2], target[:, 2])
    inter_y2 = torch.min(pred[:, 3], target[:, 3])

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    # 各自面积
    area_pred = (pred[:, 2] - pred[:, 0]).clamp(min=0) * \
                (pred[:, 3] - pred[:, 1]).clamp(min=0)
    area_target = (target[:, 2] - target[:, 0]).clamp(min=0) * \
                  (target[:, 3] - target[:, 1]).clamp(min=0)

    # 并集
    union = area_pred + area_target - inter_area + eps
    iou = inter_area / union

    # 最小闭包矩形
    enc_x1 = torch.min(pred[:, 0], target[:, 0])
    enc_y1 = torch.min(pred[:, 1], target[:, 1])
    enc_x2 = torch.max(pred[:, 2], target[:, 2])
    enc_y2 = torch.max(pred[:, 3], target[:, 3])

    enc_w = (enc_x2 - enc_x1).clamp(min=0)
    enc_h = (enc_y2 - enc_y1).clamp(min=0)
    enc_area = enc_w * enc_h + eps

    # GIoU
    giou = iou - (enc_area - union) / enc_area
    loss = 1 - giou
    return loss

```
