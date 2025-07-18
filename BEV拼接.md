# BEV拼接

**生成重叠区域掩码 `overlapMask`**

```
overlapMask = get_overlap_region_mask(imA, imB)
overlapMaskInv = cv2.bitwise_not(overlapMask)
indices = np.where(overlapMask == 255)
```

**`overlapMask`**：一个二值图，255 表示 `imA` 与 `imB` 的重叠像素区域；0 表示非重叠区。

**`overlapMaskInv`**：它的按位取反，用来在后面只保留非重叠区像素，以便提取边界。

**`indices`**：记录所有重叠像素的坐标 `(y,x)`，后续只对这些像素做细调。

**提取边界用的差异图像**

```
pythonCopyEditimA_diff = cv2.bitwise_and(imA, imA, mask=overlapMaskInv)
imB_diff = cv2.bitwise_and(imB, imB, mask=overlapMaskInv)
```

- 这里用 `overlapMaskInv` 掩码，将 `imA`（或 `imB`）中**非**重叠部分保留下来。
- 目的：从各自的非重叠区域提取最外侧轮廓，多边形近似后用于距离度量。

**对基本纯黑的diff二值化掩码**

```
mask = get_mask(img)
```

将输入 `img`（通常是彩色图或带 α 通道的图）转换成gray

```
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
```

非零像素变为 255，零像素保持 0。

**形态学膨胀**

```
mask = cv2.dilate(mask, np.ones((2, 2), np.uint8), iterations=2)
```

用一个 2×2 的全 1 核做两次膨胀，一方面填补细小的孔洞，另一方面让掩码的边界更平滑、连续。

**提取外部轮廓**

```
cnts, hierarchy = cv2.findContours(
    mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)[-2:]
```

- `cv2.RETR_EXTERNAL`：只检索最外层轮廓，不关心内层嵌套的轮廓；
- `cv2.CHAIN_APPROX_SIMPLE`：对轮廓进行点的简化，去掉冗余的中间点。

**选取最大轮廓**

```
C = sorted(cnts, key=lambda x: cv2.contourArea(x), reverse=True)[0]
```

对所有检测到的轮廓按面积降序排序，取面积最大的那个，认为它就是覆盖重叠区域的主轮廓。

**多边形逼近**

```
polygon = cv2.approxPolyDP(
    C,
    epsilon=0.009 * cv2.arcLength(C, True),
    closed=True
)
```

- `cv2.arcLength(C, True)` 计算轮廓周长；
- 以周长的 0.9% (`0.009 * 周长`) 作为逼近精度 `epsilon`，使用 Douglas–Peucker 算法将轮廓折线化成顶点更少的多边形。

**遍历所有重叠像素坐标**

`indices` 存放了 `overlapMask==255` 的所有 (y,x) 坐标。

**计算到区域 A,B 外边界的距离**

`pointPolygonTest` 返回一个带符号的最短距离（内部为正，外部为负），第三个参数 `True` 表示要计算精确距离。

```
G[y, x] = distToB / (distToA + distToB)
```

## make_luminance_balance

```
front, back, left, right = self.frames
m1, m2, m3, m4   = self.masks    # 四个重叠区域掩码
Fb, Fg, Fr       = cv2.split(front)
Bb, Bg, Br       = cv2.split(back)
Lb, Lg, Lr       = cv2.split(left)
Rb, Rg, Rr       = cv2.split(right)
```

- 将每幅图拆成 B/G/R 三个灰度通道，准备单独做亮度（灰度）调整。
- 对应的 `m1`…`m4` 分别是前–左、前–右、后–左、后–右的重叠区掩码。

**计算每对重叠区域的平均亮度比率**

重叠区的原始灰度值，`np.sum` 则把所有这些像素加起来，得到两张图在这块区域里的**总亮度**

```totalA = np.sum(grayA * mask)
totalB = np.sum(grayB * mask)
```

这里 `grayA * mask` 会把非重叠区（mask=0）的像素值变成 0，只保留重叠区的原始灰度值，`np.sum` 则把所有这些像素加起来，得到两张图在这块区域里的**总亮度**。

```
ratio = totalA / totalB
```

得到 4×3 共 12 个通道比率：

- **a1–a3**：右→前 接缝处的 B/G/R 比率
- **b1–b3**：后→右 接缝处的 B/G/R 比率
- **c1–c3**：左→后 接缝处的 B/G/R 比率
- **d1–d3**：前→左 接缝处的 B/G/R 比率

**计算全局几何平均增益** `t1–t3`

```
t1 = (a1 * b1 * c1 * d1)**0.25   # B 通道
t2 = (a2 * b2 * c2 * d2)**0.25   # G 通道
t3 = (a3 * b3 * c3 * d3)**0.25   # R 通道
```

**计算“前视图”蓝通道实际增益 x1 并微调**

```
x1 = t1 / (d1 / a1) ** 0.5
x1 = tune(x1)
```

# make_white_balance

## 1. 拆通道，计算每个通道的平均值

```
B, G, R = cv2.split(image)
m1 = np.mean(B)
m2 = np.mean(G)
m3 = np.mean(R)
```

- `m1,m2,m3` 分别是整张图的 **蓝/绿/红** 通道的平均灰度值。
- 如果图像偏色，三个值就不会相等，比如偏红时 `m3` 会更大。

------

## 2. 求全局目标亮度 `K`

```
K = (m1 + m2 + m3) / 3
```

- `K` 就是我们“希望”三条通道都能达到的平均亮度——它取三者平均，保证前后亮度总量不变。

------

## 3. 计算每个通道的增益系数

```
c1 = K / m1   # 蓝通道增益
c2 = K / m2   # 绿通道增益
c3 = K / m3   # 红通道增益
```

- 这一步的含义是：
  - 如果某个通道平均值太低（比如 `m1 < K`），那么 `c1 = K/m1 > 1`，就会把该通道整体调亮；
  - 反之，如果某通道太高，`c > 1`，就会适当压暗。

------

## 4. 应用增益，调整亮度

```
B = adjust_luminance(B, c1)
G = adjust_luminance(G, c2)
R = adjust_luminance(R, c3)
```



# 难点

OpenCV CUDA 模块里**并没有**像 CPU 端 `cv::rotate` 那样的专用 `rotate` 函数；要在 GPU 上做旋转，一般就是走仿射变换——也就是用 `cv::cuda::warpAffine` 配合一个旋转矩阵来实现。

```
cv::Point2f center( cx, cy );       // 旋转中心
double angle = θ;                   // 旋转角度（度为单位，逆时针为正）
double scale = 1.0;                 // 缩放因子
cv::Mat M_left = cv::getRotationMatrix2D(center, angle, scale);
```

```
cv::cuda::warpAffine(
    d_src,         // 输入
    d_dst,         // 输出
    M_left,             // 2×3 仿射矩阵（包含旋转部分）
    d_src.size(),  // 输出大小
    cv::INTER_LINEAR,
    cv::BORDER_CONSTANT  // 边界补全方式
);
```

**遍历目标图像每个像素位置** (x′,y′)（这里目标大小就是 `d_src.size()`）。

**逆向映射**
 用给定的仿射矩阵 M 的逆（内部自动算好）把 (x′,y′) 映射回源图像坐标 (x,y)：

**双线性插值**

- 找到 (x,y)四个最近整数点 (x0,y0),(x1,y0),(x0,y1),(x1,y1)。

- 按距离权重对这四个像素的 B/G/R 值做加权平均，得到最终 (x′,y′)处的颜色。

  

# FAQ

**1. 相机模型与标定**

> **问**：你的 `FisheyeCameraModel` 是如何从 YAML 文件中提取内外参的？为什么用鱼眼模型而不是针孔模型？

**答**：

- 在 YAML 里，我们保存了相机内参（焦距 `fx,fy`、主点 `cx,cy`）和鱼眼畸变系数 `k1…k4`。初始化时用 OpenCV 的 `cv::FileStorage` 读取这些字段，填到 `cv::Mat` 或自定义结构里。
- 选择鱼眼模型是因为摄像头为了更大视野、通常用超广角镜头，畸变很强；针孔模型只能支持轻微畸变，用鱼眼模型才能准确将图像映射到平面，减少拼接时的几何误差。

------

**2. 单目预处理流水线**

> **问**：`undistort`、`project`、`flip` 三步做了什么？怎么用 OpenCV 实现？

**答**：

- **undistort**：用 `cv::fisheye::undistortImage` 或 `fisheye::initUndistortRectifyMap` + `cv::remap`，把畸变图转成未失真图。
- **project**：先根据摄像头外参和俯仰角，构造一个从相机平面到地面平面的投影 homography，然后用 `cv::warpPerspective`（或 CUDA 端 `warpPerspective`）映射到鸟瞰参考平面。
- **flip**：根据每路安装位置翻转（`cv::flip(src,dst,0/1)`），保证所有图像方向一致，方便后续拼接。

------

**3. 重叠区域权重与掩码**

> **问**：为什么先对非重叠区做 `bitwise_and` 提取轮廓？

**答**：

- `bitwise_and(imA,imB)` 能抠出两条带真正同时存在的区域，但由于背景全 0，这一步相当于“取交集”。
- 再对非重叠区域提取轮廓，多边形拟合后能得到每幅图的最外边界，用于后面做距离度量——把越靠近 B 边界的 A 区像素在融合时往 B 偏，保证过渡平滑。

------

**4. 多边形逼近与距离测试**

> **问**：`epsilon` 怎么选？`pointPolygonTest` 的正负值是什么意思？

**答**：

- `epsilon = 0.009×arcLength` 是经验值，保证对大轮廓做 0.9% 误差的简化。太大会丢细节、影响权重精度；太小又会保留冗余顶点、耗时多。
- `pointPolygonTest(poly,p,True)` 返回点到多边形边的最短带符号距离：>0 在内，<0 在外。我们只关心落在边界一定范围(`dist < threshold`)的点，对它们做权重重计算。

------

**5. 亮度平衡（Luminance Balance）**

> **问**：为什么先算四条缝的几何平均 `t = (a·b·c·d)^¼`？

**答**：

- 四条缝分别给出不同视角、不同光照下的相对亮度比。几何平均能同时考虑所有缝，比算普通均值更稳健，避免某一路异常拉偏。
- 再把这个 `t` 按照每条缝的比例分配回各个视图，能兼顾全局一致性和局部过渡。

------

**6. 白平衡（White Balance）**

> **问**：灰度世界假设适用场景和局限？

**答**：

- 灰度世界假设：场景中平均反射率近似中性灰，三通道平均相同。适合光照较均匀、色彩分布丰富的户外场景。
- 局限在单一色调场景（大面积绿色、雪地）或光源色偏强烈时会失效，这时可采用贝叶斯估计、灰度边缘法或学习式白平衡算法。

------

**7. 拼接与融合**

> **问**：为什么不用多频带融合（Multi‑band blending）？

**答**：

- 多频带融合（Laplace 金字塔混合）效果更自然，但计算量大、难实时。我们的权重融合在接缝区宽度有限、现场算力受限下能在保证平滑的同时达到 30+ FPS 要求。

------

**8. CUDA 实现与性能**

> **问**：如何减少 CPU↔GPU 拷贝？

**答**：

- 全流程都用 `cv::cuda::GpuMat`，初始化后只在读取相机和最终展示／保存时拷贝。
- 用 CUDA 流（`cuda::Stream`）打包 `warpAffine`、`merge`、`copy` 等操作，异步执行减少同步等待。

------

**9. 多线程与同步**

> **问**：`ProjectedImageBuffer` 如何防止死锁或丢帧？

**答**：

- 用 QMutex + QWaitCondition 实现 barrier，同步到所有摄像头一帧到齐或超时。
- `drop_if_full` 为 true 时，如果后端处理慢，缓冲区满则丢弃最旧帧，避免阻塞实时帧到来。

------

**10. 异常与鲁棒性**

> **问**：标定文件缺失或尺寸不一致该怎么处理？

**答**：

- 在初始化时检查 YAML 文件可读、图像分辨率匹配标定内参，出错立即报日志并回退到默认参数或退出。
- 如果运行中某路全黑或投影全 0，可跳过该路，仅剩三路进行拼接，保证不完全失效。

------

**11. 调试与可视化**

> **问**：如何可视化 `G0–G3`、`M0–M3`？

**答**：

- 把 `G_i` 乘 255 转为灰度图保存为 PNG，直接 `imshow`；
- `M_i` 已是二值掩码，可叠加到原图上半透明显示，观察融合边界宽度与位置。

------

**12. 扩展与未来改进**

> **问**：如何支持 6 路或 360° 全景？

**答**：

- 抽象拼接流程，把“前”、“后”、“左”、“右”推广成 N 边多边形环结构，对相邻两路统一调用同样的 `get_weight_mask_matrix`、`stitch` 接口。
- 在权重计算中引入学习式 UAV 拼接网络，根据实际环境自适应调整权重分布，提升雨雪、低光场景下的拼接质量。