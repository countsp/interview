# coding

transpose(1,2) 是一种更简洁的写法，只能交换两个维度。

permute(0,2,1,3) 可以同时调整多个维度，写法更通用

x.size() == x.shape

masked_fill(mask, value) 会把 mask=True 的位置替换成指定值（这里是 -inf）

attn  = F.softmax(scores,dim = -1)  # 数dim对应的值的行/列，所有元素softmax


causal_mask 需要放到设备上

```
causal_mask = torch.triu(
    torch.ones(Tq, Tk, device=scores.device, dtype=torch.bool), diagonal=1
)
```

在 PyTorch 里，张量的数据实际上是存放在一块连续的内存 buffer 里的。有些操作（比如 transpose, permute）不会真正移动数据，而是只改变「视图」(view)，通过 stride（步长）来改变索引方式。

如果你直接对这种 非连续张量 调用 .view()，PyTorch 会报错，因为 .view() 要求底层数据是连续的。

```
out = out.transpose(1, 2).contiguous().view(B, Tq, self.d_model)

```

softmax
```
scores =
[[1.0, 2.0, 3.0],   # query1 对 key1~3 的分数
 [4.0, 5.0, 6.0]]   # query2 对 key1~3 的分数

```
scores[2,3],dim = -1 对应 3 , 那就将每行三个softmax


**Multihead attention**

```
import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    """
    多头（自/交叉）注意力
    输入:
      query: [B, Tq, d_model]
      key  : [B, Tk, d_model]  (若为自注意力，通常与query相同)
      value: [B, Tk, d_model]  (若为自注意力，通常与key相同)
      attn_mask: [B, 1, Tq, Tk] 或 [1, 1, Tq, Tk]，值为0/1；0位置会被mask掉
                 （也可传bool，True=保留，False=mask）
      causal: 是否开启上三角因果mask（用于decoder）
    超参:
      d_model: 模型维度
      num_heads: 头数 (d_model % num_heads == 0)
      dropout: 注意力分数上的dropout
    返回:
      out: [B, Tq, d_model]
      attn: [B, num_heads, Tq, Tk] (注意力权重，若需要可用于可视化/对齐)
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0, bias: bool = True):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        # 线性映射得到 Q,K,V
        self.w_q = nn.Linear(d_model, d_model, bias=bias)
        self.w_k = nn.Linear(d_model, d_model, bias=bias)
        self.w_v = nn.Linear(d_model, d_model, bias=bias)

        # 输出映射
        self.w_o = nn.Linear(d_model, d_model, bias=bias)

        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

    def _shape(self, x: torch.Tensor, B: int) -> torch.Tensor:
        # [B, T, d_model] -> [B, num_heads, T, d_head]
        return x.view(B, -1, self.num_heads, self.d_head).transpose(1, 2)

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        causal: bool = False,
        need_weights: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        支持自注意力：forward(x, x, x, ...)
        支持交叉注意力：forward(q, k, v, ...)
        attn_mask:
            - 若为布尔或0/1张量，形状可为 [B, 1, Tq, Tk] 或 [1, 1, Tq, Tk]
            - True/1 表示可见，False/0 表示mask
            - 也可传 float，mask处为 -inf（将被加到score上）
        causal:
            - 若 True，自动叠加上三角因果mask (仅允许看历史)
        """
        if key is None:   key = query
        if value is None: value = key

        B, Tq, _ = query.shape
        Tk = key.size(1)

        # 线性映射
        Q = self._shape(self.w_q(query), B)  # [B, h, Tq, d_head]
        K = self._shape(self.w_k(key),   B)  # [B, h, Tk, d_head]
        V = self._shape(self.w_v(value), B)  # [B, h, Tk, d_head]

        # 缩放点积注意力分数
        # scores: [B, h, Tq, Tk]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)

        # 叠加因果mask（上三角无效）
        if causal:
            # 上三角为True的位置应被mask；构造一个下三角保留的mask
            causal_mask = torch.triu(
                torch.ones(Tq, Tk, device=scores.device, dtype=torch.bool), diagonal=1
            )  # [Tq, Tk]，上三角True
            scores = scores.masked_fill(causal_mask, float("-inf"))

        # 叠加外部attn_mask
        if attn_mask is not None:
            # 支持 bool/byte/float
            if attn_mask.dtype == torch.bool:
                # True=可见 -> 我们需要把不可见(False)处置为 -inf
                scores = scores.masked_fill(~attn_mask, float("-inf"))
            elif attn_mask.dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
                scores = scores.masked_fill(attn_mask == 0, float("-inf"))
            else:
                # 视作 additive mask（同Transformer通用做法），直接加上去
                scores = scores + attn_mask

        # softmax -> 注意力
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        # 加权求和
        out = torch.matmul(attn, V)  # [B, h, Tq, d_head]

        # 合并头
        out = out.transpose(1, 2).contiguous().view(B, Tq, self.d_model)  # [B, Tq, d_model]

        # 输出映射
        out = self.w_o(out)
        out = self.proj_dropout(out)

        return (out, attn) if need_weights else (out, None)

# ------------- 简单自测 -------------
if __name__ == "__main__":
    torch.manual_seed(0)
    B, Tq, Tk, d_model, h = 2, 5, 5, 64, 8
    x = torch.randn(B, Tq, d_model)

    mha = MultiHeadAttention(d_model=d_model, num_heads=h, dropout=0.1)
    # 自注意力
    y, attn = mha(x, x, x, causal=True, need_weights=True)
    print("out:", y.shape)        # [2, 5, 64]
    print("attn:", attn.shape)    # [2, 8, 5, 5]

```
---

二维平面上多个矩形求并集面积

**IOU/NMS**

坐标格式要提前约定（xyxy/cxcywh），若是后者先转成 xyxy。

NMS 只在同类别内做；多尺度/多头输出要先 concat 再 per-class NMS。

工程优化：按分数阈值过滤、Top-K 预筛、GPU 向量化/矩阵化（如 PyTorch/TensorRT 内置 NMS）。

Soft-NMS 常带来更高的 AP，尤其是拥挤场景。

```
import numpy as np

def iou_xyxy(box, boxes):
    """box: (4,), boxes: (N,4)  -> IoU: (N,)"""
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter_w = np.maximum(0.0, x2 - x1)
    inter_h = np.maximum(0.0, y2 - y1)
    inter = inter_w * inter_h

    area1 = (box[2] - box[0]) * (box[3] - box[1])
    area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area1 + area2 - inter + 1e-9
    return inter / union

def nms(boxes, scores, iou_thr=0.5):
    """
    boxes:  (N,4)  format [x1,y1,x2,y2]
    scores: (N,)
    return: 保留的索引 list（按分数从高到低）
    """
    if len(boxes) == 0:
        return []

    boxes = boxes.astype(np.float32)
    scores = scores.astype(np.float32)
    order = scores.argsort()[::-1]  # 按分数降序
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break

        # 与当前最大分数框 i 的 IoU
        ious = iou_xyxy(boxes[i], boxes[order[1:]])
        # 保留 IoU 小于阈值的
        inds = np.where(ious < iou_thr)[0]
        # +1 是因为跳过了 order[0]
        order = order[inds + 1]

    return keep

def nms_per_class(boxes, scores, labels, iou_thr=0.5):
    keep_all = []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        k = nms(boxes[idx], scores[idx], iou_thr)
        keep_all.extend(idx[k])
    # 可按分数再整体排序
    return sorted(keep_all, key=lambda i: scores[i], reverse=True)
```


**Softmax**
```
import math

def softmax(x):
    """
    x: 一维列表或数组
    """
    # 防止溢出：减去最大值
    max_val = max(x)
    exps = [math.exp(i - max_val) for i in x]
    sum_exps = sum(exps)
    return [e / sum_exps for e in exps]

# 测试
scores = [2.0, 1.0, 0.1]
print("Softmax:", softmax(scores))
```

这样所有 $x_i - m \le 0$，指数不会爆炸，同时结果完全不变（因为分子分母同时乘上了 $e^{-m}$ 被约掉）。


# Focal loss
```
import torch
import torch.nn.functional as F

def focal_loss(pred, gt, alpha=2, beta=4):
    """
    pred: [B, C, H, W] —— 模型预测，sigmoid 后
    gt:   [B, C, H, W] —— 高斯真值
    """
    pos_mask = (gt == 1).float()
    neg_mask = (gt < 1).float()

    pos_loss = - (1 - pred) ** alpha * torch.log(pred + 1e-6) * pos_mask
    neg_loss = - (pred ** alpha) * ((1 - gt) ** beta) * torch.log(1 - pred + 1e-6) * neg_mask

    num_pos = pos_mask.sum()
    loss = (pos_loss.sum() + neg_loss.sum()) / torch.clamp(num_pos, min=1.0)
    return loss
```
在检测/分割里，负样本通常很多，如果不归一化，loss 会被负样本主导。

== 才是比较，不能写成 =

变量名不能用 .

log 要加 1e-6 防止数值溢出

分母加 1e-6 防止除零


# BCE
```
def bce(gt,pred):
    loss = - ( gt * torch.log(pred) +(1-gt) * torch.log(1-pred))
    return loss.mean()
```
# CE

```
def categorical_CE(gt, pred):
    """
    gt:   [N] 真实类别索引 (int)
    pred: [N, C] 模型输出的 logits
    """
    eps = 1e-6
    probs = torch.softmax(pred, dim=1)                  # [N, C]
    target_probs = probs[torch.arange(len(gt)), gt]     # 取真实类别的概率
    loss = - torch.log(target_probs + eps)
    return loss.mean()
```
