# coding

transpose(1,2) 是一种更简洁的写法，只能交换两个维度。

permute(0,2,1,3) 可以同时调整多个维度，写法更通用

x.size() == x.shape

masked_fill(mask, value) 会把 mask=True 的位置替换成指定值（这里是 -inf）

attn  = F.softmax(scores,dim = -1)  # 数dim对应的值的行/列，所有元素softmax


```
out = out.transpose(1, 2).contiguous().view(B, Tq, self.d_model)

在 PyTorch 里，张量的数据实际上是存放在一块连续的内存 buffer 里的。有些操作（比如 transpose, permute）不会真正移动数据，而是只改变「视图」(view)，通过 stride（步长）来改变索引方式。

如果你直接对这种 非连续张量 调用 .view()，PyTorch 会报错，因为 .view() 要求底层数据是连续的。

```

```
scores =
[[1.0, 2.0, 3.0],   # query1 对 key1~3 的分数
 [4.0, 5.0, 6.0]]   # query2 对 key1~3 的分数

```
scores[2,3],dim = -1 对应 3 , 那就将每行三个softmax


**Multihead attention**

```
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model:int, num_heads:int, dropout:float=0.0 ,bias:bool = False ):
        super().__init__()
        assert(d_model % num_heads ==0)
        self.d_model = d_model
        self.num_heads = num_heads

        self.w_q = nn.Linear(d_model,d_model,bias = bias)
        self.w_k = nn.Linear(d_model,d_model,bias = bias)
        self.w_v = nn.Linear(d_model,d_model,bias = bias)

        self.w_0 = nn.Linear(d_model,d_model,bias = bias)

        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, query:torch.Tensor, key: torch.Tensor, value :torch.Tensor, attn_mask:torch.Tensor, casual:bool = False ) -> Tuple[torch.Tensor,torch.Tensor]
        if key is None: key = query
        if value is None : value = query

        B, Tq ,_ = query.shape
        Tk = key.shape[1]

        Q = self.w_q(query).view(B,-1,self.num_heads,self.d_head).transpose(1,2) #把后两维变为长度*d_head
        K = self.w_k(key).view(B,-1,self.num_heads,self.d_head).transpose(1,2)
        V = self.w_v(value).view(B,-1,self.num_heads,self.d_head).transpose(1,2)

        scores = torch.matmul(Q,K.transpose(-1,-2))/math.sqrt(self.d_head)  #[B, num_heads, Tq, Tk]

        if casual:
            casual_mask = torch.triu(
                        torch.ones(Tq,Tk)，diagonal =1
                        )
            scores = scores.masked_fill(casual_mask , float("-inf"))

        attn  = F.softmax(scores,dim = -1)  # 将所有key做 softmax
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, V)  # Tq, Tk* Tk, d_head = B, num_heads, Tq,  d_head

        out = out.transpose(1, 2).contiguous().view(B, Tq, self.d_model)  # B, Tq, num_heads, d_head

        out = self.wo(out)
        out = self.proj_dropout(out)

        return (out,attn) 
        

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
