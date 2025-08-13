# coding
**NMS**

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
