# BoxDreamer 网络结构与前向过程解读

## 1. 算法核心思想

BoxDreamer 的目标是：给定若干张目标物体的参考视图，以及一张查询图，预测查询图中的 3D bounding box 八个角点，进而可通过 PnP 得到物体姿态。

它不是传统的单图检测器，也不是直接输入 CAD 模型做姿态估计。它的核心思路是：

```text
参考图像中已经知道目标 3D box 的 2D 投影
查询图像中不知道目标 3D box
模型学习从参考视图的 box-corner 表达和 RGB 外观中，推断查询视图的 box-corner heatmap
```

默认配置中，BoxDreamer 使用：

```text
pose_representation = bb8
bbox_representation = heatmap
encoder = DINOv2
decoder = BETR Transformer
```

其中 `bb8` 表示 3D bounding box 的 8 个角点，`heatmap` 表示每个角点对应一个 heatmap 通道。

## 2. 整体输入输出

### 2.1 输入 batch

数据集最终会组织出一个序列：

```text
T = reference_num + 1
前 T-1 张：reference views
最后 1 张：query view
```

主要字段来自 `src/datasets/base.py`：

```text
images:              [B, T, 3, H, W]
bbox_feat:           [B, T, 8, H, W]       # bb8 heatmap 监督/条件
bbox_proj_crop:      [B, T, 8, 2]          # 归一化 2D 角点坐标
bbox_3d:             [B, T, 8, 3]          # 3D box 八角点
poses:               [B, T, 4, 4]
non_ndc_intrinsics:  [B, T, 3, 3]
crop_parameters:     [B, T, ...]
image_masks:         [B, T, 1, H, W]
query_idx:           查询图在序列中的位置，通常是最后一张
```

默认 `image_size = 224`，`patch_size = 14`，所以 DINO / Transformer patch 网格是：

```text
224 / 14 = 16
patch 数 = 16 x 16 = 256
```

### 2.2 输出

训练时主要输出：

```text
pred_bbox: [B, T, 8, H, W]
```

但只有 query view 位置参与 loss。

测试时还会输出：

```text
regression_boxes: [B, T, 8, 2]   # 从 heatmap 解码出的 2D 八角点
pred_poses:       [B, T, 4, 4]   # 由 2D-3D corner PnP 得到的姿态
```

## 3. 数据侧如何生成 8 角点 heatmap

数据处理主要在 `src/datasets/base.py` 和 `src/datasets/utils/base/bbox_utils.py`。

流程如下：

```text
读取 RGB / bbox / pose / intrinsic / model
-> 从模型点云或 3D box 文件得到 bbox_3d
-> 根据 pose 和 K 将 bbox_3d 投影到图像
-> 按 bbox crop 并 resize 到 224 x 224
-> 更新相机内参到 crop 坐标
-> 再次投影 3D corners 到 crop 图
-> 生成 8 通道 bbox_feat heatmap
```

关键函数：

```text
prepare_bbox3d()
make_proj_bbox()
make_bbox_features()
```

`make_bbox_features(type="heatmap")` 会为 8 个角点各生成一个 heatmap，因此：

```text
bbox_feat: [T, 8, 224, 224]
```

这些 heatmap 对参考图是模型条件，对查询图是训练监督。

## 4. BoxDreamer.forward() 完整过程

主入口是：

```text
src/models/BoxDreamerModel.py
BoxDreamer.forward(data)
```

整体前向可以概括为：

```text
1. 取出图像、位姿、内参、bbox heatmap
2. 构造 query mask
3. 准备 pose / bbox 表达
4. 用 DINOv2 提取 RGB patch features
5. 将参考图 bbox heatmap + RGB feature 输入 BETR
6. 对 query 图使用 learnable query 替代未知 bbox
7. Transformer 跨视图、跨 patch 传播信息
8. 输出 query 的 8 通道 heatmap
9. 训练时计算 heatmap loss
10. 测试时解码角点并 PnP 得到 pose
```

### 4.1 提取输入

`_extract_input_data()` 做的事：

```text
poses = data["poses"]
frames = data["images"]
K = data["non_ndc_intrinsics"]
crop_params = data["crop_parameters"]
image_masks = data["image_masks"]
query_idx = data["query_idx"]
```

然后生成：

```text
camera_mask: [B, T]
```

其中 query view 为 `True`，reference views 为 `False`。

### 4.2 准备姿态/box 表达

默认配置 `rotation_type = null` 且 `pose_representation = bb8`，因此不会走相机 ray 或向量姿态分支，而是直接使用：

```python
pose_feat = data["bbox_feat"].clone()
```

也就是说，网络条件输入中的“姿态表达”本质上就是 8 个角点的 heatmap。

对于 reference views：

```text
pose_feat = 已知 bbox heatmap
```

对于 query view：

```text
pose_feat 原本有 GT，但进入 BETR 时会被 mask 掉，用 learnable query 替代
```

### 4.3 提取 RGB 特征

默认使用 DINOv2：

```text
src/models/modules/encoder/dinov2.py
DinoV2Wrapper.predict()
```

输入：

```text
frames: [B, T, 3, 224, 224]
```

会先展平为：

```text
[B*T, 3, 224, 224]
```

再经过 DINOv2：

```text
x_norm_patchtokens: [B*T, 256, 768]
```

最后恢复为：

```text
rgb_feature: [B, T, 256, 768]
```

DINOv2 默认冻结，只提供强 RGB patch 表征。

## 5. BETR Decoder 结构

BETR 位于：

```text
src/models/modules/backbone/betr.py
```

BETR 可以理解为：

```text
Box Estimation Transformer
```

它不是 encoder-decoder 形式，而是把所有视图、所有 patch token 拼成一个长序列，用多层 self-attention 让 reference 和 query 之间交换信息。

### 5.1 输入到 BETR 的内容

调用形式：

```python
query_ret = self.decoder(
    pose_feat,
    frames,
    camera_mask,
    rgb_feature,
    normalize(image_masks)
)
```

对默认 DINOv2 模式，BETR 主要用：

```text
pretrain_rgb_feat = rgb_feature
pose_feat = bbox heatmap
masks = camera_mask
```

### 5.2 bbox heatmap patchify

`pose_feat` 原始形状：

```text
[B, T, 8, 224, 224]
```

经过 patchify：

```text
每个 14 x 14 patch 展平
8 通道 x 14 x 14 = 1568 维
```

得到：

```text
pose_feat_flat: [B, T, 256, 1568]
```

再经过 `bbox_emb`：

```text
pose_feat_flat: [B, T, 256, 768]
```

### 5.3 RGB patch token

DINOv2 已经输出：

```text
rgb_flat: [B, T, 256, 768]
```

经过轻量 MLP 和 LayerNorm 后仍为：

```text
rgb_flat: [B, T, 256, 768]
```

### 5.4 mask 查询图的 bbox 条件

关键操作：

```python
pose_feat_flat_init[masks] = self.bbox_learnable_query
```

含义是：

```text
reference views:
  使用真实 bbox heatmap embedding

query view:
  不允许看到真实 bbox heatmap
  用一个可学习 query token 替代
```

这一步是 BoxDreamer 的训练核心：模型不能直接抄 query GT，只能从 query RGB 和 reference 信息中推断 query box。

### 5.5 融合 RGB 与 bbox token

默认 `use_rgb=True`，因此走 pretrained feature 分支：

```text
fuse_feat = pose_feat_flat + rgb_flat
```

再加二维位置编码：

```text
fuse_feat += 2D sin-cos positional embedding
```

得到：

```text
fuse_feat: [B, T, 256, 768]
```

### 5.6 跨视图 Transformer

BETR 将视图维度和 patch 维度合并：

```text
[B, T, 256, 768]
-> [B, T*256, 768]
```

然后送入多层 self-attention：

```text
num_decoder_layers = 12
nhead = 8
d_model = 768
```

这个 self-attention 同时在两种维度上传播信息：

```text
同一图像内部 patch 与 patch
不同视图之间 reference 与 query
```

所以 query view 的 token 可以吸收：

- 自己 RGB 外观信息。
- reference RGB 外观信息。
- reference 已知 8 角点 heatmap 信息。
- 跨视图几何与外观对应关系。

### 5.7 只取 query token 输出

Transformer 结束后恢复形状：

```text
[B, T*256, 768]
-> [B, T, 256, 768]
```

然后只取 `camera_mask=True` 的 query view：

```text
query_camera_feat: [B, 256, 768]
```

通过 `bbox_proj` 投影为每个 patch 的 heatmap 像素：

```text
Linear(768 -> 14*14*8)
```

再 unpatchify：

```text
[B, 256, 14*14*8]
-> [B, 8, 224, 224]
```

对于 heatmap 表示，最后做：

```python
query_ret = sigmoid(query_ret)
query_ret = 2 * query_ret - 1
```

所以输出范围是：

```text
[-1, 1]
```

## 6. 训练阶段如何计算 loss

Lightning 模块入口：

```text
src/lightning/BoxDreamer_lightning_model.py
training_step()
```

训练过程：

```text
self.BoxDreamer(batch)
loss_value, loss_details = self.train_loss(batch)
```

默认 loss 配置：

```text
configs/model/loss/default.yaml
```

核心监督：

```yaml
type: smooth_l1
pred_key: pred_bbox
gt_key: bbox_feat
mask_key: camera_mask
weight: [1.0, 0.0]
```

含义：

```text
只在 query view 上计算 pred_bbox 和 bbox_feat 的 SmoothL1Loss
reference views 不计算 loss
```

因此训练目标是：

```text
给定 reference 的真实 bbox heatmap 和所有 RGB
预测 query 的 bbox heatmap
```

## 7. 测试阶段如何得到角点和姿态

测试时 `BoxDreamer.forward()` 在得到 `query_ret` 后会调用：

```text
process_prediction()
```

对于 `pose_representation = bb8`：

```text
query_ret: [B, 8, 224, 224]
-> permute 为 [B, H, W, 8]
-> recover_bb8_corners()
-> 得到 pred_proj_bbox: [B, 8, 2]
```

`recover_bb8_corners()` 对 heatmap 的做法：

```text
每个角点通道取 top-k 响应位置
默认 k = 20
对 top-k 像素坐标求平均
得到该角点的 2D 坐标
```

然后：

```text
2D corners + 3D bbox corners + K
-> cv2.solvePnP
-> pred_poses
```

最终输出：

```text
regression_boxes: 归一化 2D 角点
pred_poses:       4x4 位姿矩阵
```

Demo 中会将角点转为 224 crop 坐标：

```python
box = ((box + 1) / 2) * 224
```

再画出 3D box。

## 8. Demo 推理链路

CLI 入口：

```text
boxdreamer-cli
src/demo/cli.py
src/demo/demo.py
```

真实视频 demo 过程：

```text
输入视频
-> SAM / GroundingDINO 分割目标
-> 保存 mask 和 bbox
-> 从视频中选择参考帧
-> DUSt3R 重建参考帧，得到参考相机位姿、内参、点云模型
-> 写入 reference pose / intrinsics
-> CustomDataset 组织 reference + query
-> BoxDreamer 预测 query 8 角点 heatmap
-> 解码角点、PnP 位姿
-> 绘制 3D box / 点云 / 输出视频
```

所以 demo 不是单张图直接进模型，而是带有前处理与参考重建的完整系统。

## 9. Dense Reference 可选路径

默认：

```text
dense_cfg.enable = False
```

如果启用 dense reference，代码会使用 DINO 特征在大量参考视图中选取与 query 最相似的 top-k：

```text
src/models/utils/data_processing.py
process_dense_input()
dino_matching()
```

流程：

```text
计算 query 与所有 reference 的 DINO patch 相似度
-> 选择 top-k reference
-> 只把这些 reference 和 query 输入 BETR
```

这用于参考库很大时降低计算量，并提升邻近视图质量。

## 10. 用一句话理解 BoxDreamer

BoxDreamer 可以理解为一个“跨视图 box-corner 补全网络”：

```text
它把参考图中已知的 8 角点 heatmap 当作条件，
把查询图的 8 角点 heatmap mask 掉，
再用 DINO 图像特征和 Transformer 跨视图注意力，
把查询图中应该出现的 3D box 八角点 dream 出来。
```

它的强项是泛化到新物体、少参考视图、无需推理时提供 CAD；但对于单类、单图 RGB 到 8 角点 heatmap 的任务，BoxDreamer 的几何和 heatmap 组件值得复用，完整多视图框架则偏重。
