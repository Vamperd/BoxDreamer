# MyNet 改造与数据集构建计划

## 1. 目标定位

本计划目标是基于当前 BoxDreamer 仓库和 `BACKGROUND.md` 中的 MyNet 需求，构建一条完整可执行路线：

```text
HCCEPose / BlenderProc 已有渲染数据
-> 构建 DJI Action 4 单类 8 角点监督数据集
-> 训练 RGB 图像到 8 个角点 heatmap 的模型
-> 在真实 DJI Action 4 图像上预测并可视化 8 个角点
```

本项目第一版不以恢复 6D 位姿为直接目标，而是以稳定预测 8 个包围盒角点为目标。深度图、物体位姿、相机参数只用于生成监督标签，不作为模型第一版输入。

## 2. BoxDreamer 中可复用与不建议直接沿用的部分

### 2.1 可复用能力

BoxDreamer 已经包含接近 MyNet 需求的 8 角点表示，主要集中在以下模块：

- `src/datasets/base.py`
  - 读取图像、bbox、pose、intrinsic。
  - 将 3D bbox 投影到裁剪后的 2D 图像坐标。
  - 生成 `bbox_feat` 作为 8 角点监督。

- `src/datasets/utils/base/bbox_utils.py`
  - `prepare_bbox3d()`：从 3D 模型或点云生成 8 个 3D bbox corner。
  - `make_bbox_features()`：支持将 8 个 2D corner 生成 heatmap / voting 形式监督。
  - `adjust_bbox_by_proj()`：由 8 个投影角点生成外接 2D bbox。

- `src/models/utils/box_utils.py`
  - `recover_bb8_corners()`：从 8 通道 heatmap 解码二维角点。
  - `recover_pose_from_bb8()`：从角点和 3D bbox 做 PnP，可作为后续扩展，不是第一版必须。

- `src/lightning/utils/vis/vis_utils.py`
  - `draw_3d_box()`、`reproj()` 等可视化与重投影工具。

### 2.2 不建议直接沿用的部分

BoxDreamer 的主模型不是单图角点检测模型，而是：

```text
参考视图 RGB + 参考视图 bbox heatmap + 查询视图 RGB
-> Transformer
-> 查询视图 bbox heatmap / pose
```

它依赖参考图、参考位姿、参考重建、DUSt3R、CustomDataset、多帧序列等机制。对于 `BACKGROUND.md` 中的单类 DJI Action 4、单目 RGB、ROI heatmap 网络来说，直接改 BoxDreamer 主干会引入过多无关复杂度。

因此推荐路线是：

```text
复用 BoxDreamer 的几何与 heatmap 工具
新建 MyNet 数据集、模型、训练脚本
不直接继承 BoxDreamer 的多视图 Transformer 推理框架
```

## 3. 数据集构建完整路径

### 3.1 输入数据来源

已有或待生成的数据来自 HCCEPose / BlenderProc 渲染的 BOP 风格数据，建议保持如下结构：

```text
data/dji_action4/
  models/
    obj_000001.ply
    models_info.json
  train_pbr/
    000000/
      rgb/
        000000.png
      depth/
        000000.png
      mask/
        000000_000000.png
      mask_visib/
        000000_000000.png
      scene_camera.json
      scene_gt.json
      scene_gt_info.json
```

第一版 MyNet 训练最少需要：

- RGB 图像。
- 可见 mask 或完整 mask，用于得到目标 ROI。
- `scene_camera.json` 中的 `cam_K`。
- `scene_gt.json` 中的 `cam_R_m2c`、`cam_t_m2c`、`obj_id`。
- `models_info.json` 或 `obj_000001.ply`，用于定义 3D bbox 八角点。

### 3.2 统一 3D bbox 八角点顺序

必须固定 8 个角点顺序，否则 heatmap 通道含义会混乱。建议使用 BoxDreamer 中 `consist_bbox3d()` 风格的顺序：

```text
0: min_x, min_y, min_z
1: min_x, max_y, min_z
2: max_x, max_y, min_z
3: max_x, min_y, min_z
4: min_x, min_y, max_z
5: min_x, max_y, max_z
6: max_x, max_y, max_z
7: max_x, min_y, max_z
```

构建方式：

1. 从 `models_info.json` 读取 `min_x/min_y/min_z/size_x/size_y/size_z`，或从 `obj_000001.ply` 点云求 min/max。
2. 生成固定顺序的 `corners_3d`，shape 为 `[8, 3]`。
3. 将该顺序保存为数据集元信息，训练、推理、可视化全部沿用。

建议新增：

```text
data/dji_action4_mynet/meta.json
```

内容示例：

```json
{
  "object_name": "DJI Action 4",
  "obj_id": 1,
  "corner_order": "xmin/ymin/zmin -> ... fixed 8 corners",
  "image_size": 256,
  "heatmap_size": 64,
  "crop_padding_ratio": 0.25
}
```

### 3.3 从位姿投影 3D 角点到 2D

对每一张渲染图中的目标实例执行：

```text
corners_cam = R @ corners_3d + t
u = fx * X / Z + cx
v = fy * Y / Z + cy
```

输入：

- `K` 来自 `scene_camera.json`。
- `R, t` 来自 `scene_gt.json`。
- `corners_3d` 来自上一步。

输出：

```text
corners_2d_full: [8, 2]
```

坐标应保留浮点数，不要提前取整。后续生成 heatmap 时再使用浮点中心生成高斯响应。

### 3.4 构建 ROI crop

第一版网络主线是：

```text
detector / mask bbox
-> ROI crop
-> fixed-size RGB crop
-> 8-channel heatmap
```

训练阶段 ROI 来源推荐：

1. 优先使用 `mask_visib` 计算可见 bbox。
2. 若目标严重遮挡导致可见 bbox 太小，可用 `mask` 或投影 8 角点 bbox 做补充。
3. bbox 加 padding，推荐 `padding_ratio = 0.25`。
4. 将 bbox 调整为正方形，避免 resize 改变物体比例。
5. crop 超出图像边界时用 0 或均值 padding，不丢弃样本。

每个样本保存：

```text
bbox_xyxy_full: 原图坐标 bbox
crop_box_xyxy_full: 加 padding 后的正方形 crop 框
```

### 3.5 将角点映射到 crop 坐标

给定原图角点 `corners_2d_full` 和 crop 框：

```text
x_crop = (x_full - crop_x1) * crop_size / crop_width
y_crop = (y_full - crop_y1) * crop_size / crop_height
```

输出：

```text
corners_2d_crop: [8, 2]
```

第一版建议：

- RGB crop 尺寸：`256 x 256`。
- heatmap 尺寸：`64 x 64`。
- 角点坐标先映射到 `256 x 256`，再按比例映射到 `64 x 64` 生成 heatmap。

如果某些角点投影在 crop 外：

- 不要直接丢弃样本。
- heatmap 中允许峰值落在边界外时被截断。
- 同时保存 `corner_valid` 标志，后续可以选择只监督 crop 内角点或全部角点。

第一版默认：全部 8 个角点都参与监督，heatmap 在图内部分自然截断。

### 3.6 生成 8 通道 heatmap

每个样本生成：

```text
heatmap: [8, heatmap_h, heatmap_w]
```

推荐方式：

```text
每个角点一个通道
以角点位置为中心生成 2D Gaussian
峰值为 1
背景为 0
```

建议参数：

- `heatmap_size = 64`
- `sigma = 1.5` 到 `2.5`
- 若角点在 heatmap 外，则该通道全 0，并在 `corner_valid` 中记为 0。

注意：BoxDreamer 的 `make_bbox_features(type="heatmap")` 输出范围是 `[-1, 1]`，并且高斯生成逻辑与常见关键点 heatmap 略有不同。MyNet 第一版建议使用更标准的 `[0, 1]` heatmap，配合 `MSELoss` 或 `FocalLoss`，实现更直观。

### 3.7 生成 MyNet 专用索引文件

建议生成如下结构：

```text
data/dji_action4_mynet/
  meta.json
  train.json
  val.json
  test_real.json
  crops/
    train/
      000000.png
  heatmaps/
    train/
      000000.npy
  debug_vis/
    train/
      000000.jpg
```

`train.json` 每条样本建议包含：

```json
{
  "sample_id": "000000_000000_000000",
  "rgb_path": "data/dji_action4/train_pbr/000000/rgb/000000.png",
  "mask_path": "data/dji_action4/train_pbr/000000/mask_visib/000000_000000.png",
  "crop_path": "data/dji_action4_mynet/crops/train/000000.png",
  "heatmap_path": "data/dji_action4_mynet/heatmaps/train/000000.npy",
  "obj_id": 1,
  "K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
  "R": [[...], [...], [...]],
  "t": [tx, ty, tz],
  "bbox_xyxy_full": [x1, y1, x2, y2],
  "crop_box_xyxy_full": [x1, y1, x2, y2],
  "corners_3d": [[x, y, z]],
  "corners_2d_full": [[u, v]],
  "corners_2d_crop": [[u, v]],
  "corner_valid": [1, 1, 1, 1, 1, 1, 1, 1]
}
```

### 3.8 训练/验证划分

不要随机按图片完全打散，否则相近渲染视角可能同时进入训练和验证，验证结果会虚高。

推荐划分：

- 按 scene id 划分：80% scene 训练，20% scene 验证。
- 或按视角/姿态划分：保留一部分方位角、俯仰角组合做验证。
- 真实图片只做最终测试，不进入第一版训练。

### 3.9 数据质量检查

在正式训练前必须生成 `debug_vis`：

```text
原图 + 8 角点
crop 图 + 8 角点
8 通道 heatmap 叠加图
```

检查重点：

- 角点顺序是否稳定。
- crop 后角点是否仍落在合理位置。
- bbox padding 是否过大或过小。
- DJI Action 4 长宽高方向是否与模型坐标一致。
- 是否存在 `t` 单位错误，BOP 常见为毫米，部分渲染流程可能是米。

## 4. 模型改造完整路径

### 4.1 不直接修改 BoxDreamer 主模型

新建 MyNet 模块，保留 BoxDreamer 原有推理与训练能力。推荐新增：

```text
src/mynet/
  __init__.py
  datasets/
    __init__.py
    bop_corner_dataset.py
    heatmap.py
    geometry.py
  models/
    __init__.py
    corner_resnet.py
  losses.py
  decode.py
  train.py
  infer.py
  visualize.py
configs/mynet/
  train.yaml
  infer.yaml
```

### 4.2 Dataset 设计

`BOPCornerDataset` 读取 `train.json / val.json`，输出：

```python
{
    "image": Tensor[3, 256, 256],
    "heatmap": Tensor[8, 64, 64],
    "corners_2d_crop": Tensor[8, 2],
    "corner_valid": Tensor[8],
    "crop_box_xyxy_full": Tensor[4],
    "K": Tensor[3, 3],
    "R": Tensor[3, 3],
    "t": Tensor[3],
    "sample_id": str
}
```

第一版训练只使用：

```text
image -> heatmap
```

其余字段用于评估、可视化和后续扩展。

### 4.3 网络结构

第一版使用 ResNet 主线，符合 `BACKGROUND.md`：

```text
RGB crop [3, 256, 256]
-> ResNet18 / ResNet34 backbone
-> deconv / FPN upsample head
-> heatmap logits [8, 64, 64]
```

推荐初始配置：

- Backbone：`torchvision.models.resnet18(weights=ImageNet1K_V1)`
- 输出 stride：4
- Head：
  - `Conv 3x3 + BN + ReLU`
  - `Upsample x2`
  - `Conv 3x3 + BN + ReLU`
  - `Upsample x2`
  - `Conv 1x1 -> 8 channels`
- 输出不在模型内 sigmoid，训练用 `BCEWithLogitsLoss` 或 `MSELoss` 时统一处理。

建议第一版损失：

```text
loss = MSELoss(sigmoid(pred_heatmap), gt_heatmap)
```

如果正负样本极不平衡明显，再切换到 focal heatmap loss。

### 4.4 角点解码

推理阶段：

```text
pred_heatmap = sigmoid(logits)
每通道取 argmax 或 soft-argmax
得到 corners_2d_heatmap
映射回 crop 坐标
再映射回原图坐标
```

坐标恢复：

```text
x_crop = x_hm * crop_size / heatmap_w
y_crop = y_hm * crop_size / heatmap_h

x_full = x_crop * crop_width / crop_size + crop_x1
y_full = y_crop * crop_height / crop_size + crop_y1
```

第一版建议同时输出：

```text
corners_2d_crop.json
corners_2d_full.json
vis_crop.jpg
vis_full.jpg
```

### 4.5 训练流程

推荐训练命令：

```bash
python -m src.mynet.train --config configs/mynet/train.yaml
```

`configs/mynet/train.yaml` 建议包含：

```yaml
data:
  root: data/dji_action4_mynet
  train_index: train.json
  val_index: val.json
  image_size: 256
  heatmap_size: 64

model:
  backbone: resnet18
  pretrained: true
  out_channels: 8

train:
  batch_size: 32
  epochs: 100
  lr: 0.0001
  weight_decay: 0.0001
  num_workers: 8
  amp: true

loss:
  type: mse_heatmap

output:
  ckpt_dir: models/checkpoints/mynet_dji_action4
  vis_dir: outputs/mynet_debug
```

训练中记录：

- train heatmap loss
- val heatmap loss
- 角点平均像素误差，单位为 crop pixel 和 original pixel
- PCK@2px / PCK@5px / PCK@10px
- 每个 epoch 保存若干张可视化图

### 4.6 真实图片推理流程

真实图片没有 GT pose，因此推理只需要：

```text
真实 RGB 图像
-> detector / 手动 bbox / SAM mask
-> ROI crop
-> MyNet
-> 8 heatmap
-> 8 corner
-> 可视化
```

第一版 bbox 获取方式可以按优先级选择：

1. 手动框选 DJI Action 4。
2. 使用已有检测器输出 bbox。
3. 使用 SAM / GroundingDINO 得到 mask 后转 bbox。

真实推理输入索引示例：

```json
{
  "image_path": "real_images/000001.jpg",
  "bbox_xyxy_full": [x1, y1, x2, y2]
}
```

输出：

```text
outputs/mynet_real/
  000001_corners.json
  000001_vis.jpg
```

### 4.7 双 DJI Action 4 实例的 bbox 划分与检测器选择

实际检验视频帧中同时存在两个 DJI Action 4，因此推理阶段不能假设一张图只有一个目标。第一版建议仍保持 MyNet 为“单实例 ROI 角点网络”，在 MyNet 之前增加一个多实例 detector，把每一帧中的两个目标拆成两个独立 ROI：

```text
一帧 RGB
-> detector 输出 2 个 DJI Action 4 bbox
-> 对每个 bbox 分别 crop
-> MyNet 分别预测 8 角点 heatmap
-> 将每个实例角点映射回原图
-> 按 instance_id 保存和可视化
```

推荐的真实推理输出格式：

```json
{
  "image_path": "real_images/000001.jpg",
  "instances": [
    {
      "instance_id": "left",
      "bbox_xyxy_full": [x1, y1, x2, y2],
      "score": 0.93,
      "corners_2d_full": [[u, v]]
    },
    {
      "instance_id": "right",
      "bbox_xyxy_full": [x1, y1, x2, y2],
      "score": 0.91,
      "corners_2d_full": [[u, v]]
    }
  ]
}
```

#### bbox 划分原则

每个 DJI Action 4 必须作为一个独立实例样本进入 MyNet：

- detector 输出的每个 bbox 单独做 square padding 和 resize。
- 两个 bbox 即使有重叠，也不要合并成一个大 bbox。
- 每个 ROI 输出独立的 8 通道 heatmap。
- 映射回原图后分别保存 `instance_id`、bbox、8 角点。
- 如果两个目标相互遮挡，仍以 detector 给出的可见区域 bbox 为基础，加较大 padding，推荐 `0.25 ~ 0.4`。

#### instance_id 分配

如果只是逐帧可视化：

```text
按 bbox 中心 x 坐标排序
左侧目标 -> instance_id = left
右侧目标 -> instance_id = right
```

如果需要跨帧稳定比较：

```text
detector
-> ByteTrack / BoT-SORT / DeepSORT
-> 为每个 DJI Action 4 维护稳定 track_id
-> MyNet 按 track_id 输出角点序列
```

第一版建议先用“按 x 坐标排序”的方式快速验证角点网络；当视频中两个目标会交叉、遮挡或左右位置互换时，再接入 tracker。

#### Detector 推荐

首选方案：微调 Ultralytics YOLO detect / seg。

- 原因：部署简单、速度快、训练数据格式成熟，适合固定单类目标和视频逐帧检测。
- 若新建检测器训练环境，优先使用 Ultralytics 当前最新稳定 detect / seg 模型；若现有环境已经围绕 YOLO11 配好，也可以直接使用 YOLO11。
- detect 版本输出 bbox，已经满足 MyNet ROI 需求。
- seg 版本可额外输出 mask，能在目标贴近或轻微重叠时得到更稳定 bbox。
- 数据量建议：从真实视频中抽帧标注 200~500 张，两台 DJI Action 4 都标注为同一类 `dji_action4`。

备选方案：GroundingDINO / Grounded-SAM。

- 适合冷启动阶段，不想先标注检测数据时使用。
- prompt 可用 `"DJI Action 4 camera"` 或 `"action camera"`。
- 缺点是同类小物体、相似背景、反光场景下稳定性通常不如微调后的专用 detector。
- 可用于自动预标注，再人工修正为 YOLO 训练集。

精度优先备选：RF-DETR。

- 适合后续追求更高检测精度或更强泛化时尝试。
- 第一版不建议优先上，因为 MyNet 主任务是角点热力图，检测器只需要稳定给出两个较准 ROI。

#### 训练数据如何覆盖双实例场景

MyNet 本身仍按单实例 ROI 训练，不需要把两个目标一起输入网络。但为了真实双目标场景稳定，需要额外准备 detector 数据：

```text
real_detector_dataset/
  images/
  labels/
    # YOLO 格式，每张图可以有两行 dji_action4 标注
```

标注规则：

- 每张真实帧中两个 DJI Action 4 都标注 bbox。
- 两个实例类别相同，均为 `dji_action4`。
- bbox 尽量覆盖完整物体外轮廓，而不是只覆盖可见纹理区域。
- 对严重遮挡样本，保留可判断的完整外接框，同时在备注中记录遮挡程度。

真实推理时 detector 和 MyNet 的职责划分为：

```text
Detector: 负责找到每个 DJI Action 4 实例的 2D ROI
MyNet:    负责在单个 ROI 内预测该实例的 8 个 box corner
Tracker:  可选，负责跨帧维持 instance_id
```

## 5. 与 BoxDreamer 现有代码的衔接方式

### 5.1 直接复用

可以从 BoxDreamer 迁移或调用：

- 3D bbox corner 生成逻辑：`prepare_bbox3d()` / `consist_bbox3d()`。
- 3D 到 2D 投影逻辑：`make_proj_bbox()` 或等价实现。
- 可视化：`draw_3d_box()`。
- heatmap 解码思路：`recover_bb8_corners()`。

### 5.2 建议重写

建议在 MyNet 中重写以下轻量模块，避免被 BoxDreamer 多视图数据结构绑住：

- BOP `train_pbr` 解析器。
- ROI crop 与坐标变换。
- 标准 `[0, 1]` Gaussian heatmap 生成。
- 单图 ResNet heatmap 网络。
- 单图训练与推理脚本。

### 5.3 不纳入第一版

以下内容暂不纳入 MyNet 第一版：

- DUSt3R 重建。
- 多参考视图选择。
- BoxDreamer 的 BETR Transformer。
- PnP 位姿恢复作为训练目标。
- RGB-D 输入。
- 真实数据微调。

这些可以在 8 角点预测稳定后再作为第二阶段扩展。

## 6. 推荐实施顺序

### 阶段 A：数据转换与检查

1. 准备 DJI Action 4 的 `obj_000001.ply` 和 `models_info.json`。
2. 使用 HCCEPose / BlenderProc 生成 `train_pbr`。
3. 编写 `build_mynet_dataset.py`：
   - 读取 BOP 数据。
   - 生成固定顺序 3D corners。
   - 投影 2D corners。
   - 生成 ROI crop。
   - 生成 8 通道 heatmap。
   - 输出 `train.json / val.json`。
4. 生成 `debug_vis`，人工检查 100 张以上样本。

### 阶段 B：模型与训练

1. 实现 `BOPCornerDataset`。
2. 实现 `CornerResNet`。
3. 实现 heatmap loss、角点解码、PCK 指标。
4. 在合成数据上训练第一版模型。
5. 每个 epoch 输出预测角点可视化。

### 阶段 C：真实测试

1. 准备真实 DJI Action 4 图片。
2. 为每帧获得两个 DJI Action 4 bbox，优先使用微调后的 Ultralytics YOLO detect / seg。
3. 将两个 bbox 拆成两个独立 ROI，分别运行 `src.mynet.infer`。
4. 输出每个实例的原图角点、bbox、`instance_id` 和可视化。
5. 若需要跨帧稳定身份，增加 ByteTrack / BoT-SORT / DeepSORT。
6. 观察不同视角、遮挡、光照、双目标接近或重叠时的角点稳定性。

## 7. 验收标准

数据集验收：

- 任意抽样可视化中，8 个角点投影位置与 3D box 几何一致。
- crop 图中目标主体完整，padding 不过度。
- heatmap 峰值与 crop 角点位置一致。
- 训练集和验证集无明显相邻视角泄漏。

模型验收：

- 模型输入为单张 RGB crop。
- 模型输出为 8 通道 heatmap。
- 可以解码得到 8 个 crop 坐标和原图坐标。
- 合成验证集 PCK@10px 达到可用水平。
- 真实 DJI Action 4 图片上角点可视化位置基本稳定。

## 8. 最终路径总结

完整数据路径：

```text
DJI Action 4 3D 模型
-> BOP models/obj_000001.ply + models_info.json
-> HCCEPose / BlenderProc train_pbr
-> 读取 RGB / mask / K / R / t
-> 生成固定顺序 3D bbox corners
-> 投影得到 2D corners
-> 根据 mask 或投影 bbox 生成 ROI crop
-> 映射 corners 到 crop 坐标
-> 生成 8 通道 heatmap
-> train.json / val.json / crops / heatmaps / debug_vis
```

完整模型路径：

```text
BoxDreamer 几何与 heatmap 工具
-> 抽取/重写为 MyNet 数据处理工具
-> 新建单图 BOPCornerDataset
-> 新建 ResNet heatmap 模型
-> detector 在真实帧中输出两个 DJI Action 4 ROI
-> 每个 ROI 独立输入 MyNet，8 heatmap 输出
-> heatmap loss 训练
-> argmax / soft-argmax 解码 8 角点
-> 映射回原图并可视化
```

第一版推荐保持目标简单明确：

```text
只做单类 DJI Action 4
只用合成数据训练
只输入 RGB crop
只输出 8 个角点 heatmap
真实图片只做测试验证
```
