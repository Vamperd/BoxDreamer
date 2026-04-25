# MyNet ResNet34 训练说明

MyNet 是面向 DJI Action4 单类目标的单图 8 角点 heatmap 网络。它读取 `scripts/build_mynet_dataset.py` 生成的数据，将每个 DJI Action4 实例裁剪为一个 ROI，并预测 8 通道角点热力图。

```text
data/dji_action4_mynet/
  meta.json
  train.json
  val.json
  crops/
  heatmaps/
  debug_vis/
```

训练输入是 `crops/*/*.png`，监督是 `heatmaps/*/*.npy`，网络输出为 `[B, 8, 64, 64]`。

## 1. Ubuntu + RTX4090 环境

建议在训练机的仓库根目录使用 UV 创建独立环境：

```bash
uv venv .venv-mynet --python 3.11
source .venv-mynet/bin/activate
```

RTX4090 推荐使用 CUDA 版 PyTorch。以下 CUDA 12.1 wheel 通常适合 4090；若驱动过旧，先用 `nvidia-smi` 确认驱动支持 CUDA 12.x。

```bash
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
uv pip install numpy pillow tqdm tensorboard opencv-python
```

验证 GPU 与 CUDA：

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
PY
```

如果只做 CPU 调试：

```bash
uv pip install torch torchvision torchaudio numpy pillow tqdm tensorboard opencv-python
```

## 2. 数据准备

如果已经在 Ubuntu 上跑通数据生成，应确认以下文件存在：

```text
data/dji_action4_mynet/train.json
data/dji_action4_mynet/val.json
data/dji_action4_mynet/crops/train/*.png
data/dji_action4_mynet/heatmaps/train/*.npy
data/dji_action4_mynet/crops/val/*.png
data/dji_action4_mynet/heatmaps/val/*.npy
```

快速检查样本数：

```bash
python - <<'PY'
import json
for split in ["train", "val"]:
    path = f"data/dji_action4_mynet/{split}.json"
    data = json.load(open(path, "r", encoding="utf-8"))
    print(split, len(data))
PY
```

正式训练时 `val.json` 不应为空，否则只能看到训练 loss，无法判断泛化效果。

## 3. 开始训练

RTX4090 24GB 显存可以先尝试 `batch-size 64`。如果显存不足，降到 `32` 或 `16`。

```bash
python -m src.mynet.train \
  --data-root data/dji_action4_mynet \
  --output-dir models/checkpoints/mynet_resnet34 \
  --epochs 100 \
  --batch-size 64 \
  --num-workers 8 \
  --lr 1e-4 \
  --weight-decay 1e-4 \
  --amp \
  --tensorboard
```

如果数据读取成为瓶颈，可把 `--num-workers` 提到 `12` 或 `16`；如果 CPU 或磁盘压力较大，保持 `8` 更稳。

CPU 调试命令：

```bash
python -m src.mynet.train \
  --data-root data/dji_action4_mynet \
  --output-dir models/checkpoints/mynet_resnet34_cpu \
  --device cpu \
  --no-amp \
  --batch-size 8
```

## 4. TensorBoard 查看指标

训练脚本默认启用 TensorBoard，日志默认保存到：

```text
models/checkpoints/mynet_resnet34/tensorboard/
```

训练中或训练后启动：

```bash
tensorboard \
  --logdir models/checkpoints/mynet_resnet34/tensorboard \
  --host 0.0.0.0 \
  --port 6006
```

如果训练机是远程 Ubuntu，推荐在本机开 SSH 端口转发：

```bash
ssh -L 6006:localhost:6006 user@your_ubuntu_host
```

然后在本机浏览器打开：

```text
http://localhost:6006
```

也可以指定日志目录：

```bash
python -m src.mynet.train \
  --data-root data/dji_action4_mynet \
  --output-dir models/checkpoints/mynet_resnet34 \
  --log-dir runs/mynet_resnet34_action4 \
  --tensorboard
```

## 5. 指标含义

TensorBoard 中主要看这些曲线：

- `train/loss_step`：每个训练 step 的 heatmap MSE，下降越稳定越好。
- `train/loss_epoch`：每个 epoch 的平均训练 loss，用于看整体收敛。
- `val/loss`：验证集 heatmap MSE，是选择 `best.pt` 的主要依据。
- `val/corner_px`：8 个角点在 crop 坐标下的平均像素误差，越低越好。
- `val/pck_2`、`val/pck_5`、`val/pck_10`：预测角点落在 2/5/10 像素阈值内的比例，越高越好。
- `train/lr`：当前学习率，用于确认训练超参是否符合预期。

判断训练是否有效时，优先看 `val/corner_px` 是否下降、`val/pck_5` 和 `val/pck_10` 是否上升。只看训练 loss 容易误判过拟合。

## 6. 非 TensorBoard 方式

每个 epoch 的指标也会写入：

```text
models/checkpoints/mynet_resnet34/metrics.jsonl
```

快速读取：

```bash
python - <<'PY'
import json
path = "models/checkpoints/mynet_resnet34/metrics.jsonl"
for line in open(path, "r", encoding="utf-8"):
    rec = json.loads(line)
    val = rec.get("val", {})
    print(
        rec["epoch"],
        "train_loss=", round(rec["train"]["loss"], 6),
        "val_loss=", round(val.get("loss", -1), 6),
        "corner_px=", round(val.get("corner_px", -1), 3),
        "pck10=", round(val.get("pck_10", -1), 3),
    )
PY
```

如需禁用 TensorBoard，只保留 `metrics.jsonl`：

```bash
python -m src.mynet.train \
  --data-root data/dji_action4_mynet \
  --output-dir models/checkpoints/mynet_resnet34 \
  --no-tensorboard
```

## 7. 输出文件

训练会保存：

```text
models/checkpoints/mynet_resnet34/
  train_args.json
  metrics.jsonl
  tensorboard/
  best.pt
  last.pt
  epoch_010.pt
  epoch_020.pt
  ...
```

`best.pt` 按验证集 loss 保存；如果 `val.json` 为空，则退化为按训练 loss 保存。`last.pt` 是最后一个 epoch 的权重，适合恢复或继续分析。

## 8. 当前网络结构

当前实现：

```text
RGB crop [B, 3, 256, 256]
-> ResNet34 backbone
-> FPN-style upsample head
-> logits [B, 8, 64, 64]
-> sigmoid(logits)
-> masked MSE heatmap loss
```

网络训练的是 ROI 内 8 个角点热力图。真实视频中同时出现两个 DJI Action4 时，不需要改变 MyNet 本身：检测器先给出两个 bbox，每个 bbox 裁剪成一个 ROI，分别送入 MyNet 得到各自的 8 个角点。

## 9. 常见调整

- 4090 上优先使用 `--amp`，速度和显存占用都会更友好。
- OOM 时先把 `--batch-size 64` 降到 `32`，再降到 `16`。
- `val/corner_px` 长期不降时，优先检查 bbox 是否包住完整物体、角点投影是否在 crop 内、heatmap 是否与 crop 对齐。
- `train/loss` 下降但 `val/loss` 上升时，说明可能过拟合，应增加真实视频风格数据、遮挡/曝光增强，或减小训练轮数。

## 10. 使用真实 image.png 检测效果

MyNet 是 ROI 角点网络，不是整图 detector。因此真实图片推理需要先得到每个 DJI Action4 的 bbox，再把每个 bbox 对应的 ROI 输入网络。当前推理脚本默认支持 1 个或 2 个 bbox。

### 10.1 手动框选 bbox

在有图形界面的机器上，可以直接打开图片手动框选：

```bash
python -m src.mynet.select_bboxes \
  --image image.png \
  --output outputs/mynet_infer/image_png/bboxes.json \
  --preview outputs/mynet_infer/image_png/bboxes_preview.png \
  --min-bboxes 1 \
  --max-bboxes 2
```

操作方式：

- 鼠标拖动框住第一个 DJI Action4。
- 按 `Enter` 或 `Space` 确认当前框。
- 如果有第二个 DJI Action4，继续拖动第二个框并确认。
- 框选完成后按 `Esc` 结束。

输出的 `bboxes.json` 格式为：

```json
{
  "image": "image.png",
  "format": "xyxy_full_image_pixels",
  "bboxes": [
    [120.0, 80.0, 420.0, 360.0],
    [520.0, 90.0, 810.0, 350.0]
  ]
}
```

如果远程 Ubuntu 没有图形界面，可以在本地桌面机器上运行该脚本生成 `bboxes.json`，再把 JSON 放到训练机推理。

### 10.2 使用 bbox 推理

使用刚刚手动框选得到的 bbox：

```bash
python -m src.mynet.infer_image \
  --image image.png \
  --checkpoint models/checkpoints/mynet_resnet34/best.pt \
  --output-dir outputs/mynet_infer/image_png \
  --bbox-json outputs/mynet_infer/image_png/bboxes.json \
  --device cuda
```

也可以直接在命令行传入 1 个 bbox：

```bash
python -m src.mynet.infer_image \
  --image image.png \
  --checkpoint models/checkpoints/mynet_resnet34/best.pt \
  --output-dir outputs/mynet_infer/image_png \
  --bbox 120,80,420,360 \
  --device cuda
```

或者传入 2 个 bbox，格式为 `x1,y1,x2,y2`，单位是原图像素：

```bash
python -m src.mynet.infer_image \
  --image image.png \
  --checkpoint models/checkpoints/mynet_resnet34/best.pt \
  --output-dir outputs/mynet_infer/image_png \
  --bbox 120,80,420,360 \
  --bbox 520,90,810,350 \
  --bbox-padding 0.25 \
  --device cuda
```

输出文件：

```text
outputs/mynet_infer/image_png/
  prediction_overlay.png
  predictions.json
  roi_00.png
  roi_01.png
```

查看重点：

- `prediction_overlay.png`：原图上叠加 bbox、正方形 crop 框、8 个预测角点和连线。
- `predictions.json`：保存每个实例的 `corners_2d_full`，也就是回到原图坐标系的 8 个角点。
- `roi_*.png`：实际送入网络的 ROI，可检查 bbox 和 padding 是否合理。

如果 bbox 来自 detector，也可以写成相同 JSON：

```json
{
  "bboxes": [
    [120, 80, 420, 360],
    [520, 90, 810, 350]
  ]
}
```

如果没有真实标注角点，此测试只能做可视化判断；如果要定量评估，需要另外标注 `image.png` 中两个目标的 8 个 2D 角点，再计算平均角点误差和 PCK。

## 11. 标注真实视频帧并导出训练数据

如果需要用一个真实视频同时制作 YOLO detector 数据和 MyNet 微调数据，使用：

```bash
python -m src.mynet.annotate_video \
  --video data/raw/action4.mp4 \
  --output-root data/dji_action4_video_annot \
  --sample-every-sec 1.0 \
  --min-bboxes 1 \
  --max-bboxes 2 \
  --crop-size 256 \
  --heatmap-size 64 \
  --sigma 2.0 \
  --bbox-padding 0.25
```

每个候选帧的流程：

- 先用鼠标框选 1-2 个 DJI Action4 bbox。
- 框选后按 `c` 标注 8 个角点，按 `b` 只保存 bbox 给 YOLO，按 `s` 跳过。
- 角点标注窗口中，左键点击当前角点，`i` 标为不可见，`u` 撤销，`r` 重标当前实例，`Enter` 完成当前实例。
- 完成一帧后，按 `y` 保存 bbox + 角点，按 `b` 只保存 bbox，按 `q` 保存进度并退出。

输出结构：

```text
data/dji_action4_video_annot/
  annotations.json
  frames/all/*.png
  previews/*.jpg
  yolo/
    data.yaml
    images/{train,val,test}/*.png
    labels/{train,val,test}/*.txt
  mynet/
    meta.json
    train.json
    val.json
    test.json
    crops/{train,val,test}/*.png
    heatmaps/{train,val,test}/*.npy
    debug_vis/{train,val,test}/*.jpg
```

训练 YOLO detector：

```bash
yolo detect train \
  data=data/dji_action4_video_annot/yolo/data.yaml \
  model=yolo11n.pt \
  imgsz=960 \
  epochs=100
```

微调 MyNet：

```bash
python -m src.mynet.train \
  --data-root data/dji_action4_video_annot/mynet \
  --train-index data/dji_action4_video_annot/mynet/train.json \
  --val-index data/dji_action4_video_annot/mynet/val.json \
  --output-dir models/checkpoints/mynet_video_ft \
  --epochs 80 \
  --batch-size 32 \
  --lr 3e-5 \
  --amp
```

脚本支持断点续标：重新运行同一条命令时会读取已有 `annotations.json` 并跳过已处理帧。若要从头开始，增加 `--overwrite`。
