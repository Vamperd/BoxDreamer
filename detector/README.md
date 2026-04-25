# DJI Action4 YOLO Detector

这个目录用于存放并训练 DJI Action4 的 bbox detector。数据来自 `src.mynet.annotate_video` 生成的 YOLO 标签：

```text
data/dji_action4_video_annot/yolo/
  data.yaml
  images/train/*.png
  images/val/*.png
  images/test/*.png
  labels/train/*.txt
  labels/val/*.txt
  labels/test/*.txt
```

训练产物默认写入 `detector/runs/`，最终可把 `best.pt` 复制到 `detector/weights/` 作为后续推理使用。权重和训练输出已在 `detector/.gitignore` 中忽略。

## 1. 远端 Ubuntu 环境

在远端 Ubuntu 训练机的仓库根目录执行：

```bash
uv venv .venv-detector --python 3.11
source .venv-detector/bin/activate
```

RTX4090 推荐先安装 CUDA 版 PyTorch，再安装 Ultralytics：

```bash
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
uv pip install ultralytics opencv-python pillow numpy tensorboard
```

确认 GPU 可用：

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
PY
```

## 2. 准备数据

先用视频标注脚本生成 YOLO 数据：

```bash
python -m src.mynet.annotate_video \
  --video data/raw/action4.mp4 \
  --output-root data/dji_action4_video_annot \
  --sample-every-sec 1.0 \
  --min-bboxes 1 \
  --max-bboxes 2
```

检查标签数量：

```bash
find data/dji_action4_video_annot/yolo/labels -name "*.txt" | wc -l
find data/dji_action4_video_annot/yolo/images -name "*.png" | wc -l
```

`data/dji_action4_video_annot/yolo/data.yaml` 应指向 `images/train`、`images/val`、`images/test`。

## 3. 启动训练

推荐先从小模型开始，便于快速验证标注是否正确：

```bash
python detector/train_yolo.py \
  --data data/dji_action4_video_annot/yolo/data.yaml \
  --model yolo11n.pt \
  --project detector/runs \
  --name dji_action4_yolo \
  --imgsz 960 \
  --epochs 100 \
  --batch 16 \
  --device 0 \
  --workers 8 \
  --amp
```

如果显存还有余量，可把 `--batch` 提到 `32`；如果 OOM，降到 `8`。如果希望 Ultralytics 自动估计 batch：

```bash
python detector/train_yolo.py \
  --data data/dji_action4_video_annot/yolo/data.yaml \
  --model yolo11n.pt \
  --batch auto \
  --imgsz 960 \
  --epochs 100
```

中断后继续训练：

```bash
python detector/train_yolo.py \
  --data data/dji_action4_video_annot/yolo/data.yaml \
  --model detector/runs/dji_action4_yolo/weights/last.pt \
  --resume
```

## 4. 查看训练效果

主要看这些文件：

```text
detector/runs/dji_action4_yolo/
  results.csv
  results.png
  confusion_matrix.png
  PR_curve.png
  weights/best.pt
  weights/last.pt
```

重点指标：

- `metrics/mAP50(B)`：bbox 在 IoU 0.5 下的检测效果。
- `metrics/mAP50-95(B)`：更严格的综合 mAP。
- `metrics/precision(B)`：预测框中有多少是对的。
- `metrics/recall(B)`：真实目标有多少被找到了。
- `val/box_loss`：验证集 bbox 回归误差。

也可以用 TensorBoard 查看：

```bash
tensorboard --logdir detector/runs --host 0.0.0.0 --port 6006
```

如果训练机是远端服务器，在本地开 SSH 转发：

```bash
ssh -L 6006:localhost:6006 user@your_ubuntu_host
```

然后浏览器打开 `http://localhost:6006`。

## 5. 推理测试

用训练好的 detector 检测图片或视频：

```bash
python detector/predict_yolo.py \
  --weights detector/runs/dji_action4_yolo/weights/best.pt \
  --source image.png \
  --project detector/predictions \
  --name image_test \
  --imgsz 960 \
  --conf 0.25 \
  --device 0 \
  --save-txt \
  --save-conf
```

检测视频：

```bash
python detector/predict_yolo.py \
  --weights detector/runs/dji_action4_yolo/weights/best.pt \
  --source data/raw/action4.mp4 \
  --project detector/predictions \
  --name video_test \
  --imgsz 960 \
  --conf 0.25
```

如果效果满意，可以保存最终权重：

```bash
mkdir -p detector/weights
cp detector/runs/dji_action4_yolo/weights/best.pt detector/weights/dji_action4_yolo_best.pt
```

后续整图流程建议为：YOLO detector 先输出 1-2 个 DJI Action4 bbox，再把每个 bbox 送入 MyNet 做 8 角点预测。
