# yolov5-deepsparse-blogpost

Repo for the blogpost on using Yolov5 with DeepSparse Engine

## Installation

`pip install tensorboard seaborn pyyaml tqdm sparseml opencv-python`

`pip3 install torch==1.9.0 torchvision==0.10.0 --extra-index-url https://download.pytorch.org/whl/cu111`

### Train
`python train.py --cfg ./models_v5.0/yolov5s.yaml --weights "" --data asl.yaml --hyp data/hyps/hyp.finetune.yaml --weights yolov5s.pt --img 416`

### Export
`python export.py --weights runs/train/exp/weights/best.pt --include onnx`
