# yolov5-deepsparse-blogpost

Repo for the blogpost on using Yolov5 with DeepSparse Engine

## Installation

`pip install tensorboard seaborn pyyaml tqdm sparseml opencv-python deepsparse`


`pip3 install torch==1.9.0 torchvision==0.10.0 --extra-index-url https://download.pytorch.org/whl/cu111`

### Train
`python train.py --cfg ./models_v5.0/yolov5s.yaml --data pistols.yaml --hyp data/hyps/hyp.finetune.yaml --weights yolov5s.pt --img 416 --batch-size 256 --optimizer SGD --epochs 300 --device 0 --project yolov5-deepsparse --name yolov5s-sgd`

### Export
`python export.py --weights yolov5-deepsparse/yolov5s-sgd/weights/best.pt --include onnx`

### Detect
`python detect.py --weights yolov5-deepsparse/yolov5s-sgd/weights/best.pt --source data/pexels-koolshooters-8105530.mp4 --data data/pistols.yaml --imgsz 416 --view-img --nosave --device cpu`


### Annotate CPU PyTorch Engine
`python annotate.py yolov5-deepsparse/yolov5s-sgd/weights/best.pt --source data/pexels-karolina-grabowska-5243197.mp4 --no-save --engine torch --image-shape 416 416 --device cpu --conf-thres 0.6`

### Annotate ONNX No Optimization
`python annotate.py yolov5-deepsparse/yolov5s-sgd/weights/best.onnx --source data/pexels-karolina-grabowska-5243197.mp4 --no-save --engine deepsparse --device cpu --conf-thres 0.6`