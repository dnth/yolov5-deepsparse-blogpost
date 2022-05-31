# yolov5-deepsparse-blogpost

Repo for the blogpost on using Yolov5 with DeepSparse Engine

## Installation

`pip install tensorboard seaborn pyyaml tqdm sparseml opencv-python deepsparse`

`pip install setuptools==59.5.0`

`pip3 install torch==1.9.0 torchvision==0.10.0 --extra-index-url https://download.pytorch.org/whl/cu111`

### Train
`python train.py --cfg ./models_v5.0/yolov5s.yaml --data pistols.yaml --hyp data/hyps/hyp.scratch.yaml --weights yolov5s.pt --img 416 --batch-size 64 --optimizer SGD --epochs 100 --device 0 --project yolov5-deepsparse --name yolov5s-sgd`

`python train.py --cfg ./models_v5.0/yolov5s.yaml --recipe ../recipes/yolov5s.pruned.md --data pistols.yaml --hyp data/hyps/hyp.scratch.yaml --weights yolov5s.pt --img 416 --batch-size 64 --optimizer SGD --device 0 --project yolov5-deepsparse --name yolov5s-sgd-pruned`

`python train.py --cfg ./models_v5.0/yolov5s.yaml --recipe ../recipes/yolov5.transfer_learn_pruned_quantized.md --data pistols.yaml --hyp data/hyps/hyp.scratch.yaml --weights yolov5s.pt --img 416 --batch-size 64 --optimizer SGD --device 0 --project yolov5-deepsparse --name yolov5s-sgd-pruned-quantized`

### Export
`python export.py --weights yolov5-deepsparse/yolov5s-sgd/weights/best.pt --include onnx --imgsz 416`

`python export.py --weights yolov5-deepsparse/yolov5s-sgd-pruned/weights/best.pt --include onnx --imgsz 416`

`python export.py --weights yolov5-deepsparse/yolov5s-sgd-pruned-quantized/weights/best.pt --include onnx --imgsz 416`

### Detect
`python detect.py --weights yolov5-deepsparse/yolov5s-sgd/weights/best.pt --source data/pexels-cottonbro-8717592.mp4 --data data/pistols.yaml --imgsz 416 --view-img --nosave --device cpu`


### Annotate CPU PyTorch Engine
`python annotate.py yolov5-deepsparse/yolov5s-sgd/weights/best.pt --source data/pexels-cottonbro-8717592.mp4 --no-save --engine torch --image-shape 416 416 --device cpu --conf-thres 0.6`

### Annotate ONNX No Optimization
`python annotate.py yolov5-deepsparse/yolov5s-sgd/weights/best.onnx --source data/pexels-cottonbro-8717592.mp4 --no-save --engine deepsparse --device cpu --conf-thres 0.6 --image-shape 416 416`

### Annotate with Pruning
`python annotate.py yolov5-deepsparse/yolov5s-sgd-pruned/weights/best.onnx --source data/pexels-cottonbro-8717592.mp4 --no-save --engine deepsparse --device cpu --conf-thres 0.6 --image-shape 416 416`

### Annotate with Pruning and Quant
`python annotate.py yolov5-deepsparse/yolov5s-sgd-pruned-quantized/weights/best.onnx --source data/pexels-cottonbro-8717592.mp4 --no-save --engine deepsparse --device cpu --conf-thres 0.3 --image-shape 416 416 --quantized-input`