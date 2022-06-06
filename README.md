# Supercharging YOLOv5: How I Got 182.4 FPS Inference Without a GPU
![image](https://dicksonneoh.com/images/portfolio/supercharging_yolov5/post_image.png)

Companion repo for the [blogpost](https://dicksonneoh.com/portfolio/supercharging_yolov5_180_fps_cpu/).

## Installation

`pip install tensorboard seaborn pyyaml tqdm sparseml opencv-python deepsparse`

`pip install setuptools==59.5.0`

`pip3 install torch==1.9.0 torchvision==0.10.0 --extra-index-url https://download.pytorch.org/whl/cu111`

### Train
`python train.py --cfg ./models_v5.0/yolov5s.yaml --data pistols.yaml --hyp data/hyps/hyp.scratch.yaml --weights yolov5s.pt --img 416 --batch-size 64 --optimizer SGD --epochs 100 --device 0 --project yolov5-deepsparse --name yolov5s-sgd`

`python train.py --cfg ./models_v5.0/yolov5s.yaml --recipe ../recipes/yolov5s.pruned.md --data pistols.yaml --hyp data/hyps/hyp.scratch.yaml --weights yolov5s.pt --img 416 --batch-size 64 --optimizer SGD --device 0 --project yolov5-deepsparse --name yolov5s-sgd-pruned`

`python train.py --cfg ./models_v5.0/yolov5s.yaml --recipe ../recipes/yolov5.transfer_learn_pruned_quantized.md --data pistols.yaml --hyp data/hyps/hyp.scratch.yaml --weights yolov5s.pt --img 416 --batch-size 64 --optimizer SGD --device 0 --project yolov5-deepsparse --name yolov5s-sgd-pruned-quantized`

`python train.py --data pistols.yaml --cfg ./models_v5.0/yolov5s.yaml --weights zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned_quant-aggressive_94?recipe_type=transfer --img 416 --batch-size 64 --hyp data/hyps/hyp.scratch.yaml --recipe ../recipes/yolov5.transfer_learn_pruned_quantized.md --optimizer SGD --device 0 --project yolov5-deepsparse --name yolov5s-sgd-pruned-quantized-transfer`

#### YOLOv5n
`python train.py --cfg ./models_v5.0/yolov5n.yaml --recipe ../recipes/yolov5.transfer_learn_pruned_quantized.md --data pistols.yaml --hyp data/hyps/hyp.scratch.yaml --weights yolov5n.pt --img 416 --batch-size 64 --optimizer SGD --device 0 --project yolov5-deepsparse --name yolov5n-sgd-pruned-quantized`

### Export
`python export.py --weights yolov5-deepsparse/yolov5s-sgd/weights/best.pt --include onnx --imgsz 416 --dynamic --simplify`

`python export.py --weights yolov5-deepsparse/yolov5s-sgd-pruned/weights/best.pt --include onnx --imgsz 416 --dynamic --simplify`

`python export.py --weights yolov5-deepsparse/yolov5s-sgd-pruned-quantized/weights/best.pt --include onnx --imgsz 416 --dynamic --simplify`

`python export.py --weights yolov5-deepsparse/yolov5s-sgd-pruned-quantized-transfer/weights/best.pt --include onnx --imgsz 416 --dynamic --simplify`

#### YOLOv5n
`python export.py --weights yolov5-deepsparse/yolov5n-sgd-pruned-quantized/weights/best.pt --include onnx --imgsz 416 --dynamic --simplify`

### Detect
`python detect.py --weights yolov5-deepsparse/yolov5s-sgd/weights/best.pt --source data/pexels-cottonbro-8717592.mp4 --data data/pistols.yaml --imgsz 416 --view-img --nosave --device cpu`


### One shot
`python train.py --cfg ./models_v5.0/yolov5s.yaml --data pistols.yaml --hyp data/hyps/hyp.scratch.yaml --weights yolov5-deepsparse/yolov5s-sgd/weights/best.pt --img 416 --batch-size 64 --optimizer SGD --epochs 100 --device 0 --project yolov5-deepsparse --name yolov5s-sgd-one-shot --one-shot`

`python export.py --weights yolov5-deepsparse/yolov5s-sgd-one-shot/weights/checkpoint-one-shot.pt --include onnx --imgsz 416 --dynamic --simplify`

`python annotate.py yolov5-deepsparse/yolov5s-sgd-one-shot/weights/checkpoint-one-shot.onnx --source data/pexels-cottonbro-8717592.mp4 --engine deepsparse --device cpu --conf-thres 0.7 --image-shape 416 416 --num-cores 4`

### Annotate CPU PyTorch Engine
`python annotate.py yolov5-deepsparse/yolov5s-sgd/weights/best.pt --source data/pexels-cottonbro-8717592.mp4 --engine torch --image-shape 416 416 --device cpu --conf-thres 0.7`

### Annotate ONNX No Optimization
`python annotate.py yolov5-deepsparse/yolov5s-sgd/weights/best.onnx --source data/pexels-cottonbro-8717592.mp4 --engine deepsparse --device cpu --conf-thres 0.7 --image-shape 416 416 --num-cores 4`

### Annotate with Pruning
`python annotate.py yolov5-deepsparse/yolov5s-sgd-pruned/weights/best.onnx --source data/pexels-cottonbro-8717592.mp4 --engine deepsparse --device cpu --conf-thres 0.7 --image-shape 416 416 --num-cores 4`

### Annotate with Pruning and Quant
`python annotate.py yolov5-deepsparse/yolov5s-sgd-pruned-quantized/weights/best.onnx --source data/pexels-cottonbro-8717592.mp4 --engine deepsparse --device cpu --conf-thres 0.7 --image-shape 416 416 --quantized-input --num-cores 4`

### Annotate with Pruning and Quant and TL
`python annotate.py yolov5-deepsparse/yolov5s-sgd-pruned-quantized-transfer/weights/best.onnx --source data/pexels-cottonbro-8717592.mp4 --engine deepsparse --device cpu --conf-thres 0.8 --image-shape 416 416 --quantized-input --num-cores 4`

`python annotate.py yolov5-deepsparse/yolov5n-sgd-pruned-quantized-hardswish/weights/best.onnx --source data/pexels-cottonbro-8717592.mp4 --engine deepsparse --device cpu --conf-thres 0.7 --image-shape 416 416 --quantized-input --num-cores 4`

### Wandb Dashboard
https://wandb.ai/dnth/yolov5-deepsparse?workspace=user-dnth
