#!/bin/sh

## clone the PyTorch repository to setup exact directory structures as the original trained
git clone https://github.com/pytorch/vision.git
cd vision
git checkout v0.3.0
cp references/detection/engine.py ../
cp references/detection/transforms.py ../
cp references/detection/utils.py ../
cp references/detection/coco_utils.py ../
grep -v torch._six references/detection/coco_eval.py > ../coco_eval.py
cd /content
grep -v "import torch._six" vision/references/detection/coco_eval.py | sed 's/torch._six.string_classes/str/' > coco_eval.py


sudo apt install imagemagick
convert -crop 4x3@ cropped.png +repage +adjoin /content/dataset/images/%0d.png
convert -crop 4x3@ mask.png +repage +adjoin /content/dataset/masks/%0d.png
