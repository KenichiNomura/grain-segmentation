import numpy as np
import sys, os, random
import cv2

import matplotlib
import matplotlib.pylab as plt

import torch, torchvision

plt.rcParams["axes.grid"] = False

import torch
import torch.utils.data
from torch.utils.data import Dataset

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from PIL import Image

def draw_image(image, output, coco_names):

  result_image = np.array(image.copy())
  colors = [[random.randint(0, 255) for _ in range(3)] for _ in coco_names]
  
  for box, label, score in zip(output['boxes'], output['labels'], output['scores']):
    if score > 0.5:
      color = random.choice(colors)
  
      # draw box
      tl = round(0.002 * max(result_image.shape[0:2])) + 1  # line thickness
      c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
      cv2.rectangle(result_image, c1, c2, color, thickness=tl)
      # draw text
      display_txt = "%s: %.1f%%" % (coco_names[label], 100*score)
      tf = max(tl - 1, 1)  # font thickness
      t_size = cv2.getTextSize(display_txt, 0, fontScale=tl / 3, thickness=tf)[0]
      c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
      cv2.rectangle(result_image, c1, c2, color, -1)  # filled
      cv2.putText(result_image, display_txt, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
  
  plt.figure(figsize=(10, 6))
  plt.imshow(result_image)
  plt.savefig('save_image.jpg')

def draw_masks(output):

  masks = None

  for score, mask in zip(output['scores'], output['masks']):
    if score > 0.5:
      if masks is None:
        masks = mask
      else:
        masks = torch.max(masks, mask)
  
  plt.figure(figsize=(10, 6))
  plt.imshow(masks.squeeze(0).cpu().numpy())
  plt.savefig('save_mask.jpg')

def make_mask_from_coco(annotation_path):
  from pycocotools.coco import COCO
  from PIL import Image
  
  #annotation_path = 'instances_default.json'
  coco = COCO(annotation_path)
  
  img_id = coco.getImgIds()[0]
  img = coco.loadImgs(1)[0]
  print(img)
  
  ann_ids = coco.getAnnIds(imgIds=[img['id']])
  anns = coco.loadAnns(ann_ids)
  
  mask = coco.annToMask(anns[0])
  for i in range(len(anns)):
      mask += coco.annToMask(anns[i])
  
  masking = Image.fromarray(mask * 255)
  #masking.show()
  masking.save('mask.png')

class CustomDataset(Dataset):
  def __init__(self, dir_path, transforms=None):
      ## initializing object attributes
      self.transforms = transforms
      self.dir_path = dir_path
      ## from dir_path
      ## added list of masks from the PedMasks directory
      self.mask_list = list(sorted(os.listdir(os.path.join(dir_path, "masks"))))
      ## added list of actual images from directory lists
      self.image_list = list(sorted(os.listdir(os.path.join(dir_path, "images"))))
  def __getitem__(self, idx):
      # get images and mask
      img_path = os.path.join(self.dir_path, "images", self.image_list[idx])
      mask_path = os.path.join(self.dir_path, "masks", self.mask_list[idx])
      image_obj = Image.open(img_path).convert("RGB")
      mask_obj = Image.open(mask_path)
      mask_obj = np.array(mask_obj)
      obj_ids = np.unique(mask_obj)
      # background has the first id so excluding that
      obj_ids = obj_ids[1:]
      # splitting mask into binaries
      masks_obj = mask_obj == obj_ids[:, None, None]
      # bounding box
      num_objs = len(obj_ids)
      bboxes = []
      for i in range(num_objs):
          pos = np.where(masks_obj[i])
          xmax = np.max(pos[1])
          xmin = np.min(pos[1])
          ymax = np.max(pos[0])
          ymin = np.min(pos[0])
          bboxes.append([xmin, ymin, xmax, ymax])
      image_id = torch.tensor([idx])
      masks_obj = torch.as_tensor(masks_obj, dtype=torch.uint8)
      bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
      labels = torch.ones((num_objs,), dtype=torch.int64)
      area = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
      iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
      target = {}
      target["image_id"] = image_id
      target["masks"] = masks_obj
      target["boxes"] = bboxes
      target["labels"] = labels
      target["area"] = area
      target["iscrowd"] = iscrowd
      if self.transforms is not None:
          image_obj, target = self.transforms(image_obj, target)
      return image_obj, target
  def __len__(self):
      return len(self.image_list)

def get_transform_data(train):
  import transforms as T
  transforms = []
  # PIL image to tensor for PyTorch model
  transforms.append(T.ToTensor())
  if train:
      # basic image augmentation techniques
      ## can add few more for experimentation
      transforms.append(T.RandomHorizontalFlip(0.5))
  return T.Compose(transforms)

def get_traininig_and_test_dataset(datapath):
  import utils

  # get the traffic data to transform
  train_dataset = CustomDataset(datapath, get_transform_data(train=True))
  test_dataset = CustomDataset(datapath, get_transform_data(train=False))
  
  # train test split
  torch.manual_seed(1)
  indices = torch.randperm(len(train_dataset)).tolist()
  train_dataset = torch.utils.data.Subset(train_dataset, indices[:-10])
  test_dataset = torch.utils.data.Subset(test_dataset, indices[-10:])
  
  # define training and validation data loaders
  train_data_loader = torch.utils.data.DataLoader(
      train_dataset, batch_size=2, shuffle=True, num_workers=4,
      collate_fn=utils.collate_fn)
  test_data_loader = torch.utils.data.DataLoader(
      test_dataset, batch_size=1, shuffle=False, num_workers=4,
      collate_fn=utils.collate_fn)

  return train_dataset, test_dataset, train_data_loader, test_data_loader

def modify_model(classes_num):
  # model already trained on COCO loaded from PyTorch repository
  maskrcnn_model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
  # number of input features identification
  in_features = maskrcnn_model.roi_heads.box_predictor.cls_score.in_features
  # head is changed
  maskrcnn_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, classes_num)
  in_features_mask = maskrcnn_model.roi_heads.mask_predictor.conv5_mask.in_channels
  hidden_layer = 256
  maskrcnn_model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, classes_num)
  return maskrcnn_model

def model_inference_on_testdata(model, test_dataset, device, batch_size=5):
  import utils

  test_data_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=True, num_workers=4,
    collate_fn=utils.collate_fn)

  images, targets = next(iter(test_data_loader))
  #for images, targets in train_data_loader: exit
  images = list(image.to(device) for image in images)
  targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

  prediction = model(images, targets)

  image_test = torchvision.utils.make_grid(images).permute(1, 2, 0).cpu()

  imgs = torch.cat([prediction[n]['masks'][0] for n in range(len(prediction))], dim=2)
  image_mask = torchvision.utils.make_grid(imgs).permute(1,2,0).cpu() 

  return image_test, image_mask
