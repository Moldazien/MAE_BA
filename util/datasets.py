# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import PIL

from torchvision import datasets, transforms
import torch

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

import numpy as np

def crop_fkt(bbox, img, percentile):  #eigene crop funktion um bilder auf bildausschnitte zu zuschneiden
  width, height = img.size
  width_edge = bbox[2]*percentile
  height_edge = bbox[3]*percentile
  #edge = (width_edge + height_edge)
  tx = max(bbox[0]-height_edge, 0)
  ty = max(bbox[1]-width_edge, 0)
  bx = min(bbox[0]+bbox[2]+height_edge, width)
  by = min(bbox[1]+bbox[3]+width_edge, height)

  h_box = bx - tx
  w_box = by - ty

  return (tx, ty, bx, by)

import os
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader

from pycocotools.coco import COCO
from PIL import Image

#falls es bei folgenden Datenklassen zu fehlern kommt, zuerst die datenklasse auskommentieren, und erneut testen

#datenklasse für iNat19. Wenn mit iNat19 gearbeitet wird
class iNaturalistDataset(Dataset):
  def __init__(self, annotations_file, dataset_dir, transform):
    self.annotations_file = annotations_file
    self.coco = COCO(annotations_file)  #for fast reading
    self.transform = transform
    self.dataset = dataset_dir
    self.img_ids = self.coco.getImgIds()

    def mapping(taxonomy):
      mapping = {}
      cat_ids = self.coco.getCatIds()
      categories = self.coco.loadCats(cat_ids)
      by_tax_cat = list(set([cat[taxonomy] for cat in categories]))
      by_tax_cat.sort()
      numb_cats = len(by_tax_cat)
      for i in range(numb_cats):
        for cat in categories:
          if cat[taxonomy] == by_tax_cat[i]:
            mapping[cat['id']] = i + 1
      return mapping

    self.class_mapping = mapping('name')  #normaly name but changed to genus for testing
    self.cat_len = len(self.class_mapping)

  def __len__(self):
    return len(self.img_ids)

  def __getitem__(self, idx):
    img_id = self.img_ids[idx]
    img = self.coco.loadImgs(img_id)
    ann_ids = self.coco.getAnnIds(img[0]['id'])
    anns = self.coco.loadAnns(ann_ids)
    img_path = self.dataset + '/images/' + img[0]['file_name']
    image = None
    bbox = None
    class_id = None
    with open(img_path, 'rb') as f:
        image = Image.open(f)
        image = image.convert('RGB')

    if len(anns) > 0:
      bbox = anns[0]['bbox']
      class_id = anns[0]['category_id']
    else:
      bbox = [0,0,img[0]['width'], img[0]['height']]
      class_id = -1

    percentile = 0.3
    box = crop_fkt(bbox, image, percentile)
    crop_img = image.crop(box=box)
    crop_img = self.transform(crop_img)
    ground_truth = self.class_mapping[class_id]
    ground_truth = torch.tensor(ground_truth)
    return crop_img, ground_truth


from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  #um immer ein bild zu laden. sonst stürzt ab, wenn ein bild nicht ganz korrekt aus google drive geladen wird, weil z.b. kurz verbindungsabbruch

#Datensatzklasse für channelisland camera traps dataset
class channelisland(Dataset):
  def __init__(self, annotations_file, dataset_dir, transform):
    self.annotations_file = annotations_file
    self.coco = COCO(annotations_file)  #for fast reading
    self.transform = transform
    
    self.dataset = dataset_dir
    self.img_ids = self.coco.getImgIds()
    self.ann_ids = self.coco.getAnnIds()

  def __len__(self):
    return len(self.img_ids)

  
  def __getitem__(self, idx):
    ann_id = self.ann_ids[idx]
    ann = self.coco.loadAnns([ann_id])
    img_id = ann[0]['image_id']
    img = self.coco.loadImgs([img_id])
    #print(str(ann_id) + ' ' + img[0]['file_name'])  #ausgabe um bei inferenz auslesen zu können, wo fehler auftraten bei klassifikation
    img_path = self.dataset + '/images/' + img[0]['file_name']
    image = None
    bbox = None
    class_id = None
    with open(img_path, 'rb') as f:
        image = Image.open(f)
        image = image.convert('RGB')

    bbox = ann[0]['bbox']
    class_id = ann[0]['category_id']
    percentile = 3 # 
    box = crop_fkt(bbox, image, percentile)
    crop_img = image.crop(box=box)
    crop_img = self.transform(crop_img)
    ground_truth = class_id
    ground_truth = torch.tensor(ground_truth)
    return crop_img, ground_truth


def build_dataset(is_train, args):  #funktion um korrekte datenklasse für mae zu erstellen
    transform = build_transform(is_train, args)
    """ #code für iNat19. wenn mit channelislands gearbeitet wird, muss das hier auskommentiert sein
    annPATH = ""
    dataset_path = '/content/dataset' 
    if is_train:
        annPATH = '/content/gdrive/MyDrive/Datasets/iNaturalist2019/annotations/meg_train.json'
    else:
        annPATH = '/content/gdrive/MyDrive/Datasets/iNaturalist2019/annotations/meg_test.json' # 

    dataset = iNaturalistDataset(annPATH, dataset_path, transform)
    """
    annPATH = ""  #code für channelislands camera traps dataset. muss auskommentiert werden, um mit channel islands camera traps dataset zu arbeitn
    dataset = None
    if is_train:
      dataset_path = '/content/dataset' 
      annPATH = 'PATH_TO_CHANNELISLANDS_TRAIN_ANNOTATIONS' #alter PATH nicht mehr gültig
      dataset = dataset = channelisland(annPATH, dataset_path, transform)
    else:
      dataset_path = 'PATH_TO_CHANNELISLANDS_TESTIMAGES'  #alter path nicht mehr gültig
      annPATH = 'PATH_TO_CHANNELISLANDS_TRAIN_ANNOTATIONS' #alter  PATH nicht mehr gültig
      dataset = channelisland(annPATH, dataset_path, transform)
    
    return dataset


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size= (224, 224),
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform  #weiteres zuschneiden unterdrücken. wenn
    t = []
    #if args.input_size <= 224:
    #    crop_pct = 224 / 256
    #else:
    crop_pct = 1.0
    size = args.input_size #int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
