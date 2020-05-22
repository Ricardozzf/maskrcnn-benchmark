# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import argparse
import cv2

from maskrcnn_benchmark.config import cfg
from predictor import COCODemo

import time,os
from pycocotools.coco import COCO
import torch
from maskrcnn_benchmark.structures.bounding_box import BoxList
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Webcam Demo")
    parser.add_argument(
        "--config-file",
        default="../configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for the prediction to be shown",
    )
    parser.add_argument(
        "--min-image-size",
        type=int,
        default=800,
        help="Smallest size of the image to feed to the model. "
            "Model was trained with 800, which gives best results",
    )
    parser.add_argument(
        "--show-mask-heatmaps",
        dest="show_mask_heatmaps",
        help="Show a heatmap probability for the top masks-per-dim masks",
        action="store_true",
    )
    parser.add_argument(
        "--masks-per-dim",
        type=int,
        default=2,
        help="Number of heatmaps per dimension to show",
    )
    parser.add_argument(
        "opts",
        help="Modify model config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--img-dir",
        type=str,
        default="/home/zouzhaofan/Dataset/PublicData/face++/val",
    )
    parser.add_argument(
        "--valjson",
        type=str,
        default='/home/zouzhaofan/Dataset/PublicData/face++/val_fxywh.json',
    )

    args = parser.parse_args()

    # load config from file and command-line arguments
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    path = args.img_dir

    # prepare object that handles inference plus adds predictions on top of image
    coco_demo = COCODemo(
        cfg,
        confidence_threshold=args.confidence_threshold,
        show_mask_heatmaps=args.show_mask_heatmaps,
        masks_per_dim=args.masks_per_dim,
        min_image_size=args.min_image_size,
        weight_loading="/home/zouzhaofan/Work/Github/maskrcnn-benchmark/retina-person-fwh/model_final.pth"
    )

    #cam = cv2.VideoCapture(0)
    #while True:
    coco_val = COCO(args.valjson)
    cv2.namedWindow("COCO detections",0)
    err = []
    for (key, value) in coco_val.imgs.items():
        start_time = time.time()
        #ret_val, img = cam.read()
        img = cv2.imread(os.path.join(path, value["file_name"]))
        annIds = coco_val.getAnnIds(imgIds=key)
        anns = coco_val.loadAnns(annIds)
        targets = []
        for ann in anns:
            targets.append(torch.tensor(ann["bbox"]))

        targets = torch.cat(targets, 0)
        targets = targets.view(-1,8)
        img_size = (value['width'], value['height'])
        
        targets_boxlist = BoxList(targets[:,:4], img_size, mode="xywh")
        targets_boxlist.vwvh = (targets[:, 4:]).to(torch.float32)
        
        composite = coco_demo.run_on_opencv_image(img, targets_boxlist, err)
        
        print("Time: {:.2f} s / img".format(time.time() - start_time))
        '''
        cv2.imshow("COCO detections", composite)
        if cv2.waitKey(0) == 27:
            break  # esc to quit
        '''
    
    err = np.array(err)
    np.savetxt("err.txt", err)
    #print("err_w:{}  err_h:{}".format(np.mean(np.abs(err_w)), np.mean(np.abs(err_h))))
    cv2.destroyAllWindows() 


if __name__ == "__main__":
    main()
