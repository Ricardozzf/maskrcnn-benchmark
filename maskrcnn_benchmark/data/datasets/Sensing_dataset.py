from maskrcnn_benchmark.structures.bounding_box import BoxList
import os
from PIL import Image
import torch

class SensingDataset(object):
    def __init__(self, annPath, imagePath, remove_annotations_without_images, transforms= None):
        # as you would do normally
        self.anns = []
        for root, _, files in os.walk(annPath):
            for filename in files:
                if not filename.endswith('.txt'):
                    continue
                self.anns.append(os.path.join(root, filename))
        self.images = []
        for root, _, files in os.walk(imagePath):
            for filename in files:
                if not filename.endswith('.jpg'):
                    continue
                self.images.append(os.path.join(root, filename))

        if remove_annotations_without_images:
            self.anns = [
                annfile
                for annfile in self.anns
                if(os.path.exists(os.path.join(annPath, annfile.replace('.txt', '.jpg'))))
            ]
            self.images = [
                imagefile
                for imagefile in self.images
                if(os.path.exists(os.path.join(imagePath, imagefile.replace('.jpg', '.txt'))))
            ]
        self.transforms = transforms
        
        self.classes = ['background', 'person']

    def __getitem__(self, idx):
        # load the image as a PIL Image
        txtPath = self.anns[idx]
        imPath = txtPath.replace('labels', 'images').replace('.txt', '.jpg')
        image = Image.open(imPath).convert("RGB")
        
        # load the bounding boxes as a list of list of boxes
        # in this case, for illustrative purposes, we use
        # x1, y1, x2, y2 order.
        labels = []
        boxes = []
        with open(txtPath) as fp:
            for line in fp.readlines():
                cnt = line.rstrip().split(' ')
                labels.append(int(cnt[0])+1)
                x1 = float(cnt[1])
                y1 = float(cnt[2])
                x2 = float(cnt[3])
                y2 = float(cnt[4])
                boxes.append([x1,y1,x2,y2])
        
        # and labels
        labels = torch.tensor(labels)

        # create a BoxList from the boxes
        boxlist = BoxList(boxes, image.size, mode="xyxy")
        # add the labels to the boxlist
        boxlist.add_field("labels", labels)

        if self.transforms:
            image, boxlist = self.transforms(image, boxlist)

        # return the image, the boxlist and the idx in your dataset
        return image, boxlist, idx

    def __len__(self):
        return len(self.lines)

    def get_img_info(self, idx):
        # get img_height and img_width. This is used if
        # we want to split the batches according to the aspect ratio
        # of the image, as it can be more efficient than loading the
        # image from disk
        image = Image.open(self.lines[idx])
        img_width,img_height = image.size

        return {"height": img_height, "width": img_width}
    
    def get_groundtruth(self, idx):
        line = self.lines[idx]
        imPath = line.rstrip()
        txtPath = imPath.replace('images', 'labels').replace('.jpg', '.txt')

        image = Image.open(imPath).convert("RGB")
        
        # load the bounding boxes as a list of list of boxes
        # in this case, for illustrative purposes, we use
        # x1, y1, x2, y2 order.
        labels = []
        boxes = []
        with open(txtPath) as fp:
            for line in fp.readlines():
                cnt = line.rstrip().split(' ')
                labels.append(int(cnt[0])+1)
                x1 = float(cnt[1])
                y1 = float(cnt[2])
                x2 = float(cnt[3])
                y2 = float(cnt[4])
                boxes.append([x1,y1,x2,y2])
        
        # and labels
        labels = torch.tensor(labels)

        # create a BoxList from the boxes
        target = BoxList(boxes, image.size, mode="xyxy")
        # add the labels to the boxlist
        target.add_field("labels", labels)

        return target

    def map_class_id_to_class_name(self, class_id):
        return self.classes[class_id]