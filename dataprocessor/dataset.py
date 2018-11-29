''' Document Localization using Recursive CNN
 Maintainer : Khurram Javed
 Email : kjaved@ualberta.ca '''

import csv
import logging
import os
from os import listdir
from os.path import isfile, join, isdir
import xml.etree.ElementTree as ET

import numpy as np
from torchvision import transforms

import utils.utils as utils
from tqdm import tqdm
import json
import cv2

# To incdude a new Dataset, inherit from Dataset and add all the Dataset specific parameters here.
# Goal : Remove any data specific parameters from the rest of the code

logger = logging.getLogger('iCARL')


class Dataset():
    '''
    Base class to reprenent a Dataset
    '''

    def __init__(self, name):
        self.name = name
        self.data = []
        self.labels = []


class SmartDoc(Dataset):
    '''
    Class to include MNIST specific details
    '''

    def __init__(self, directory="data"):
        super().__init__("smartdoc")
        self.data = []
        self.labels = []
        for d in directory:
            self.directory = d
            self.train_transform = transforms.Compose([transforms.Resize([32, 32]),
                                                       transforms.ColorJitter(1.5, 1.5, 0.9, 0.5),
                                                       transforms.ToTensor()])

            self.test_transform = transforms.Compose([transforms.Resize([32, 32]),
                                                      transforms.ToTensor()])

            logger.info("Pass train/test data paths here")

            self.classes_list = {}

            file_names = []
            print (self.directory, "gt.csv")
            with open(os.path.join(self.directory, "gt.csv"), 'r') as csvfile:
                spamreader = csv.reader(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                import ast
                for row in spamreader:
                    file_names.append(row[0])
                    self.data.append(os.path.join(self.directory, row[0]))
                    test = row[1].replace("array", "")
                    self.labels.append((ast.literal_eval(test)))
        self.labels = np.array(self.labels)

        self.labels = np.reshape(self.labels, (-1, 8))
        logger.debug("Ground Truth Shape: %s", str(self.labels.shape))
        logger.debug("Data shape %s", str(len(self.data)))

        self.myData = [self.data, self.labels]


class SmartDocDirectories(Dataset):
    '''
    Class to include MNIST specific details
    '''

    def __init__(self, directory="data"):
        super().__init__("smartdoc")
        self.data = []
        self.labels = []

        for folder in os.listdir(directory):
            if (os.path.isdir(directory + "/" + folder)):
                for file in os.listdir(directory + "/" + folder):
                    images_dir = directory + "/" + folder + "/" + file
                    if (os.path.isdir(images_dir)):

                        list_gt = []
                        tree = ET.parse(images_dir + "/" + file + ".gt")
                        root = tree.getroot()
                        for a in root.iter("frame"):
                            list_gt.append(a)

                        im_no = 0
                        for image in os.listdir(images_dir):
                            if image.endswith(".jpg"):
                                # print(im_no)
                                im_no += 1

                                # Now we have opened the file and GT. Write code to create multiple files and scale gt
                                list_of_points = {}

                                # img = cv2.imread(images_dir + "/" + image)
                                self.data.append(os.path.join(images_dir, image))

                                for point in list_gt[int(float(image[0:-4])) - 1].iter("point"):
                                    myDict = point.attrib

                                    list_of_points[myDict["name"]] = (
                                        int(float(myDict['x'])), int(float(myDict['y'])))

                                ground_truth = np.asarray(
                                    (list_of_points["tl"], list_of_points["tr"], list_of_points["br"],
                                     list_of_points["bl"]))
                                ground_truth = utils.sort_gt(ground_truth)
                                self.labels.append(ground_truth)

        self.labels = np.array(self.labels)

        self.labels = np.reshape(self.labels, (-1, 8))
        logger.debug("Ground Truth Shape: %s", str(self.labels.shape))
        logger.debug("Data shape %s", str(len(self.data)))

        self.myData = []
        for a in range(len(self.data)):
            self.myData.append([self.data[a], self.labels[a]])

class SelfCollectedDataset(Dataset):
    '''
    Class to include MNIST specific details
    '''

    def __init__(self, directory="data"):
        super().__init__("smartdoc")
        self.data = []
        self.labels = []

        files = [f for f in listdir(directory) if isfile(join(directory, f))]
        for file in tqdm(files):
            name = file.split('.')
            if(name[-1] != 'json'):
                if((''.join(name[:-1]) + '.json') in files):
                    img_path = os.path.join(directory, file)
                    json_path = os.path.join(directory, ''.join(name[:-1])+'.json')

                    f = open(json_path, 'r')
                    body = json.load(f)
                    gt = []
                    points = body['shapes'][0]['points']
                    for point in points:
                        gt.append(point)
                    # #main
                    # INTER_MARGIN_RATIO = 0.2
                    # EXTER_INTER_RATIO = 1
                    # img = cv2.imread(img_path)
                    # # Calculate shape of the bounding box of card
                    # x,y,w,h = cv2.boundingRect(np.array(points))
                    # inter_margins = INTER_MARGIN_RATIO * np.array([w,h])

                    # # Shape of image
                    # x_max = img.shape[1]
                    # y_max = img.shape[0]

                    # # create a array of x,y
                    # points_x = []
                    # points_y = []
                    # for point in points:
                    #     #point[0] ~ x, point[1] ~ y
                    #     points_x.append(point[0])
                    #     points_y.append(point[1])
                    # points_x = np.array(points_x)
                    # points_y = np.array(points_y)

                    # # computer margin for each point
                    # expand_direct_x = find_expand_direct(points_x)
                    # expand_direct_y = find_expand_direct(points_y)
                    # margins_x = inter_margins[0]*expand_direct_x
                    # margins_y = inter_margins[1]*expand_direct_y

                    # # expand to 4 boxes
                    # boxes = []
                    # for i in range(0,4):
                    #     box = expand_to_box(points_x[i], points_y[i], x_max, y_max, margins_x[i], margins_y[i], EXTER_INTER_RATIO)
                    #     # boxes.append(box)
                    #     # gt.append(utils.sort_gt(box))
                    #     gt = np.array(box).astype(np.float32)
                    #     ground_truth = utils.sort_gt(gt)
                    #     self.labels.append(ground_truth)
                    #     self.data.append(img_path)
                    gt = np.array(gt).astype(np.float32)
                    ground_truth = utils.sort_gt(gt)
                    self.labels.append(ground_truth)
                    self.data.append(img_path)
              
        # for image in os.listdir(directory):
        #     # print (image)
        #     if image.endswith("jpg") or image.endswith("JPG"):
        #         if os.path.isfile(os.path.join(directory, image + ".csv")):
        #             with open(os.path.join(directory, image + ".csv"), 'r') as csvfile:
        #                 spamwriter = csv.reader(csvfile, delimiter=' ',
        #                                         quotechar='|', quoting=csv.QUOTE_MINIMAL)

        #                 img_path = os.path.join(directory, image)

        #                 gt = []
        #                 for row in spamwriter:
        #                     gt.append(row)
        #                 gt = np.array(gt).astype(np.float32)
        #                 ground_truth = utils.sort_gt(gt)
        #                 self.labels.append(ground_truth)
        #                 self.data.append(img_path)
        self.labels = np.array(self.labels)

        self.labels = np.reshape(self.labels, (-1, 8))

        print("Ground Truth Shape: %s", str(self.labels.shape))
        print("Data shape %s", str(len(self.data)))
        logger.debug("Ground Truth Shape: %s", str(self.labels.shape))
        logger.debug("Data shape %s", str(len(self.data)))

        self.myData = []
        for a in range(len(self.data)):
            self.myData.append([self.data[a], self.labels[a]])


class SmartDocCorner(Dataset):
    '''
    Class to include MNIST specific details
    '''

    def __init__(self, directory="data"):
        super().__init__("smartdoc")
        self.data = []
        self.labels = []
        for d in directory:
            self.directory = d
            self.train_transform = transforms.Compose([transforms.Resize([32, 32]),
                                                       transforms.ColorJitter(0.5, 0.5, 0.5, 0.5),
                                                       transforms.ToTensor()])

            self.test_transform = transforms.Compose([transforms.Resize([32, 32]),
                                                      transforms.ToTensor()])

            logger.info("Pass train/test data paths here")

            self.classes_list = {}

            file_names = []
            with open(os.path.join(self.directory, "gt.csv"), 'r') as csvfile:
                spamreader = csv.reader(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                import ast
                for row in spamreader:
                    file_names.append(row[1])
                    self.data.append(os.path.join(self.directory, row[1]))
                    test = row[2].replace("array", "")
                    # test = test.replace("|","")
                    self.labels.append((ast.literal_eval(test)))
        self.labels = np.array(self.labels)

        self.labels = np.reshape(self.labels, (-1, 2))
        logger.debug("Ground Truth Shape: %s", str(self.labels.shape))
        logger.debug("Data shape %s", str(len(self.data)))

        self.myData = [self.data, self.labels]

#check if box out range of image
def valid(x , x_min, x_max):
    if(x < x_min):
        return x_min
    elif(x > x_max):
        return x_max
    else:
        return x

# crate a array with the vertical/horizal direction for expandation of points
# 1 ~ x/y increase if point move to center
# -1 ~ x/y decrease if point move to center
def find_expand_direct(points_x):
    expand_direct = np.ones((4))
    for i, x in enumerate(points_x):
        if(sum(x > points_x) > 1):
            expand_direct[i] = -1
    return expand_direct

#expand to box, ex_in_r = external_area/internal_area for a box
def expand_to_box(x, y, x_max, y_max, margin_x, margin_y, ex_in_r):
    x1 = valid(int(x - margin_x * ex_in_r), 0, x_max)
    y1 = valid(int(y - margin_y * ex_in_r), 0, y_max)
    x2 = valid(int(x + margin_x), 0, x_max)
    y2 = valid(int(y + margin_y), 0, y_max)
    return [[x1,y1], [x1,y2], [x2,y2], [x2,y1]]

