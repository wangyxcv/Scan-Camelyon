import cv2
from matplotlib import pyplot as plt
import numpy as np
import csv
import os
import scipy.ndimage as ndimage

def find_boxes_1(position):
    boxes = []
    begin = 0
    for i in range(1,len(position[0])):
        if abs(position[0][i]-position[0][i-1]) > 10 or i == len(position[0])-1:
            box = [0,0,0,0]
            box[1] = np.min(position[0][begin:i-1])
            box[0] = np.min(position[1][begin:i-1])
            box[3] = np.max(position[0][begin:i-1])
            box[2] = np.max(position[1][begin:i-1])
            boxes.append(box)
            begin = i
    return boxes

def correct_box(img, x1, x2, y1, y2):
    for i in range(x1, x2):
        if img[i, y1-1] > 0:
            j = y1-2
            while img[i,j] > 0 and j>=0:
                j = j-1
            return correct_box(img, x1, x2, j+1, y2)
        if img[i, y2] > 0:
            j = y2+1
            while img[i,j]>0 and j<img.shape[1]:
                j = j+1
            return correct_box(img, x1, x2, y1, j)
    for j in range(y1, y2):
        if img[x1-1, j] >0:
            i = x1-2
            while img[i,j] >0 and i>=0:
                i = i-1
            return correct_box(img, i+1, x2, y1, y2)
        if img[x2, j] >0:
            i = x2+1
            while img[i,j] >0 and i<img.shape[0]:
                i = i+1
            return correct_box(img, x1, i, y1, y2)
    return (x1,x2,y1,y2)

def find_boxes(img):
    boxes = []
    assert len(img.shape)==2
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i,j]>0:
                x1 = i
                y1 = j
                for jj in range(j+1,img.shape[1]):
                    if img[i,jj]==0:
                        y2 = jj
                        break
                for ii in range(i+1, img.shape[0]):
                    if img[ii,j]==0:
                        x2 = ii
                        break
                #print (x1, x2, y1, y2)
                box = correct_box(img, x1, x2, y1, y2)
                img[box[0]:box[1], box[2]:box[3]] = 0
                boxes.append(box)
    return boxes

                
file_dict = {}
with open("/opt/intern/users/yuewang/dataset/Camelyon17/stage_labels.csv") as file:
    csv_read = csv.reader(file)
    for x in csv_read:
        file_dict[x[0]] = x[1]

path = "/opt/intern/users/yuewang/ScanNet-FCN/output/prediction/random_list/"

#out_file = open("label_and_mask_close.txt", "w")
csvfile = open("predict_train_area.csv", "wb")
csvwriter = csv.writer(csvfile, dialect='excel')
csvwriter.writerow(["patient", "stage"])
files = os.listdir(path)
files.sort()
file_count = 0
patient_list = []
macro = 0
micro = 0
itc = 0
negative = 0
for file in files:
#for file in ["mask_patient_039_node_1.jpg"]:
    if os.path.isdir(path+file):
        continue
    print file
    node_list = []
    node_list.append(file.split(".")[0]+".tif")
    img = cv2.imread(path+file)
    img[img<180] = 0
    image_open = ndimage.binary_opening(img[:,:,0], structure=np.ones((5,5)))
    image_close = ndimage.binary_closing(image_open, structure=np.ones((5,5)))
    image_ = np.where(image_close==True, 255, 0)
    boxes = find_boxes(image_close)   #(x1, y1, x2, y2)
    #length = [max(box[1]-box[0], box[3]-box[2]) for box in boxes]
    length = [(box[1]-box[0])*(box[3]-box[2]) for box in boxes]
    length.sort()
    print length
    gt = file_dict[file.split(".")[0]+".tif"]
    if len(length) == 0:
        label = "negative"
        negative += 1
    elif length[-1]>=200 or (len(length)>50 and length[-5]>=30):
        label = "macro"
        macro += 1
    elif length[-1]>=50 or (len(length)>1 and length[-2]>=30):
        label = "micro"
        micro += 1
    elif len(length)>1 and length[-2]>=20:
        label = "itc"
        itc += 1
    else:
        label = "negative"
        negative += 1
    node_list.append(label)
    node_list.append(gt)
    node_list.append(str(len(length)))
    node_list.append(str(length))
    #print label
    patient_list.append(node_list)
    file_count += 1
    if file_count%5 == 0:
        if macro > 0:
            if macro+micro>3:
                patient_label = "pN2"
            else:
                patient_label = "pN1"
        elif micro > 0:
            patient_label = "pN1mi"
        elif itc > 0:
            patient_label = "pN0(i+)"
        else:
            patient_label = "pN0"
        
        csvwriter.writerow([file.split("_node")[0]+".zip", file_dict[file.split("_node")[0]+".zip"], patient_label])
        for node in patient_list:
            csvwriter.writerow(node)
        patient_list = []
        macro = 0
        micro = 0
        itc = 0
        negative = 0
#csvwriter.close()
    #gt = cv2.imread(path+"mask_patient_039_node_1.jpg")
    #out_file.write(file+" "+label.ljust(8)+" "+str(len(length))+" "+str(length)+"\n")
    '''
    plt.subplot(1,3,1)
    plt.imshow(img)
    plt.subplot(1,3,2)
    plt.imshow(image_, cmap="gray")
    plt.subplot(1,3,3)
    plt.imshow(gt)
    plt.show()
    '''
csvfile.close()
