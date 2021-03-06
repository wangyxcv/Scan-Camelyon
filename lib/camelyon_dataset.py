import sys
import os
import random

import cv2
import numpy as np
import xml.etree.ElementTree as ET

sys.path.append('/usr/local/bin/')
import multiresolutionimageinterface as mir

class MaskAnnotation:
    def __init__(self, xml_file):
        self.parse_xml(xml_file)
    
    def parse_xml(self, xml_file):
        etree = ET.parse(xml_file)
        root = etree.getroot()
        annotations, annotation_groups = root.getchildren()
        self.contours = []
        for annotation in annotations.getchildren():
            attrib, contour = self.parse_annotation(annotation)
            if attrib['PartOfGroup'] != 'metastases':
                continue
            self.contours.append((attrib, contour))
        self.groups = self.parse_group(annotation_groups)

    def parse_annotation(self, annotation):
        assert(annotation.tag == 'Annotation')
        coordinates = annotation.find('Coordinates')
        contour = []
        for coordinate in coordinates.getchildren():
            attrib = coordinate.attrib
            order = int(attrib['Order'])
            x = float(attrib['X'])
            y = float(attrib['Y'])
            contour.append((order, x, y))
        contour = np.array([(item[1], item[2])  for item in sorted(contour)])
        attrib = annotation.attrib
        return attrib, contour

    def parse_group(self, annotation_groups):
        groups = []
        for group in annotation_groups.getchildren():
            attrib = group.attrib
            groups.append(attrib)
        return groups

    def is_inside(self, rect, contour):
        # intersection
        x1 = max(int(np.min(contour[:,0])+0.5), rect[0])
        y1 = max(int(np.min(contour[:,1])+0.5), rect[1])
        x2 = min(int(np.max(contour[:,0])+0.5), rect[2])
        y2 = min(int(np.max(contour[:,1])+0.5), rect[3])
        if x1>x2 or y1>y2:
            return False # non-overlap
        return True

    def get_mask(self, rect, fill=1):
        rect_x1,rect_y1,rect_x2,rect_y2 = rect
        rect_w = rect_x2-rect_x1+1
        rect_h = rect_y2-rect_y1+1
        mask = np.zeros((rect_h, rect_w), np.uint8)
        pts = []
        for attrib, contour in self.contours:
            if self.is_inside(rect, contour) is False:
                continue
            contour = contour.copy()
            contour[:, 0] = np.minimum(np.maximum(contour[:,0]-rect_x1+0.5, 0), rect_w-1)
            contour[:, 1] = np.minimum(np.maximum(contour[:,1]-rect_y1+0.5, 0), rect_h-1)
            contour = contour.astype(np.int32)
            pts.append(contour)
        cv2.fillPoly(mask, pts, fill)
        return mask

    # for verification
    def visulize(self, mr_image):
        for attrib, contour in self.contours:
            print(contour)
            x1 = int(np.min(contour[:,0])-100)
            y1 = int(np.min(contour[:,1])-100)
            x2 = int(np.max(contour[:,0])+100)
            y2 = int(np.max(contour[:,1])+100)
            image_patch = mr_image.getUCharPatch(x1, y1, x2-x1+1, y2-y1+1, 0)
            num_points = contour.shape[0]
            for i in range(num_points):
                pts1_x,pts1_y = contour[i]
                pts2_x,pts2_y = contour[(i+1)%num_points]
                pts1_x = int(pts1_x-x1)
                pts1_y = int(pts1_y-y1)
                pts2_x = int(pts2_x-x1)
                pts2_y = int(pts2_y-y1)
                cv2.line(image_patch, (pts1_x,pts1_y), (pts2_x,pts2_y), (0,255,0), 2)
            mask = self.get_mask((x1,y1,x2,y2), 255)
            show_image(mask, 'mask')
            show_image(image_patch, 'image_patch')
            cv2.waitKey()

def show_image(im, name='im', max_size=800):
    h = im.shape[0]
    w = im.shape[1]
    if max(h,w) > max_size:
        scale = 1.0*max_size/max(h,w)
        im = cv2.resize(im, None, fx=scale, fy=scale)
    cv2.imshow(name, im)

class MRImageSampler:
    '''  
        MRImageSampler:
            sample image patch from single MRImage onfly
            [porperty]
            sample_size: return patch size
            fg_ratio: ratio of foreground patch or sample uniformly if fg_ratio < 0
            skip_empty: skip all background
    '''
    def __init__(self, tif_file, xml_file, sample_size=224,
            fg_ratio=-1, skip_empty=True):
        assert(fg_ratio<1.0)
        self.sample_size = sample_size
        self.fg_ratio = fg_ratio
        self.skip_empty = skip_empty
        self.tif_file = tif_file
        self.xml_file = xml_file
        # load MRImage
        reader = mir.MultiResolutionImageReader()
        self.mr_image = reader.open(tif_file)
        self.max_size = self.mr_image.getDimensions()
        # load annotation
        self.anno = MaskAnnotation(xml_file)
        rect = []
        for attrib, contour in self.anno.contours:
            x1 = int(np.min(contour[:,0])+0.5)
            y1 = int(np.min(contour[:,1])+0.5)
            x2 = int(np.max(contour[:,0])+0.5)
            y2 = int(np.max(contour[:,1])+0.5)
            #print('fg rect', (x1,y1,x2,y2), attrib)
            rect.append((x1,y1,x2,y2))
        self.rect = rect
        self.image_id = int(self.tif_file.split("/")[-1].split("_")[1])/20
        if self.image_id ==4:
            level = 7
        else:
            level = 5
        self.ds, self.thr_image = self.binary_img()
    
    def binary_img(self):
        if self.image_id ==4:
            level = 7
        else:
            level = 5

        ds = self.mr_image.getLevelDownsample(level)
        shape = self.mr_image.getLevelDimensions(level)
        image = self.mr_image.getUCharPatch(0,0,shape[0], shape[1], level)
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        if self.image_id == 4:
            gray_image[gray_image>200] = 255
        else :
            gray_image[gray_image==0] = 255
        blur = cv2.GaussianBlur(gray_image,(5,5),0)
        ret1,binary_image = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        return ds, binary_image

    def sample_rect(self):
        if self.fg_ratio < 0 or random.random()>self.fg_ratio:
            # fg region is far less than bg region, so it can be ignored
            x = random.randint(0, self.max_size[0]-self.sample_size-1)
            y = random.randint(0, self.max_size[1]-self.sample_size-1)
        else:
            # sample region overlap with fg and fg's center inside sample region
            rect = random.choice(self.rect)
            cx = int((rect[2]+rect[0])/2.0)
            cy = int((rect[3]+rect[1])/2.0)
            x = random.randint(cx-self.sample_size, cx)
            y = random.randint(cy-self.sample_size, cy) 
            x = min(max(x,0), self.max_size[0]-self.sample_size-1)
            y = min(max(y,0), self.max_size[1]-self.sample_size-1)
        return (x, y, x+self.sample_size-1, y+self.sample_size-1)

    def is_empty(self, image_patch):
        #check = image_patch == image_patch[0,0,:]
        return len(np.unique(image_patch))<30
        
    def is_error(self, image_patch):
        return self.image_id == 4 and np.min(image_patch) == 255

    def is_background(self, rect):
        x1 = rect[0]/self.ds
        y1 = rect[1]/self.ds
        x2 = rect[2]/self.ds
        y2 = rect[3]/self.ds
        return 0 not in np.unique(self.thr_image[int(y1):int(y2+0.5), int(x1):int(x2+0.5)])

    def next(self):
        count_try = 0
        while True:
            #count_try += 1
            rect = self.sample_rect()
            result = self.sub_region(rect)
            if result is None:
                continue
            return result

    def sub_region(self, rect):
        x1,y1,x2,y2 = rect
        #if self.skip_empty and self.is_background(rect):
        #    print rect, "is background"
        #    return None
        if self.skip_empty and self.is_background(rect):
            return None
        image_patch = self.mr_image.getUCharPatch(x1, y1, x2-x1+1, y2-y1+1, 0)

        if self.is_error(image_patch):
             reader = mir.MultiResolutionImageReader()
             self.mr_image = reader.open(self.tif_file)
        
        if self.skip_empty and self.is_empty(image_patch):
            return None
        mask = self.anno.get_mask((x1,y1,x2,y2), 255)
        return image_patch, mask

class CamelyonDataset:
    def __init__(self, tif_files, xml_files, sample_size,
            bg_ratio, skip_empty=True):
        self.skip_empty = skip_empty
        self.samplers = []
        for tif_file, xml_file in zip(tif_files, xml_files):
            sampler = MRImageSampler(tif_file, xml_file, sample_size,
                    bg_ratio, skip_empty)
            self.samplers.append(sampler)

    def next(self):
        sampler = random.choice(self.samplers)
        return sampler.next()

if __name__ == '__main__':
    '''
    reader = mir.MultiResolutionImageReader()
    mr_image = reader.open(sys.argv[1])
    anno = MaskAnnotation(sys.argv[2])
    anno.visulize(mr_image)
    '''
    if False:
        # signle MRImage
        sampler = MRImageSampler(sys.argv[1], sys.argv[2], 512, 0.1)
    else:
        # multi MRImage
        tif_root = '/opt/intern/users/yuewang/dataset/Camelyon17/images/'
        tif_files = [ os.path.join(tif_root, x) for x in sorted(os.listdir(tif_root))]
        xml_root = '/opt/intern/users/yuewang/dataset/Camelyon17/label/'
        xml_files = [ os.path.join(xml_root, x) for x in sorted(os.listdir(xml_root))]
        sampler = CamelyonDataset(tif_files, xml_files, 244, 0.1)

    while True:
        image_patch, mask = sampler.next()
        show_image(image_patch, 'image_patch')
        show_image(mask, 'mask')
        #print image_patch.shape, mask.shape
        cv2.waitKey(2000)
