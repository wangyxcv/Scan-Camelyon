from data_layer import image_rotate
import cv2

img = cv2.imread("test_2000.jpg")
count = 0
while True:
    count += 1
    print count
    img_rotate = image_rotate(img, 100, 500)
