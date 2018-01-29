import numpy as np
import random

def random_label(l, out_file, no_count=None):
    lenght = len(list)
    #l = list.copy()
    for i in range(lenght):
        idx = random.randint(0, lenght-i-1)
        if no_count is not None:
            a = l[idx].split("_")
            file_name = a[0]+"_"+a[1]+"_"+a[2]+"_"+a[3]
            #print file_name
            if file_name in no_count:
                del l[idx]
                continue

        out_file.write(l[idx]+"\n")
        del l[idx]

def eval_label(original, positive, train_list):
    positive_list = []
    negative_list = []

    for list in original:
        if list[1] == "1":
            positive_list.append(list[0])
        else:
            negative_list.append(list[0])

    for list in positive:
        if list[1] == "1":
            positive_list.append(list[0])
        else:
            negative_list.append(list[0])

    n_index = 0
    p_index = 0
    index = 0

    print "positive:", len(positive_list)
    print "negative:", len(negative_list)
    while n_index < len(negative_list):
        a = random.randint(1,2)
        if a == 1:
            if p_index == len(positive_list):
                p_index = 0
                print "positive has finish"
            train_list.write(positive_list[p_index]+" 1"+"\n")
            p_index += 1
        else:
            train_list.write(negative_list[n_index]+" 0"+"\n")
            n_index += 1
        index += 1


with open("/opt/intern/users/yuewang/dataset/Camelyon17/train_list.txt", "r") as f:
    list = [x.strip() for x in f.readlines()]
with open("/opt/intern/users/yuewang/dataset/Camelyon17/positive_all.txt", "r") as f:
    for x in f.readlines():
        list.append(x.strip())
out = open("/opt/intern/users/yuewang/dataset/Camelyon17/random_train_list.txt", "w")

no_count = ["patient_017_node_4", "patient_039_node_1", "patient_046_node_4", "patient_064_node_0", "patient_089_node_3"]
random_label(list, out, no_count)

out.close()

#with open("/opt/intern/users/yuewang/dataset/Camelyon17/original_all_1.txt", "r") as f:
#    original = [x.strip().split(" ") for x in f.readlines()]
#with open("/opt/intern/users/yuewang/dataset/Camelyon17/positive_list.txt", "r") as f:
#    positive = [x.strip().split(" ") for x in f.readlines()]
