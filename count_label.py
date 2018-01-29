import numpy as np

with open("/opt/intern/users/yuewang/dataset/Camelyon17/positive_list.txt", "r") as f:
    original = [x.strip().split(" ") for x in f.readlines()]
with open("/opt/intern/users/yuewang/dataset/Camelyon17/original_all_1.txt", "r") as f:
    original_1 = [x.strip().split(" ") for x in f.readlines()]

positive=0
negative=0
positive_1=0
negative_1=0
for list in original:
    #print list[0].split("_")
    #if list[0].split("_")[1] == "020" and list[0].split("_")[3] == "4":
        if list[1] == "1":
            positive+=1
        else:
            negative+=1
for list in original_1:
    #if list[0].split("_")[1] == "020" and list[0].split("_")[3] == "4":
        if list[1] == "1":
            positive_1+=1
        else:
            negative_1+=1

print positive, negative
print positive_1, negative_1
