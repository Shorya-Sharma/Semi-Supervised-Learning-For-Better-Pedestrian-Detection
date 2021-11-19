import json
file_list = []
file_content_list = []
with open('FLIR_ADAS_1_3/train/thermal_annotations.json', mode='r', encoding='gbk') as f2:
    mdata = json.load(f2)
    for i in mdata['images']:
        file_list.append(i['file_name'][14:-5]+".txt")
        file_content_list.append("")
    for b in mdata['annotations']:
        if b['category_id'] == 1:
            id = b['image_id']
            tmp = "0 "
            for j in range(len(b['bbox'])):
                if j == 1:
                    tmp += str(float(b['bbox'][j]/512)) + " "
                elif j == 3:
                    tmp += str(float(b['bbox'][j]/512)) +"\n"
                else:
                    tmp += str(float(b['bbox'][j]/640)) + " "
            file_content_list[id] += tmp


train_list_map = []
for n in range(len(file_list)):
    if file_content_list[n] != "":
        with open("data/custom/labels/"+file_list[n],mode='w',encoding='gbk') as w2:
            w2.write(file_content_list[n])
            train_list_map.append("data/custom/images/"+file_list[n][:-3] +"jpg")


file_list = []
file_content_list = []
with open('FLIR_ADAS_1_3/val/thermal_annotations.json', mode='r', encoding='gbk') as f2:
    mdata = json.load(f2)
    for i in mdata['images']:
        file_list.append(i['file_name'][14:-5]+".txt")
        file_content_list.append("")
    for b in mdata['annotations']:
        if b['category_id'] == 1:
            id = b['image_id']
            tmp = "0 "
            for j in range(len(b['bbox'])):
                if j == 1:
                    tmp += str(float(b['bbox'][j]/512)) + " "
                elif j == 3:
                    tmp += str(float(b['bbox'][j]/512)) +"\n"
                else:
                    tmp += str(float(b['bbox'][j]/640)) + " "
            file_content_list[id] += tmp

print("\n\n")
valid_list_map = []
for n in range(len(file_list)):
    if file_content_list[n] != "":
        with open("data/custom/labels/"+file_list[n],mode='w',encoding='gbk') as w2:
            w2.write(file_content_list[n])
            valid_list_map.append("data/custom/images/"+file_list[n][:-3] +"jpg")


with open("data/custom/train.txt",mode='w',encoding='gbk') as f:
    for i in train_list_map:
        f.write(i + "\n")

with open("data/custom/valid.txt",mode='w',encoding='gbk') as f:
    for i in valid_list_map:
        f.write(i + "\n")