import os

'''ilab'''
id_dict = {}
bg_dict = {}
pose_dict = {}
id_cnt = 0
bg_cnt = 0
pose_cnt = 0
for roots, dirs, files in os.walk('/home2/ilab2M_pose/train_img_c00_10class'):
    for file in files:
        category = file.split('-')[0]
        id = file.split('-')[1]
        background = category + file.split('-')[2]
        pose = file.split('-')[3] + file.split('-')[4]
        if id not in id_dict:
            id_dict[id] = id_cnt
            id_cnt += 1
        if background not in bg_dict:
            bg_dict[background] = bg_cnt
            bg_cnt += 1
        if pose not in pose_dict:
            pose_dict[pose] = pose_cnt
            pose_cnt += 1
        #print(file)
'''fonts'''
letter_dict = {}
size_dict = {}
fg_dict = {}
bg_dict1 = {}
style_dict = {}
letter_cnt = 0
size_cnt = 0
fg_cnt = 0
bg_cnt1 = 0
style_cnt = 0
for roots, dirs, files in os.walk('/home2/fonts_dataset_center'):
    for file in files:
        letter = file.split('_')[0]
        size = file.split('_')[1]
        fg = file.split('_')[2]
        bg = file.split('_')[3]
        style = file.split('_')[4].split('.')[0]
        #print(file)
        if letter not in letter_dict:
            letter_dict[letter] = letter_cnt
            letter_cnt += 1
        if size not in size_dict:
            size_dict[size] = size_cnt
            size_cnt += 1
        if fg not in fg_dict:
            fg_dict[fg] = fg_cnt
            fg_cnt += 1
        if bg not in bg_dict1:
            bg_dict1[bg] = bg_cnt1
            bg_cnt1 += 1
        if style not in style_dict:
            style_dict[style] = style_cnt
            style_cnt += 1


'''face'''
idf_dict = {}
exp_dict = {}
posef_dict = {}
idf_cnt = 0
exp_cnt = 0
posef_cnt = 0
for roots, dirs, files in os.walk('/home2/RaFD/train/data/'):
    for file in files:
        idf = file.split('_')[0]
        expression = file.split('_')[2].split('.')[0]
        posef = file.split('_')[1]
        if idf not in idf_dict:
            idf_dict[idf] = idf_cnt
            idf_cnt += 1
        if expression not in exp_dict:
            exp_dict[expression] = exp_cnt
            exp_cnt += 1
        if posef not in posef_dict:
            posef_dict[posef] = posef_cnt
            posef_cnt += 1
print(posef_cnt, posef_dict)
print(idf_cnt, idf_dict)
print(exp_cnt, exp_dict)