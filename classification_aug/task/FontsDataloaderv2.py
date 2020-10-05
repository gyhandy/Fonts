import os
import pandas as pd
import numpy as np
from skimage import io
import random


def FontsDataLoader(dataset_path,dataset_size,):
    image_set = []
    attribute_set = []
    image_id = 0
    count_images = 0
    size_attribute_set = ['larget','medium','small']
    background_color_attribute_set = ['cyan','green','red','silver']
    #background_color_attribute_set = ['choo']
    front_color_attribute_set = []
    attribute_cols = ['id','a','b','c','d','e']
    num_images_per_letter_fcolor = images_per_label // 52
    #group_attribute_cols = attribute_cols.remove(bias_attribute)
    i = 0
    for letter_dir in os.listdir(dataset_path):
        temp_images_per_letter = 0
        letter_path = os.path.join(dataset_path,letter_dir)
        for size_dir in os.listdir(letter_path):
            if not (size_dir in size_attribute_set):
                continue
            letter_size_path = os.path.join(letter_path,size_dir)
            for front_color_dir in os.listdir(letter_size_path):
                if not (front_color_dir in bias_attribute_set):
                    continue
                temp_images_per_letter_fcolor = 0
                letter_size_fcolor_path = os.path.join(letter_size_path,front_color_dir)

                for bcolor_dir in os.listdir(letter_size_fcolor_path):
                    if not (bcolor_dir in background_color_attribute_set):
                        continue
                    letter_size_fcolor_bcolor_path = os.path.join(letter_size_fcolor_path,bcolor_dir)

                    for root, dirs, files in os.walk(letter_size_fcolor_bcolor_path, topdown=False):
                        for file in files:
                            #fonts_category_count = fonts_category_count + 1
                            image = io.imread(os.path.join(root,file))
                            #image = np.rollaxis(image,2,0)
                            foldername = root.split('/')
                            # be careful about file name
                            attribute = foldername[3:]
                            attribute.insert(0,image_id)
                            attribute_set.append(attribute)
                            image_set.append(image)
                            count_images = count_images+1
                            image_id = image_id+1

                            temp_images_per_letter = temp_images_per_letter +1
                            temp_images_per_letter_fcolor = temp_images_per_letter_fcolor +1
                            if temp_images_per_letter >= images_per_label or temp_images_per_letter_fcolor >= num_images_per_letter_fcolor  :
                                break
                        if temp_images_per_letter >= images_per_label or temp_images_per_letter_fcolor >= num_images_per_letter_fcolor :
                            break
                    if temp_images_per_letter >= images_per_label or temp_images_per_letter_fcolor >= num_images_per_letter_fcolor:
                        break
                if temp_images_per_letter >= images_per_label:
                    break
            if temp_images_per_letter>=images_per_label:
                break

        if count_images >= dataset_size :
            break


if __name__=="__main__":
    bias_attribute = 'c'
    # 'chocolate', 'cyan', 'green','orange''purple'
    bias_attribute_set = ['blue' ,'pink','red','silver','Yellow']
    label_attribute = 'a'
    label_attribute_set = ['a','A','b','c','d','E','S','v','W','w']
    dataset_path = ''
    image_set , output_labels = FontsDataLoader()

    image_set_save_path = "/lab/tmpig23b/u/he-data/train_image_set_small"
    label_set_save_path = "/lab/tmpig23b/u/he-data/train_label_set_small"

    np.save(image_set_save_path,image_set)
    np.save(label_set_save_path,output_labels)