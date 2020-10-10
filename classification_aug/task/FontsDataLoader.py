import os
import pandas as pd
import numpy as np
from skimage import io
import random
class FontsDataGenerator:
    def __init__(self,data_path):
        self.data_path = data_path
        self.image_set = []
        self.image_train_id = []
        self.image_test_id  = []
    def get_cols_indexes(self,num_attr):
        res = []
        ch = 'a'
        for i in range(num_attr):
            res.append(ch)
            ch = ch+1
        return res
    def generate_train_set(self,data_size,attr_select):
        total_column = ['id','a','b','c','d','e']
        group_by_column = []
    def generate_samples(self,size):
        return self.image_set

class TestDataSet:
    def __init__(self):
        return
    def sleep(self):
        return
#dataset_path="/lab/tmpig23b/u/he-data/fonts_dataset_version0"


def FontsDataLoader(dataset_path,dataset_size,images_per_label,label_attribute,label_attribute_set,bias_attribute,bias_attribute_set):
    image_set = []
    attribute_set = []
    image_id = 0
    count_images = 0
    need_mkdir = True
    letter_attribute_set = ['a', 'A', 'b', 'B', 'c', 'C', 'd', 'D', 'e', 'E']
    size_attribute_set = ['large','small']
    front_color_attribute_set = ['blue1','Yellow1']
    background_color_attribute_set = ['blue','red']
    font_attribute_set = ['freemono','ubuntu','waree']
    #background_color_attribute_set = ['choo']
    image_save_path = '/lab/tmpig23b/u/he-data/font_dataset_little_240'
    classification = False
    if need_mkdir:
        os.mkdir(image_save_path)
    if classification and not os.path.isdir(image_save_path):
        os.mkdir(image_save_path)
    attribute_cols = ['id','a','b','c','d','e']
    #num_images_per_letter_fcolor = images_per_label // 52
    #group_attribute_cols = attribute_cols.remove(bias_attribute)
    #i = 0

    for letter_dir in os.listdir(dataset_path):
        if not (letter_dir in letter_attribute_set):
            continue
        #temp_images_per_letter = 0
        letter_path = os.path.join(dataset_path,letter_dir)
        if need_mkdir:
            save_letter_path = os.path.join(image_save_path,letter_dir)
            os.mkdir(save_letter_path)
        for size_dir in os.listdir(letter_path):
            if not (size_dir in size_attribute_set):
                continue

            letter_size_path = os.path.join(letter_path,size_dir)
            if need_mkdir:
                save_letter_size_path = os.path.join(save_letter_path,size_dir)
                os.mkdir(save_letter_size_path)
            for front_color_dir in os.listdir(letter_size_path):
                if not (front_color_dir in front_color_attribute_set):
                    continue
                temp_images_per_letter_fcolor = 0
                letter_size_fcolor_path = os.path.join(letter_size_path,front_color_dir)
                if need_mkdir:
                    save_letter_size_fcolor_path = os.path.join(save_letter_size_path,front_color_dir)
                    os.mkdir(save_letter_size_fcolor_path)
                for bcolor_dir in os.listdir(letter_size_fcolor_path):
                    if not (bcolor_dir in background_color_attribute_set):
                        continue
                    letter_size_fcolor_bcolor_path = os.path.join(letter_size_fcolor_path,bcolor_dir)
                    if need_mkdir:
                        save_letter_size_fcolor_bcolor_path = os.path.join(save_letter_size_fcolor_path,bcolor_dir)
                        os.mkdir(save_letter_size_fcolor_bcolor_path)
                    for font_dir in os.listdir(letter_size_fcolor_bcolor_path):
                        if not (font_dir in font_attribute_set):
                            continue
                        letter_size_fcolor_bcolor_font_path = os.path.join(letter_size_fcolor_bcolor_path,font_dir)
                        if need_mkdir:
                            save_letter_size_fcolor_bcolor_font_path = os.path.join(save_letter_size_fcolor_bcolor_path, font_dir)
                            os.mkdir(save_letter_size_fcolor_bcolor_font_path)
                    #for root, dirs, files in os.walk(letter_size_fcolor_bcolor_font_path, topdown=False):
                        for file in os.listdir(letter_size_fcolor_bcolor_font_path):
                            #fonts_category_count = fonts_category_count + 1
                            root = os.path.join(letter_size_fcolor_bcolor_font_path,file)
                            image = io.imread(os.path.join(letter_size_fcolor_bcolor_font_path,file))
                            if need_mkdir:
                                save_path = os.path.join(save_letter_size_fcolor_bcolor_font_path,file)
                                io.imsave(save_path,image)
                            if classification:
                                classification_save_dir = os.path.join(image_save_path,letter_dir)
                                if not os.path.isdir(classification_save_dir):
                                    os.mkdir(classification_save_dir)
                                classfication_save_path = os.path.join(classification_save_dir,file)
                                #classfication_save_path_copy1 = classfication_save_path+"copy1.png"
                                #classfication_save_path_copy2 = classfication_save_path+ "copy2.png"
                                #classfication_save_path_copy3 = classfication_save_path+ "copy3.png"
                                io.imsave(classfication_save_path,image)
                                #io.imsave(classfication_save_path_copy1, image)
                                #io.imsave(classfication_save_path_copy2, image)
                                #io.imsave(classfication_save_path_copy3, image)
                            #image = np.rollaxis(image,2,0)
                            foldername = root.split('/')
                            # be careful about file name
                            attribute = foldername[3:8]
                            attribute.insert(0,image_id)
                            attribute_set.append(attribute)
                            image_set.append(image)
                            count_images = count_images+1
                            image_id = image_id+1

                            #temp_images_per_letter = temp_images_per_letter +1
                            #temp_images_per_letter_fcolor = temp_images_per_letter_fcolor +1
                            #if temp_images_per_letter >= images_per_label or temp_images_per_letter_fcolor >= num_images_per_letter_fcolor  :
                             #   break
                        #if temp_images_per_letter >= images_per_label or temp_images_per_letter_fcolor >= num_images_per_letter_fcolor :
                         #   break
                    #if temp_images_per_letter >= images_per_label or temp_images_per_letter_fcolor >= num_images_per_letter_fcolor:
                     #   break
                #if temp_images_per_letter >= images_per_label:
                 #   break
            #if temp_images_per_letter>=images_per_label:
             #   break

        #if count_images >= dataset_size :
         #   break
    print("the loaded image number is ", count_images)
    attribute_set = pd.DataFrame(data=attribute_set,columns=['id','a','b','c','d','e'])
    #output_ids = np.zeros((count_images,),dtype=int)
    output_labels = np.empty((count_images,),dtype=object)
    output_labels[:] = "###"
    #group_id = 0
    #target_id_set = []
    for index,row in attribute_set.iterrows():
        output_labels[row['id']] = row[label_attribute]
    #attribute_set_group = attribute_set.groupby(by=[label_attribute])

    #num_attributes = len(bias_attribute_set)
    #count_groups = 0
    target_training_id = []
    target_test_id = []
    #for name, group in attribute_set_group:
        # filter group meets with the condition
    #    images_per_group = 0
     #   count_attributes_table = dict.fromkeys(bias_attribute_set, 0)
      #  count_groups  = count_groups + 1
       # for row_index,row in group.iterrows():
        #    if row[bias_attribute] in bias_attribute_set and count_attributes_table[row[bias_attribute]] < num_images_per_letter_fcolor:
         #       target_id_set.append(row['id'])
          #      images_per_group = images_per_group + 1
           #     count_attributes_table[row[bias_attribute]] = count_attributes_table[row[bias_attribute]] + 1

            #if sum(count_attributes_table.values()) >= images_per_label:
             #   break
    #print("the number of available attributes is {} ".format(num_attributes))
    #print("the number of available labels is {}".format(len(target_id_set)//images_per_label))

    output_labels = pd.factorize(output_labels)[0].tolist()
    output_labels = np.array(output_labels)
    image_set = np.array(image_set)
    # output_labels=np.unique(output_labels,return_inverse=True)[1].tolist()
    #image_set = image_set[target_id_set]
    #output_ids = output_ids[target_id_set]
    #output_labels = output_labels[target_id_set]
    print("data set size {} ".format(image_set.shape[0]))
    return image_set,output_labels

if __name__=="__main__":
    bias_attribute = 'c'
    # 'chocolate', 'cyan', 'green','orange''purple'
    bias_attribute_set = ['blue1','green1','pink1','red1','silver1','Yellow1']
    label_attribute = 'a'
    label_attribute_set = ['a','A','b','B','c','C','d','D','e','E']
    image_set , output_labels = FontsDataLoader(dataset_path="/home2/fonts_dataset_version2",
                                                dataset_size=10000000,
                                                images_per_label=1000,
                                                label_attribute='a',
                                                label_attribute_set=label_attribute_set,
                                                bias_attribute = bias_attribute,
                                                bias_attribute_set = bias_attribute_set)

    #image_set_save_path = "/lab/tmpig23b/u/he-data/fonts_dataset_test_5400"
    #label_set_save_path = "/lab/tmpig23b/u/he-data/fonts_test_label_5400"

    #np.save(image_set_save_path,image_set)
    #np.save(label_set_save_path,output_labels)