import os
import glob
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import random



def filter_images(path:str,b=0.1,delete=False):
    
    imgs = glob.glob(path+'*.jpg')
    
    erase=[]
    
    for _,i in enumerate(imgs):
        image = cv2.imread(i,0)
        tresh = round(len(image)*b)
        
        border = [image[:,0:tresh],image[0:tresh,:],image[len(image)-tresh:len(image),:],image[:,len(image)-tresh:len(image)]]
        
        b_perc = [np.count_nonzero(a==0)/a.size for a in border]
        
        if any(p > 0.4 for p in b_perc):
            erase.append(i)
    
    if delete: 
        for i in erase:
            os.remove(i)
    
    return erase


def crop_images(loadpath:str,savepath:str):
    
    imgs = glob.glob(loadpath+'*.jpg')
    
    for _,i in enumerate(imgs):
        image = cv2.imread(i,0)
        cr_image = image[49:177, 49:177]
        
        cv2.imwrite(savepath+i[12:],cr_image)
    
    return imgs



def sel_images(loadpath:str,savepath:str):
    
    imgs = glob.glob(loadpath+'*.jpg')
    imgs_sel = [i for i in imgs if len(i)==40]
    
    for _,i in enumerate(imgs_sel):
        image = cv2.imread(i,0)
        cv2.imwrite(savepath+i[12:],image)
    
    return imgs_sel
    

def save_samples(class_number, num=9):
    n=int(math.sqrt(num))
    img_list = glob.glob('croped_pics/*')
    labels_dict = pd.read_csv('landmarks_map-proj-v3_classmap.csv', names=['class_number','class_name'],index_col='class_number')
    labels = pd.read_csv('labels-map-proj-v3.txt',sep=' ',names=['file','class'])
    img_df = pd.DataFrame([i[12:] for i in img_list],columns=['file'])
    img_df=img_df.merge(labels, on='file', how='left')
    
    
    examples = [cv2.imread(i,0) for i in random.sample(['croped_pics/'+i for i in img_df.file[img_df['class']==class_number]],num)]
    # plot images
    for i in range(n * n):
        # define subplot
        plt.subplot(n, n, 1 + i)
        # turn off axis
        plt.axis('off')
        # plot raw pixel data
        plt.imshow(examples[i], cmap='gray_r')
    # save plot to file
    filename = f'sample_plot_class_[{class_number}]_{labels_dict.iloc[class_number][0]}.png'
    plt.savefig(filename)
    plt.close()

for i in range(8):
    save_samples(i)
    
    


for _,i in enumerate(imgs_sel):
    image = cv2.imread(i,0)
    cv2.imwrite(savepath+i[12:],image)