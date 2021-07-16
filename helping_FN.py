import numpy as np
import pandas as pd
import pickle 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import cv2
from sklearn.utils import shuffle
from numpy import newaxis  #http://stackoverflow.com/questions/7372316/how-to-make-a-2d-numpy-array-a-3d-array

import matplotlib.pyplot as plt

"""***********************************************************************"""
def get_statistics(input_labels):
    """
    Returns a DataFrame containing the count of every label in input_labels array.
    It gives a rough estimation of how many more images are needed to be added to create a balance among various labels
    """
    
    img_labels = pd.DataFrame( {'Labels': input_labels })
    #img_per_label = img_labels.groupby('Labels').size()
    count_per_label = pd.DataFrame({'img_count' : img_labels.groupby('Labels').size()}).reset_index()
    count_per_label['added_imgs']= np.floor_divide(count_per_label['img_count'].max(),count_per_label['img_count'])-1
    
    return count_per_label
"""***********************************************************************"""
def show_images(image_list, titles, plot_size, size = (2,2) ):
    
    """
    A handy function to display images
    """
    
    rows,cols = size[0], size[1]
    lables = titles.reshape(rows, cols)
    
    if(len(image_list[0].shape) > 2 ): # RGB Image
        
        img_width, img_hight, ch = image_list[0].shape
        images = image_list.reshape(rows,cols,img_width,img_hight,ch)
    else:
        img_width, img_hight = image_list[0].shape
        images = image_list.reshape(rows,cols,img_width,img_hight)
    
    
      
    f1, ax = plt.subplots(rows, cols ,figsize=plot_size)
    f1.tight_layout()
    f1.subplots_adjust(hspace=0.15)

    for i in range(rows):
        for j in range(cols):
            
            counter = i*cols +j
            ax[i][j].imshow(images[i][j],'gray')
            ax[i][j].set_title(str(counter)+": " + lables[i][j], color ='r', fontweight='bold')
            ax[i][j].axis('off')
    
"""***********************************************************************"""
def convert_eq_gray(in_img):
    
    '''
        Convert an input color image to equalized gray image
        EXPAND is a flag:
            TRUE: change input from (32*32 --> 32*32*1): used with original images
            FALSE: keep the dimension (32*32): used in generating new images
    '''
   
    equiliezed_gray = []
    
    for index in range(0, len(in_img)):
    
        img = in_img[index].squeeze()
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        #Image Enhancement via Histogram Equalization
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2,2)) # clipLimit=2.0, tileGridSize=(8,8)
        equ_img = clahe.apply(gray_img)
        
         #denoising the image
        #equ_img =cv2.fastNlMeansDenoisingMulti(equ_img, 2, 5, None, 4, 7, 35)
        
        equ_img = equ_img[..., newaxis] #convert from 32*32 --> 32*32*1
        equiliezed_gray.append(equ_img)
        
    
    return equiliezed_gray

"""***********************************************************************"""
def adjust_images(in_img, pivot):
    
    '''
        GOAL: Goal: creating a balance among various image classes by adding extra 
              derived images to labels with low image count
        
        defines 13 transformations of a given image to compensate for calss imbalance.
        return suitable number based on a caluclated pivot value
        the pivot is usually the ratio between max and min number of images per label
        INPUT: img- RGB 32*32*3 color image
        OUTPUT: list of 32*32*1 image
        
        the pivot determines the number of required images
    '''
    
    blue_ch, green_ch, red_ch = cv2.split(in_img)
    
    hls = cv2.cvtColor(in_img, cv2.COLOR_RGB2HLS)

    H = hls[:,:,0]
    L = hls[:,:,1]
    S = hls[:,:,2]

    
    gray_img = cv2.cvtColor(in_img, cv2.COLOR_RGB2GRAY)
  
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2,2))
    equ_gray_img = clahe.apply(gray_img)
    
 
    
    # The following kernels were adapted from
    # https://www.packtpub.com/mapt/book/Application+Development/9781785283932/2/ch02lvl1sec22/Sharpening
    
    kernel_sharpen_1 = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    kernel_sharpen_2 = np.array([[1,1,1], [1,-7,1], [1,1,1]])
    kernel_sharpen_3 = np.array([[-1,-1,-1,-1,-1],
                             [-1,2,2,2,-1],
                             [-1,2,8,2,-1],
                             [-1,2,2,2,-1],
                             [-1,-1,-1,-1,-1]]) / 8.0
    
    
    rows,cols = equ_gray_img.shape
    M1 = cv2.getRotationMatrix2D((cols/2,rows/2),45,1) # image rotation
    M2 = cv2.getRotationMatrix2D((cols/2,rows/2),135,1) # image rotation
    M3 = cv2.getRotationMatrix2D((cols/2,rows/2),270,1) # image rotation
    
    noise = np.random.randn(*gray_img[1].shape)*10 #random noise
    
    
    tf1 = cv2.flip(gray_img, 0) # flip image horizontally
    tf2 = cv2.flip(gray_img,1) # flip  image vertically
    tf3 = cv2.flip(gray_img, -1) # flip image both horisontally and vertically
    tf4 = np.copy(blue_ch)
    tf5 = np.copy(green_ch)
    tf6 = np.copy(red_ch)
    tf7 = cv2.filter2D(equ_gray_img, -1, kernel_sharpen_1)
    tf8 = cv2.filter2D(equ_gray_img, -1, kernel_sharpen_2)
    tf9 = cv2.filter2D(equ_gray_img, -1, kernel_sharpen_3)
    tf10 = cv2.warpAffine(equ_gray_img,M1,(cols,rows))
    tf11 = cv2.warpAffine(equ_gray_img,M2,(cols,rows))
    tf12 = cv2.warpAffine(equ_gray_img,M3,(cols,rows))
    tf13 = np.copy(L)
    tf14 = np.copy(S)
    tf15 = [i+noise for i in gray_img] # Add this noise to images
    #tf16 = cv2.fastNlMeansDenoisingMulti(gray_img, 2, 5, None, 4, 7, 35) this is very slow
    tf16 = S+red_ch+tf3
    tf17 = tf1+tf2
    tf18 = tf11+S

    
    img_list = [tf1,tf2,tf3, tf4, tf5, tf6, tf7, tf8, tf9, tf10, tf11, tf12, tf13, tf14, tf15, tf16, tf17, tf18]
    
    #random.choice(img_list)
    random_sel = shuffle(img_list)
    
    #Add extra images according to the pivot
    #in case pivot is 1 return the normal gray image
    
    if(pivot ==1):
        return [equ_gray_img]
    else:
        if(pivot > 18):
            pivot = 18
        
        return random_sel[0:pivot]

"""***********************************************************************"""
def generate_images(in_img, in_label):
    
    additional_images = []
    additional_labels = []
    
    #returns min and max for the number of images per label
    label_stat = get_statistics(in_label)
    
    # convert to Dict. for easy look up
    label_stat_dict = label_stat.set_index('Labels')['added_imgs'].to_dict()
    
    #print(label_stat_dict)
    
    for index in range(0, len(in_img)):
        
        #get number of images to be generated
        
        pivot = label_stat_dict[in_label[index]]
        
        img = in_img[index]
        
                
        new_images = adjust_images(img, pivot)
        #print(len(new_images), pivot)
        
        new_labels = [in_label[index]]*len(new_images) # all new generated images will have the same label
            
        additional_images.append(new_images)
        additional_labels.append(new_labels)
 
    
    
    print("The length before extendeing images = {:} and the length of the assoc. label = {:} ".format(len(additional_images), len(additional_labels)))
    
    flatten_labels = [item for sublist in additional_labels for item in sublist]
    image_labels = np.asarray(flatten_labels) 
    
    
    flatten_images = [item for sublist in additional_images for item in sublist]
    image_list = []
    
    for index in range(0, len(flatten_images)):
        temp_adj_img = np.asarray(flatten_images[index])
        temp_adj_img = temp_adj_img[..., newaxis]
        image_list.append(temp_adj_img)
    
    print("The length after extendeing images = {:} and the length of the assoc. label = {:} ".format(len(image_list), len(image_labels)))
    print("--------------------------------------------------")
  

    return image_list, image_labels

"""***********************************************************************"""
def get_Extended_Images(org_img, org_label):
    
    # Converting original Images to gray scale
    org_gray_Imgs = convert_eq_gray(org_img)

    extended_images, extended_labels = generate_images(org_img, org_label)
    #img_lists = np.asarray(extended_images + gray_Imgs )
    
    img_lists = org_gray_Imgs + extended_images
    #image_labels = np.concatenate((label, extended_labels), axis=0)
    image_labels = np.hstack((org_label, extended_labels))
    
    return np.asarray(img_lists), image_labels 

"""***********************************************************************"""

"""***********************************************************************"""

"""***********************************************************************"""

"""***********************************************************************"""
"""***********************************************************************"""



