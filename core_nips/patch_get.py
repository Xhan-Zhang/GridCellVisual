
import cv2
import random
import utils
import os
import numpy as np
from sklearn.decomposition import PCA
from core_nips.defaults import get_default_hp

hp = get_default_hp()
picture_dir = hp['picture_dir']




def generate_patch(img=None,box_left=0, box_right=0, box_up=0, box_below=0):

    cropImg = img[box_up:box_below, box_left:box_right]

    img_gray = cv2.cvtColor(cropImg,cv2.COLOR_RGB2GRAY)
    img_gray = img_gray.flatten()
    if np.max(img_gray)-np.min(img_gray) == 0:
        img_gray = np.zeros_like(img_gray)
    else:
        img_gray = (img_gray - np.min(img_gray))/ \
                              (np.max(img_gray)-np.min(img_gray))


    return img_gray


def generate_patch_random(img=0,h = 8,w = 8):


    count=0
    pca_matrixs = []
    while 1:
        y = random.randint(1, 550)
        x = random.randint(1, 550)
        cropImg = img[(y):(y + h), (x):(x + w)]
        cv2.imwrite(hp['picture_dir'] + str(count) + '.png', cropImg)
        gray_image = cv2.imread(hp['picture_dir'] + str(count) + '.png',0)
        gray_image = gray_image.flatten()
        pca_matrixs.append(gray_image)
        count+=1

        if count==2000:
            break

    pca_matrix_array = np.array(pca_matrixs)
    return pca_matrix_array.transpose()

def pca_image(img=0):

    pca_matrix = generate_patch_random(img=img)

    pca = PCA(n_components=20)
    new_pca_matrix = pca.fit_transform(pca_matrix)
    explained_variance = pca.explained_variance_ratio_

    #normalization

    new_pca_matrix = (new_pca_matrix - np.min(new_pca_matrix))/ \
                          (np.max(new_pca_matrix)-np.min(new_pca_matrix))

    np.save('./img/'+"pca_matrix_imgnature1.npy",new_pca_matrix)

    return new_pca_matrix

if __name__ == '__main__':
    img=cv2.imread('./img/nature1.png')
    cropImg_550 = img[1:552, 1:552]
    cv2.imwrite( 'cropImg_0.png', cropImg_550)
    img=cv2.imread('cropImg_0.png')


    new_pca_matrix = pca_image(img=img)




