import cv2
import os
import numpy as np
from PIL import Image
import math
import sys
import OpenEXR
import Imath
import shutil
import glob

def conv_depth_disparity(data, baseline, focal_length_x, focal_length_y):
  
    nrows, ncols = data.shape
    k_mat=np.matrix([[focal_length_x, 0, (nrows/2.0)],[0, focal_length_y, (ncols/2.0)],[0, 0, 1.0]])
    
    for i in range(nrows):
        for j in range(ncols):
            norm_point =np.linalg.inv(k_mat)*np.matrix([[i], [j], [1.0]])
            theta = angle_between(norm_point.transpose(), np.array([0,0,1.0]))
            data[i,j]=(data[i,j])*theta
            if not data[i,j]==0.0:
                data[i,j]=(baseline*focal_length_x)/data[i,j]

    return data

def conv_depth_disparity_cam_cen(data, baseline, focal_length_x, focal_length_y):
    nrows, ncols = data.shape
    
    for i in range(nrows):
        for j in range(ncols):
            if not data[i,j]==0.0:
                data[i,j]=(baseline*focal_length_x)/data[i,j]

    return data

def read_K_mat(file_path):
    with open(file_path, 'r') as gt_file:
        file_lines = gt_file.readlines()
        return np.asarray(file_lines[0].split(" ")[1:-1]).reshape((3, 3)).astype(float)


def read_P_mat(file_path):
    with open(file_path, 'r') as gt_file:
        file_lines = gt_file.readlines()
        return np.asarray(file_lines[2].split(" ")[1:-1]).reshape((4, 4)).astype(float)

def rename_depth_files(depth_files_path):
    file_list = glob.glob(os.path.join(depth_files_path, "*.exr"))
    new_file_list = []
    for file_path in file_list:
        old = file_path
        new = file_path.split("_depth")[0] + ".exr"
        new_file_list.append(new)
        os.rename(old, new)
    return new_file_list

def conv_gt(path_root):
    for resolution_folder in os.listdir(path_root):
        path = os.path.join(path_root, resolution_folder)
        k_mat = read_K_mat(os.path.join(path, "calib.txt"))
        p_mat = read_P_mat(os.path.join(path, "calib.txt"))

        gt_path = os.path.join(path, "gt")
        gt_png_path = create_folder(os.path.join(path, "gt_png"))
        gt_image_paths = rename_depth_files(gt_path)
        for gt_image_path in gt_image_paths:
            image_data = universal_reader(gt_image_path)
            image_data = conv_depth_disparity(image_data, p_mat[0, 3], k_mat[0, 0], k_mat[1,1])
            image_name = os.path.basename(gt_image_path).split(".")[0]
            universal_writer(os.path.join(gt_path, image_name + ".raw"), image_data)
            universal_writer(os.path.join(gt_png_path, image_name + ".png"), image_data)
            os.remove(gt_image_path)

def conv_gt_dir(path, out_path, baseline, focal_length_x, focal_length_y):
    
    gt_folder_paths = sorted(glob.glob(os.path.join(path, "*/")))

    for gt_folder_num, gt_folder_path in enumerate(gt_folder_paths):
        image_data = universal_reader(glob.glob(os.path.join(gt_folder_path, "*.exr"))[0])
        image_data = conv_depth_disparity(image_data, baseline, focal_length_x, focal_length_y)
        universal_writer(os.path.join(out_path, str(gt_folder_num) + ".raw"), image_data)
        universal_writer(os.path.join(out_path, str(gt_folder_num) + ".png"), image_data)


def conv_dir(path, out_path, input_type="png", output_type="raw"):
    
    img_paths = sorted(glob.glob(os.path.join(path, "*." + input_type)))

    for img_num, img_path in enumerate(img_paths):
        image_data = universal_reader(img_path)
        universal_writer(os.path.join(out_path, str(img_num) + "." + output_type), image_data)


def unit_vector(vector):
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)

def universal_reader(path, height=480, width=640, dtype="float32"):
    extension = path.split(".")[-1]
    if(extension =="exr"):
        pt = Imath.PixelType(Imath.PixelType.FLOAT)
        golden = OpenEXR.InputFile(path)
        dw = golden.header()['dataWindow']
        size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
        redstr = golden.channel('R', pt)
        img_data = np.fromstring(redstr, dtype = np.float32)
        img_data.shape = (size[1], size[0]) # Numpy arrays are (row, col)

    if(extension=="raw"):
        img_data = np.fromfile(path, dtype=dtype)
        img_data = np.reshape(img_data, (height, width))
    if(extension=="pfm"):
        img_data = open(path, "rb")  # reopen the file
        
        pos = img_data.tell()  # Save the current position
        img_data.seek(0, 2)  # Seek to the end of the file
        length = img_data.tell()  # The current position is the length
        img_data.seek(pos)  # Return to the saved position
        
        img_data.seek(length-(height*width*4), os.SEEK_SET)  # seek
        
        img_data = np.fromfile(img_data, dtype=dtype)
        img_data = np.reshape(img_data, (height, width))
        img_data = np.flip(img_data, 0)

        #img_data[img_data>96.0] = 0.0
        #img_data[img_data<0.0] = 0.0
    if(extension=="png" or extension=="jpg"):
        img_data = cv2.imread(path,0)

    return img_data

def universal_writer(path, img_data):
    extension = path.split(".")[-1]
    if(extension=="raw"):
        img_data.tofile(path)
    if(extension=="pfm"):
        scale = 1
        file_object  = open(path, "wr")
        color = None
        if img_data.dtype.name != 'float32':
            raise Exception('Image dtype must be float32.')

        if len(img_data.shape) == 3 and img_data.shape[2] == 3: # color image
            color = True
        elif len(img_data.shape) == 2 or len(img_data.shape) == 3 and img_data.shape[2] == 1: # greyscale
            color = False
        else:
            raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

        file_object.write('PF\n' if color else 'Pf\n')
        file_object.write('%d %d\n' % (img_data.shape[1], img_data.shape[0]))

        endian = img_data.dtype.byteorder

        if endian == '<' or endian == '=' and sys.byteorder == 'little':
            scale = -scale

        file_object.write('%f\n' % scale)

        img_data.tofile(file_object)
            
    if(extension=="png" or extension=="jpg"):
        # scipy.misc.toimage(path, img_data)
        im = Image.fromarray(img_data)
        if im.mode != 'RGB':
            im = im.convert('RGB')
        im.save(path)
        # cv.imsave(path,img_data)
    return

def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return folder_path

