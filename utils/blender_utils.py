import bpy
from mathutils import Vector, Matrix, Euler
import os
import shutil
import random
import math
import numpy as np

baseline = 0.035
def create_movi_path(resolution, data_width):
    # create directories for output structure
    base_prj_name = bpy.path.basename(bpy.context.blend_data.filepath).split('.')[0]
    output_root = os.path.join("/home", os.getlogin(), "blender_output")
    base_out_dir = check_folder(os.path.join(output_root, base_prj_name, resolution))
    data_dir = create_folder(os.path.join(base_out_dir, data_width))
    l_data_dir = create_folder(os.path.join(data_dir, "left"))
    r_data_dir = create_folder(os.path.join(data_dir, "right"))
    gt_data_dir = create_folder(os.path.join(base_out_dir, "gt"))
    return base_out_dir, l_data_dir, r_data_dir, gt_data_dir

def universal_writer(path, img_data):
    extension = path.split(".")[-1]
    if(extension=="raw"):
        img_data.tofile(path)
    if(extension=="pfm"):
        scale = 1
        with open(path, "w") as file_object:
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

            if endian == '<' or endian == '=' and os.sys.byteorder == 'little':
                scale = -scale

            file_object.write('%f\n' % scale)

            img_data.tofile(file_object)

    return


def numpy_to_matrix(numpy_mat):
    return Matrix(numpy_mat.tolist())


def get_calibration_matrix_K(camd):
    f_in_mm = camd.lens
    scene = bpy.context.scene
    resolution_x_in_px = scene.render.resolution_x
    resolution_y_in_px = scene.render.resolution_y
    scale = scene.render.resolution_percentage / 100
    sensor_width_in_mm = camd.sensor_width
    sensor_height_in_mm = camd.sensor_height
    pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
    if (camd.sensor_fit == 'VERTICAL'):
        # the sensor height is fixed (sensor fit is horizontal),
        # the sensor width is effectively changed with the pixel aspect ratio
        s_u = resolution_x_in_px * scale / sensor_width_in_mm / pixel_aspect_ratio
        s_v = resolution_y_in_px * scale / sensor_height_in_mm
    else: # 'HORIZONTAL' and 'AUTO'
        # the sensor width is fixed (sensor fit is horizontal),
        # the sensor height is effectively changed with the pixel aspect ratio
        s_u = resolution_x_in_px * scale / sensor_width_in_mm
        s_v = resolution_y_in_px * scale * pixel_aspect_ratio / sensor_height_in_mm

    # Parameters of intrinsic calibration matrix K
    alpha_u = f_in_mm * s_u
    alpha_v = f_in_mm * s_v
    u_0 = resolution_x_in_px*scale / 2
    v_0 = resolution_y_in_px*scale / 2
    skew = 0 # only use rectangular pixels

    K = Matrix(
        ((alpha_u, skew,    u_0),
        (    0  ,  alpha_v, v_0),
        (    0  ,    0,      1 )))
    return K


def rand(radius):
    return random.uniform(-radius,radius)


def rpy_rotation_mat(roll_deg, pitch_deg, yaw_deg):
    roll_rot = Matrix.Rotation(math.radians(roll_deg), 4, 'X')
    pitch_rot = Matrix.Rotation(math.radians(pitch_deg), 4, 'Y')
    yaw_rot = Matrix.Rotation(math.radians(yaw_deg), 4, 'Z')
    return  yaw_rot * pitch_rot * roll_rot


def stereo_transform(l_obj, r_obj, baseline):
    r_obj.matrix_world = l_obj.matrix_world.copy()
    trans_local = Vector((baseline, 0.0, 0.0))
    trans_world = r_obj.matrix_world.to_3x3() * trans_local
    r_obj.matrix_world.translation += trans_world

def get_calibration_matrix_P(l_cam, r_cam):
    return l_cam.matrix_world.inverted() * r_cam.matrix_world

def print_calib_file(l_cam_obj, r_cam_obj, output_directory, filename):
    # print k matrix for left and right camera
    calib_file_path = os.path.join(output_directory, filename + ".txt")
    with open(calib_file_path, 'w') as calib_file:
        left_K = get_calibration_matrix_K(l_cam_obj.data)
        right_K = get_calibration_matrix_K(r_cam_obj.data)
        calib_file.write("Cam0 ")
        [calib_file.write(str(data[0]) + " " + str(data[1]) + " " + str(data[2]) + " ") for data in left_K]
        calib_file.write("\n")
        calib_file.write("Cam1 ")
        [calib_file.write(str(data[0]) + " " + str(data[1]) + " " + str(data[2]) + " ") for data in right_K]
        calib_file.write("\n")
        p = get_calibration_matrix_P(l_cam_obj, r_cam_obj)
        print(p)
        calib_file.write("P1 ")
        [calib_file.write(str(data[0]) + " " + str(data[1]) + " " + str(data[2]) + " " + str(data[3]) + " ") for data in p]
        calib_file.write("\n")

def check_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return folder_path

def create_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path)
    return folder_path

def setup_render_options(scene, resolution, data_width, use_GPU=True):
    # setup render options for GPU/CPU
    use_GPU = True
    if use_GPU:
        scene.cycles.device = 'GPU'
        bpy.context.user_preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
        bpy.context.user_preferences.addons['cycles'].preferences.compute_device = 'CUDA_2'
        scene.render.tile_x = 256
        scene.render.tile_y = 256
    else:
        scene.cycles.device = 'CPU'
        scene.render.tile_x = 16
        scene.render.tile_y = 16

    scene.cycles.max_bounces = 4
    scene.cycles.samples = 2000

    x_res, y_res = resolution.split("x")
    scene.render.resolution_x = int(x_res) 
    scene.render.resolution_y = int(y_res)
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'BW'
    if(data_width=="16bit"): 
        scene.render.image_settings.color_depth = '16'
    else:
        scene.render.image_settings.color_depth = '8'

    scene.render.image_settings.compression = 0
    scene.use_nodes = True


def setup_cams(l_cam, res):
    pixel_width = 0.003
    l_cam.data.sensor_height = res[1] * pixel_width 
    l_cam.data.sensor_width= res[0] * pixel_width
    l_cam.data.lens=2.793615
    l_cam.data.sensor_fit='HORIZONTAL'
    l_cam.select = True
    bpy.ops.object.duplicate()
    r_cam = bpy.data.objects["Camera.001"]
    l_cam.name = "L"
    r_cam.name = "R"
    return l_cam, r_cam


def setup_scene_node_graph(tree):
    # init scene node graph
    if type(tree.links) != None:
        links = tree.links

        for n in tree.nodes:
            tree.nodes.remove(n)

    # set up scene compositor graph
    render_node = tree.nodes.new(type="CompositorNodeRLayers")
    output_node = tree.nodes.new(type="CompositorNodeOutputFile")

    output_node.format.file_format = 'OPEN_EXR'
    output_node.format.color_depth = '32'
    output_node.format.use_zbuffer = True

    links.new(render_node.outputs[2], output_node.inputs[0])
    
    return output_node


def setup_blender_env(scene, resolution, data_width, use_GPU=True):
    setup_render_options(scene, resolution, data_width, use_GPU=True)
    return setup_scene_node_graph(scene.node_tree)


def read_calibration_files(root_path, convert_mm_to_m, invert):
    l_cam_k=[]
    r_cam_k=[]
    P1_mat = []
    for calib_no in range(10):
        calib_path=os.path.join(root_path, str(calib_no) + ".txt")
        with open(calib_path, 'r') as calib_data:
            for line_no, line in enumerate(calib_data):
                if line_no == 0:
                    l_cam_k.append(np.matrix(line.split(" ")[1:-1]).astype(np.double).reshape((3,3)))
                if line_no == 1:
                    r_cam_k.append(np.matrix(line.split(" ")[1:-1]).astype(np.double).reshape((3,3)))
                if line_no == 2:
                    mat = np.matrix(line.split(" ")[1:-1]).astype(np.double).reshape((4,4))
                    if convert_mm_to_m:
                        mat[0,3]=mat[0,3]/1000.0
                        mat[1,3]=mat[1,3]/1000.0
                        mat[2,3]=mat[2,3]/1000.0
                        mat[1,0] *= -1
                        mat[2,0] *= -1
                        mat[0,1] *= -1
                        mat[0,2] *= -1
                    P1_mat.append(mat)
    return l_cam_k, r_cam_k, P1_mat


def calc_calibration_errors(l_gt_k, r_gt_k, lr_gt_p, l_k, r_k, lr_p):
    l_k_errors = []
    r_k_errors = []
    lr_poses = []
    
    for l_gt_k_mat, l_k_mat in zip(l_gt_k, l_k):
        l_k_errors.append(l_k_mat - l_gt_k_mat)
    
    for r_gt_k_mat, r_k_mat in zip(r_gt_k, r_k):
        r_k_errors.append(r_k_mat - r_gt_k_mat)
    
    for lr_gt_p_mat, lr_p_mat in zip(lr_gt_p, lr_p):
        error = np.linalg.inv(lr_p_mat)*lr_gt_p_mat
        lr_poses.append(numpy_to_matrix(error))

    return l_k_errors, r_k_errors, lr_poses


def render_lr_cams(scene, l_cam, l_data_dir, r_cam, r_data_dir, output_node, frame_number):
    # set filename for GT depth
    output_node.file_slots[0].path = str(frame_number) +  "_depth"

    # render scene from left and right camera
    scene.camera = r_cam 
    scene.render.filepath = os.path.join(r_data_dir, "im" + str(frame_number))
    bpy.ops.render.render(animation=False, write_still=True)

    scene.camera = l_cam 
    scene.render.filepath = os.path.join(l_data_dir, "im" + str(frame_number))
    bpy.ops.render.render(animation=False, write_still=True)


def render_scenes(lr_poses, scene, l_cam, l_data_dir, r_cam, r_data_dir, output_node):
    # save init pose
    pose_init = l_cam.matrix_world.copy()
    
    for frame_number, pose_pair in enumerate(lr_poses):
        l_cam.matrix_world = pose_pair[0]
        r_cam.matrix_world = pose_pair[1]
        render_lr_cams(scene, l_cam, l_data_dir, r_cam, r_data_dir, output_node, frame_number)
    
    # reset left cam pose
    l_cam.matrix_world = pose_init


def random_pose_delta():
    pose_delta = Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    return random_error(pose_delta)


def random_error(matrix_world, x_ang_thres=2, y_ang_thres=2, z_ang_thres=2, trans_thres=0):
    pose = matrix_world.copy()
    # rotation
    rotation = rpy_rotation_mat(rand(x_ang_thres),rand(y_ang_thres),rand(z_ang_thres))
    pose = matrix_world * rotation
    # translation
    pose.translation.x += rand(trans_thres)
    pose.translation.y += rand(trans_thres)
    pose.translation.z += rand(trans_thres)
    return pose


def local_transform(matrix_world, x_trans=0, y_trans=0, z_trans=0):
    pose = matrix_world.copy()
    trans_local = Vector((x_trans, y_trans, z_trans))
    trans_world = matrix_world.to_3x3() * trans_local
    pose.translation += trans_world
    return pose


def random_positions(lr_poses, x_ang_thres=20, y_ang_thres=20, z_ang_thres=20, trans_thres=1):
    poses = []
    for lr_pose in lr_poses:
        l_pose = random_error(lr_pose[0], x_ang_thres, y_ang_thres, z_ang_thres, trans_thres)
        r_pose = local_transform(l_pose, x_trans=baseline)
        poses.append([l_pose, r_pose])
    return poses

def linear_translation(camera_pose_pairs, path_length=5.0):
    step_size=float(path_length)/len(camera_pose_pairs)
    poses = []
    orig_l_cam_pose = camera_pose_pairs[0][0]
    orig_r_cam_pose = camera_pose_pairs[0][1]

    for step_num in range(len(camera_pose_pairs)+1):
        l_cam_pose = local_transform(orig_l_cam_pose, z_trans=step_size*(-step_num))
        r_cam_pose = local_transform(l_cam_pose, x_trans=baseline)
        print(l_cam_pose)
        poses.append([l_cam_pose, r_cam_pose])
    return poses

def const_rotation(camera_pose_pairs, angle_range_radius=45.0, axis="x"):
    poses = []
    angle = angle_range_radius
    
    for pose_num, lr_pose in enumerate(camera_pose_pairs):

        if axis == "y":
            rotation = rpy_rotation_mat(0, angle, 0)
        elif axis == "z":
            rotation = rpy_rotation_mat(0, 0, angle)
        elif axis == "x":
            rotation = rpy_rotation_mat(angle, 0, 0)
        
        r_pose = local_transform(lr_pose[0], x_trans=baseline)
        r_pose = r_pose*rotation
        poses.append([lr_pose[0], r_pose])
    return poses

def axis_rotation(camera_pose_pairs, angle_range_radius=45.0, axis="x"):
    poses = []
    angle_step_size = float(angle_range_radius)*2.0/len(camera_pose_pairs)
    angle = -angle_range_radius
    
    for pose_num, lr_pose in enumerate(camera_pose_pairs):
        rotation = rpy_rotation_mat(angle, 0, 0)
        if axis == "y":
            rotation = rpy_rotation_mat(0, angle, 0)
        elif axis == "z":
            rotation = rpy_rotation_mat(0, 0, angle)
        r_pose = local_transform(lr_pose[0], x_trans=baseline)
        r_pose = r_pose*rotation
        poses.append([lr_pose[0], r_pose])
        angle+=angle_step_size
    return poses

def calc_camera_poses(l_cam_poses, num_samples, error_mode="no_error", traj_mode="stationary"):
    
    camera_poses = []

    for l_cam_pose in l_cam_poses:
        # create baseline for right camera 
        r_cam_pose = local_transform(l_cam_pose, x_trans=baseline)
        
        # init camera list of left-right camera pose tuples  
        camera_pose_pairs = [[l_cam_pose, r_cam_pose] for _ in range(num_samples)]
        
        # calc trajectory, stationry, axis_rotation, linear, random, linear_random_rotations
        if traj_mode == "linear":
            camera_pose_pairs = linear_translation(camera_pose_pairs, path_length=5.0)
        elif traj_mode == "random":
            camera_pose_pairs = random_positions(camera_pose_pairs)
        elif traj_mode == "rotation":
            camera_pose_pairs = axis_rotation(camera_pose_pairs)
        elif traj_mode == "linear_random_rotation":
            camera_pose_pairs = linear_translation(camera_pose_pairs)
            camera_pose_pairs = random_positions(camera_pose_pairs, thrans_thres=0)
        elif traj_mode == "linear_random_translation":
            camera_pose_pairs = linear_translation(camera_pose_pairs)
            camera_pose_pairs = random_positions(camera_pose_pairs, x_ang_thres=0, y_angle_thres=0, z_angle_thres=0)

        # add errors to cameras: no error, random error, or parameter sequence
        if error_mode == "random":
            camera_pose_pairs = [[pose_pair[0], random_error(pose_pair[1])] for pose_pair in camera_pose_pairs]
        elif error_mode == "const":
            pose_delta = random_pose_delta()
            camera_pose_pairs = [[pose_pair[0], pose_pair[1]*pose_delta] for pose_pair in camera_pose_pairs]
        elif error_mode == "dyncal":
            # read dyncal results and calc errors compared to GT
            result_path = os.path.join("/home", os.getlogin() , "blender_output/dyncal_results/Kitchen_06_rotations")
            gt_path = os.path.join("/home", os.getlogin() , "blender_output/dyncal_gt/Kitchen_06_rotations")
            l_gt_k, r_gt_k, lr_gt_p = read_calibration_files(gt_path, False, False)
            l_k, r_k, lr_p = read_calibration_files(result_path, True, True)
            l_k_errors, r_k_errors, lr_poses = calc_calibration_errors(l_gt_k, r_gt_k, lr_gt_p, l_k, r_k, lr_p)
            camera_pose_pairs = [[l_cam_pose, r_cam_pose] for _ in range(len(lr_poses))]
            camera_pose_pairs = [[camera_pose_pairs[i][0], camera_pose_pairs[i][1]*lr_poses[i]] for i in range(len(camera_pose_pairs))]
        elif error_mode == "x_axis":
            camera_pose_pairs = axis_rotation(camera_pose_pairs, 1, "x")
        elif error_mode == "y_axis":
            camera_pose_pairs = axis_rotation(camera_pose_pairs, 1, "y")
        elif error_mode == "z_axis":
            camera_pose_pairs = axis_rotation(camera_pose_pairs, 1, "z")
        elif error_mode == "cx_axis":
            camera_pose_pairs = const_rotation(camera_pose_pairs, 1, "x")
        elif error_mode == "cy_axis":
            camera_pose_pairs = const_rotation(camera_pose_pairs, 1, "y")
        elif error_mode == "cz_axis":
            camera_pose_pairs = const_rotation(camera_pose_pairs, 1, "z")

        camera_poses = camera_poses + camera_pose_pairs
    
    return camera_poses
