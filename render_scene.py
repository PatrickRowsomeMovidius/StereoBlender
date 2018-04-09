import bpy
import sys
sys.path.append("/home/rowsomep/scripts/utils/")
from blender_utils import *
import argparse

def parse_options():
    # get the args passed to blender after "--", all of which are ignored by
    # blender so scripts may receive their own arguments
    argv = sys.argv

    if "--" not in argv:
        argv = []  # as if no args are passed
    else:
        argv = argv[argv.index("--") + 1:]  # get all args after "--"

    # When --help or no args are given, print this help
    usage_text = (
        "Run blender in background mode with this script:"
        "  --python " + __file__ + " -- [options]"
        )

    parser = argparse.ArgumentParser(description=usage_text)

    parser.add_argument("-e", "--error_mode", dest="error_mode", type=str, default="none",
            help="This is a string option to set the error mode, possible values are: random, none, dyncal, x_axis, y_axis, z_axis")    
    parser.add_argument("-t", "--traj_mode", dest="traj_mode", type=str, default="none",
            help="This is a string option to set the trajectory mode, possible values are: none, random, linear, rotation, linear_random_rotation, linear_random_translation")
    parser.add_argument("-n", "--num_samples", dest="num_samples", type=int, default=10, help="Number of sequences to render")

    args = parser.parse_args(argv)

    return args

def check_camera_bookmarks():
    camera_bookmarks = ["Camera", "Camera.a", "Camera.b", "Camera.c"]
    l_poses = []
    for camera_bookmark in camera_bookmarks:
        if bpy.data.objects.get(camera_bookmark) is not None:
            cam_pose = bpy.data.objects[camera_bookmark].matrix_world.copy()
            l_poses.append(cam_pose)
    return l_poses

def main():
    options = parse_options() 
    """
    Values for data_width:
        8bit
        16bit
    
    All resolutions are supported.

    """
    resolutions=["640x480"]
    data_widths=["8bit"]
    
    l_poses = check_camera_bookmarks()
    l_cam, r_cam = setup_cams(bpy.data.objects["Camera"])
    lr_poses = calc_camera_poses(l_poses, options.num_samples, options.error_mode, options.traj_mode)

    for resolution in resolutions:
        for data_width in data_widths:

            # create dataset directories
            base_dir, l_data_dir, r_data_dir, gt_data_dir = create_movi_path(resolution, data_width)

            # setup Blender environment
            scene = bpy.context.scene
            output_node = setup_blender_env(scene, resolution, data_width)
            
            # set cameras and print calibration files
            output_node.base_path = gt_data_dir 
            stereo_transform(l_cam, r_cam, 0.2)
            print_calib_file(l_cam, r_cam, base_dir, "calib")
            
            # render scenes at defined poses
            render_scenes(lr_poses, scene, l_cam, l_data_dir, r_cam, r_data_dir, output_node)
            
    l_cam.name = "Camera"
    r_cam.select = True
    bpy.ops.object.delete()

if __name__ == '__main__':
    main()

