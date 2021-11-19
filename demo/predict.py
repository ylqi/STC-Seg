import os

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import glob
import multiprocessing as mp
import time
import cv2
import tqdm

from detectron2.data import MetadataCatalog
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from demo import VisualizationDemo
from stcSeg.config import get_cfg

import numpy as np
import scipy.ndimage

# constants
WINDOW_NAME = "COCO detections"


def setup_cfg(args):
    # load config from file and command-line arguments
    
    # Set score_threshold for builtin models
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def run(demo, args, image_path, logger, cfg):
    if args.input:
        if os.path.isdir(args.input[0]):
            # print(args.input)
            args.input = [os.path.join(args.input[0], fname) for fname in sorted(os.listdir(args.input[0]))]
        elif len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"

        if args.tracking:
            tracking_dict = {}

        # print(args.input)

        seq_name = args.input[0].split('/')[-2]

        for path in tqdm.tqdm(args.input, disable=not args.output, desc=seq_name):
            # use PIL, to be consistent with evaluation
            # print(path)
            img = read_image(path, format="BGR")
            # print("img:", img.shape)

            if args.tracking_with_depth:
                if args.split == "test":
                    depth_file_path = '%s/%s/%s/depth/%s/%s_disp.npy' % (args.data_root, args.dataset, image_path[args.split], seq_name, path.split('/')[-1].split(".")[0])
                    depth_data = np.load(depth_file_path)
                    depth_data = depth_data[0].transpose(1,2,0)
                else:
                    depth_file_path = '%s/%s/%s/depth/%s/%s.npy' % (args.data_root, args.dataset, image_path[args.split], seq_name, path.split('/')[-1].split(".")[0])
                    depth_data = np.load(depth_file_path)
                    # depth_file_path = '%s/%s/%s/depth_monodepth2/%s/%s_disp.npy' % (args.data_root, args.dataset, image_path[args.split], seq_name, path.split('/')[-1].split(".")[0])
                    depth_data = depth_data.transpose(1,2,0)
                    # depth_data = depth_data[0].transpose(1,2,0)

                
                # 
            else:
                depth_data = None

            if args.tracking_with_flow:
                try:
                    flow_file_path = '%s/%s/%s/optical_flow/%s/%s.flo' % (args.data_root, args.dataset, image_path[args.split], seq_name, path.split('/')[-1].split(".")[0])
                    f = open(flow_file_path, 'rb')

                    x = np.fromfile(f, np.int32, count=1) # not sure what this gives
                    w = int(np.fromfile(f, np.int32, count=1)) # width
                    h = int(np.fromfile(f, np.int32, count=1)) # height

                    flow_data = np.fromfile(f, np.float32) # vector 
                    flow_data = np.reshape(flow_data, newshape=(h, w, 2)); # convert to x,y - flow
                    flow_data = scipy.ndimage.zoom(flow_data, (img.shape[0] / h, img.shape[1] / w, 1))
                    # print("flow:", flow_data.shape)
                except Exception as e:
                    print(e)
                    flow_data = None
            else:
                flow_data = None

            if args.tracking_with_feature != '':
                from torchvision import models
                feature_model = models.vgg19(pretrained=True)
                feature_model.classifier = feature_model.classifier[:-3]
                feature_model = feature_model.eval()
            else:
                feature_model = None

            start_time = time.time()
            if args.tracking:
                predictions, visualizer, visualized_output, tracking_dict = demo.run_and_track_on_images(img, tracking_dict, path,
                                                                                  depth_data, flow_data, feature_model, box_mode=args.box_mode, args=args)

                
            else:
                predictions, visualized_output = demo.run_on_image(img)
            # logger.info(
            #     "{}: detected {} instances in {:.2f}s".format(
            #         path, len(predictions["instances"]), time.time() - start_time
            #     )
            # )

            if args.output:
                if os.path.isdir(args.output):
                    assert os.path.isdir(args.output), args.output
                    out_filename = os.path.join(args.output, os.path.basename(path))
                else:
                    if len(args.input) > 1:
                        os.mkdir(args.output)
                        out_filename = os.path.join(args.output, os.path.basename(path))
                    else:
                        out_filename = args.output

                visualized_output.save(out_filename)
            else:
                cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
                if cv2.waitKey(0) == 27:
                    break  # esc to quit
    elif args.webcam:
        assert args.input is None, "Cannot have both --input and --webcam!"
        cam = cv2.VideoCapture(0)
        for vis in tqdm.tqdm(demo.run_on_video(cam)):
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.imshow(WINDOW_NAME, vis)
            if cv2.waitKey(1) == 27:
                break  # esc to quit
        cv2.destroyAllWindows()
    elif args.video_input:
        video = cv2.VideoCapture(args.video_input)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        basename = os.path.basename(args.video_input)

        if args.output:
            if os.path.isdir(args.output):
                output_fname = os.path.join(args.output, basename)
                output_fname = os.path.splitext(output_fname)[0] + ".mkv"
            else:
                output_fname = args.output
            assert not os.path.isfile(output_fname), output_fname
            output_file = cv2.VideoWriter(
                filename=output_fname,
                # some installation of opencv may not support x264 (due to its license),
                # you can try other format (e.g. MPEG)
                fourcc=cv2.VideoWriter_fourcc(*"x264"),
                fps=float(frames_per_second),
                frameSize=(width, height),
                isColor=True,
            )
        assert os.path.isfile(args.video_input)
        for vis_frame in tqdm.tqdm(demo.run_on_video(video), total=num_frames):
            if args.output:
                output_file.write(vis_frame)
            else:
                cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
                cv2.imshow(basename, vis_frame)
                if cv2.waitKey(1) == 27:
                    break  # esc to quit
        video.release()
        if args.output:
            output_file.release()
        else:
            cv2.destroyAllWindows()


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 Demo")
    parser.add_argument(
        "--config-file",
        default="configs/BoxInst/MS_R_50_3x.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument("--input", nargs="+", help="A list of space separated input images")
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument("--data_root", help="dataset root")
    parser.add_argument("--dataset", default="", help="dataset name")
    parser.add_argument("--split", help="split name")
    parser.add_argument("--image_folder", help="image folder")
    parser.add_argument("--input_dir", default="", help="A list of input frames")
    parser.add_argument("--output_dir", help="output dir")

    parser.add_argument("--tracking", action="store_true", help="Take tracking.")
    parser.add_argument("--tracking_with_depth", action="store_true", help="Take tracking with depth.")
    parser.add_argument("--tracking_with_flow", action="store_true", help="Take tracking with flow.")
    parser.add_argument("--tracking_with_feature", default="")  # vgg19

    parser.add_argument("--tracking_mode", default="DoublePoint+SD")

    parser.add_argument("--instance_txt_out_dir", default="", help="instance_txt output dir")

    parser.add_argument("--box_mode", default="XYXY_ABS")
    parser.add_argument("--mask_tracking_distance", default=200)  
    parser.add_argument("--survival", default=9)  
    

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        nargs="+",
        default=[0.47, 0.58],
        help="Minimum score for instance predictions to be shown",
    )
    # parser.add_argument(
    #     "--confidence-threshold",
    #     type=float,
    #     default=0.3,
    #     help="Minimum score for instance predictions to be shown",
    # )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=["MODEL.WEIGHTS", "models/STC-Seg_MS_R_101_kitti_mots"],
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    if args.instance_txt_out_dir != "":
        if os.path.exists(args.instance_txt_out_dir):
            os.system("rm -rf %s" % args.instance_txt_out_dir)
        os.makedirs(args.instance_txt_out_dir, exist_ok=True)

    if args.input_dir != "":
        cfg = setup_cfg(args)
        demo = VisualizationDemo(cfg)
        image_path = {"train": "training", "val": "training", "test": "testing"}

        seq_folder_list = os.listdir(args.input_dir)
        seq_folder_list = sorted(seq_folder_list)

        for seq_folder in seq_folder_list:
            input_path = os.path.join(args.input_dir, seq_folder)
            output_path = os.path.join(args.output_dir, seq_folder)
            if os.path.exists(output_path):
                os.system("rm -r %s" % output_path)
            os.makedirs(output_path, exist_ok=True)

            args.input = [input_path]
            args.output = output_path

            run(demo, args, image_path, logger, cfg)

    elif args.dataset == "KITTI_MOTS":
        cfg = setup_cfg(args)
        demo = VisualizationDemo(cfg)
        image_path = {"train": "training", "val": "training", "test": "testing"}
        folders_path = os.path.join(args.data_root, args.dataset, image_path[args.split], args.image_folder)

        seq_folder_list = os.listdir(folders_path)

        split_seqmaps = {'val': 'val.seqmap', 'test': 'test.seqmap', 'train': 'train.seqmap'}

        seqmap = []
        seqmap_file = open("%s/%s/splits/%s" % (args.data_root, args.dataset, split_seqmaps[args.split]), 'r')
        for line in seqmap_file:
            fields = line.split(" ")
            seq = "%04d" % int(fields[0])
            seqmap.append(seq)

        seq_folder_list = sorted(seq_folder_list)


        for seq_folder in seq_folder_list:

            if seq_folder not in seqmap:
                continue

            input_path = os.path.join(folders_path, seq_folder)
            output_path = os.path.join(args.output_dir, args.dataset, args.split, args.image_folder, seq_folder)
            if os.path.exists(output_path):
                os.system("rm -r %s" % output_path)
            os.makedirs(output_path, exist_ok=True)

            args.input = [input_path]
            args.output = output_path

            run(demo, args, image_path, logger, cfg)

    elif args.dataset == "YTVIS":
        cfg = setup_cfg(args)
        demo = VisualizationDemo(cfg)

        image_path = {"train": "train", "val": "valid", "test": "test"}
        folders_path = os.path.join(args.data_root, args.dataset, image_path[args.split], args.image_folder)

        seq_folder_list = os.listdir(folders_path)

        seq_folder_list = sorted(seq_folder_list)


        for seq_folder in tqdm.tqdm(seq_folder_list):

            input_path = os.path.join(folders_path, seq_folder)
            output_path = os.path.join(args.output_dir, args.dataset, args.split, args.image_folder, seq_folder)
            if os.path.exists(output_path):
                os.system("rm -r %s" % output_path)
            os.makedirs(output_path, exist_ok=True)

            args.input = [input_path]
            args.output = output_path

            run(demo, args, image_path, logger, cfg)

    else:
        run(demo, args, logger)

    






