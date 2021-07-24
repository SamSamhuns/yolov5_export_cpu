import os
import time
import argparse
from functools import partial

import cv2
import torch
from openvino.inference_engine import IECore

from utils.general import DataStreamer
from utils.detector_utils import save_output, non_max_suppression, preprocess_image


def parse_arguments(desc):
    # Parse Arguments
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-i', '--input_path',
                        required=True,  type=str,
                        help='Path to Input: Video File or Image file')
    parser.add_argument('--model_xml',
                        default='models/yolov5_openvino/yolov5s.xml',
                        help='OpenVINO XML File')
    parser.add_argument('--model_bin',
                        default='models/yolov5_openvino/yolov5s.bin',
                        help='OpenVINO BIN File')
    parser.add_argument('-d', '--target_device',
                        default='CPU', type=str,
                        help='Target Plugin: CPU, GPU, FPGA, MYRIAD, MULTI:CPU,GPU, HETERO:FPGA,CPU')
    parser.add_argument('-m', '--media_type',
                        default='image', type=str,
                        choices=('image', 'video'),
                        help='Type of Input: image, video')
    parser.add_argument('-o', '--output_dir',
                        default='output',  type=str,
                        help='Output directory')
    parser.add_argument('-t', '--threshold',
                        default=0.6,  type=float,
                        help='Object Detection Accuracy Threshold')

    return parser.parse_args()


def get_openvino_core_net_exec(model_xml_path, model_bin_path, target_device="CPU"):
    # load IECore object
    OVIE = IECore()

    # load CPU extensions if availabel
    lib_ext_path = '/opt/intel/openvino/inference_engine/lib/intel64/libcpu_extension.so'
    if 'CPU' in target_device and os.path.exists(lib_ext_path):
        print(f"Loading CPU extensions from {lib_ext_path}")
        OVIE.add_extension(lib_ext_path, "CPU")

    # load openVINO network
    OVNet = OVIE.read_network(
        model=model_xml_path, weights=model_bin_path)

    # create executable network
    OVExec = OVIE.load_network(
        network=OVNet, device_name=target_device)

    return OVIE, OVNet, OVExec


def inference(args) -> None:
    """
    Run Object Detection Application
    """
    print("Running Inference for {}: {}".format(args.media_type, args.input_path))
    # Load Network and Executable
    OVIE, OVNet, OVExec = get_openvino_core_net_exec(
        args.model_xml, args.model_bin, args.target_device)

    # Get Input, Output Information
    InputLayer = next(iter(OVNet.input_info))
    OutputLayer = list(OVNet.outputs)[-1]

    print("Available Devices: ", OVIE.available_devices)
    print("Input Layer: ", InputLayer)
    print("Output Layer: ", OutputLayer)
    print("Model Input Shape: ",
          OVNet.input_info[InputLayer].input_data.shape)
    print("Model Output Shape: ", OVNet.outputs[OutputLayer].shape)

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    start_time = time.time()
    N, C, H, W = OVNet.input_info[InputLayer].input_data.shape
    preprocess_func = partial(preprocess_image, in_size=(W, H))
    data_stream = DataStreamer(args.input_path, args.media_type, preprocess_func)

    for i, (orig_input, model_input) in enumerate(data_stream, start=1):
        # Inference
        start = time.time()
        results = OVExec.infer(inputs={InputLayer: model_input})
        end = time.time()

        inf_time = end - start
        print('Inference Time: {} Seconds Single Image'.format(inf_time))
        fps = 1. / (end - start)
        print('Estimated Inference FPS: {} FPS Single Image'.format(fps))

        fh, fw = orig_input.shape[0:2]
        # Write fos, inference info on Image
        text = 'FPS: {}, INF: {}'.format(round(fps, 2), round(inf_time, 2))
        cv2.putText(orig_input, text, (0, 20), cv2.FONT_HERSHEY_COMPLEX,
                    0.6, (0, 125, 255), 1)

        # Print Bounding Boxes on Image
        detections = results[OutputLayer]
        detections = torch.from_numpy(detections)
        detections = non_max_suppression(
            detections, conf_thres=0.4, iou_thres=0.5, agnostic=False)

        save_path = os.path.join(
            args.output_dir, f"frame_openvino_{str(i).zfill(5)}.jpg")
        save_output(detections[0], orig_input, save_path,
                    threshold=args.threshold, model_in_HW=(H, W),
                    line_thickness=None, text_bg_alpha=0.0)

    elapse_time = time.time() - start_time
    print(f'Total Frames: {i}')
    print(f'Total Elapsed Time: {elapse_time:.3f} Seconds'.format())
    print(f'Final Estimated FPS: {i / (elapse_time):.2f}')


if __name__ == '__main__':
    args = parse_arguments(
        desc="Basic OpenVINO Example for person/object detection")
    inference(args)
