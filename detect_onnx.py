import os
import time
from functools import partial

import torch
import numpy as np
import onnxruntime

from utils.general import parse_arguments, DataStreamer
from utils.detector_utils import save_output, preprocess_image, non_max_suppression, w_non_max_suppression


@torch.no_grad()
def detect_onnx(src_path: str,
                media_type: str,
                threshold: float = 0.6,
                official: bool = True,
                onnx_path: str = "models/yolov5s.onnx",
                output_dir: str = "output",
                num_classes: int = 80) -> None:
    session = onnxruntime.InferenceSession(onnx_path)
    model_batch_size = session.get_inputs()[0].shape[0]
    model_h = session.get_inputs()[0].shape[2]
    model_w = session.get_inputs()[0].shape[3]
    in_w = 640 if (model_w is None or isinstance(model_w, str)) else model_w
    in_h = 640 if (model_h is None or isinstance(model_h, str)) else model_h
    print("Input Layer: ", session.get_inputs()[0].name)
    print("Output Layer: ", session.get_outputs()[0].name)
    print("Model Input Shape: ", session.get_inputs()[0].shape)
    print("Model Output Shape: ", session.get_outputs()[0].shape)

    start_time = time.time()
    preprocess_func = partial(preprocess_image, in_size=(in_w, in_h))
    data_stream = DataStreamer(src_path, media_type, preprocess_func)
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    for i, (orig_input, model_input) in enumerate(data_stream, start=1):
        batch_size = model_input.shape[0] if isinstance(
            model_batch_size, str) else model_batch_size
        input_name = session.get_inputs()[0].name

        # inference
        start = time.time()
        outputs = session.run(None, {input_name: model_input})
        end = time.time()

        inf_time = end - start
        print('Inference Time: {} Seconds Single Image'.format(inf_time))
        fps = 1. / (end - start)
        print('Estimated Inference FPS: {} FPS Single Image'.format(fps))

        batch_detections = []
        # model.model[-1].export = boolean ---> True:3 False:4
        if official:  # recommended
            # model.model[-1].export = False ---> outputs[0] (1, xxxx, 85)
            # Use the official code directly
            batch_detections = torch.from_numpy(np.array(outputs[0]))
            batch_detections = non_max_suppression(
                batch_detections, conf_thres=0.4, iou_thres=0.5, agnostic=False)
        else:
            # model.model[-1].export = False ---> outputs[1]/outputs[2]/outputs[2]
            # model.model[-1].export = True  ---> outputs
            # (1, 3, 20, 20, 85)
            # (1, 3, 40, 40, 85)
            # (1, 3, 80, 80, 85)
            # same anchors for 5s, 5l, 5x
            anchors = [[116, 90, 156, 198, 373, 326], [
                30, 61, 62, 45, 59, 119], [10, 13, 16, 30, 33, 23]]

            boxs = []
            a = torch.tensor(anchors).float().view(3, -1, 2)
            anchor_grid = a.clone().view(3, 1, -1, 1, 1, 2)
            if len(outputs) == 4:
                outputs = [outputs[1], outputs[2], outputs[3]]
            for index, out in enumerate(outputs):
                out = torch.from_numpy(out)
                # batch = out.shape[1]
                feature_w = out.shape[2]
                feature_h = out.shape[3]

                # Feature map corresponds to the original image zoom factor
                stride_w = int(in_w / feature_w)
                stride_h = int(in_h / feature_h)

                grid_x, grid_y = np.meshgrid(
                    np.arange(feature_w), np.arange(feature_h))

                # cx, cy, w, h
                pred_boxes = torch.FloatTensor(out[..., :4].shape)
                pred_boxes[..., 0] = (torch.sigmoid(
                    out[..., 0]) * 2.0 - 0.5 + grid_x) * stride_w  # cx
                pred_boxes[..., 1] = (torch.sigmoid(
                    out[..., 1]) * 2.0 - 0.5 + grid_y) * stride_h  # cy
                pred_boxes[..., 2:4] = (torch.sigmoid(
                    out[..., 2:4]) * 2) ** 2 * anchor_grid[index]  # wh

                conf = torch.sigmoid(out[..., 4])
                pred_cls = torch.sigmoid(out[..., 5:])

                output = torch.cat((pred_boxes.view(batch_size, -1, 4),
                                    conf.view(batch_size, -1, 1),
                                    pred_cls.view(batch_size, -1, num_classes)),
                                   -1)
                boxs.append(output)

            outputx = torch.cat(boxs, 1)
            # NMS
            batch_detections = w_non_max_suppression(
                outputx, num_classes, conf_thres=0.4, nms_thres=0.3)
        if output_dir is not None:
            save_path = os.path.join(
                output_dir, f"frame_onnx_{str(i).zfill(5)}.jpg")
            save_output(batch_detections[0], orig_input, save_path,
                        threshold=threshold, model_in_HW=(in_h, in_w),
                        line_thickness=None, text_bg_alpha=0.0)

    elapse_time = time.time() - start_time
    print(f'Total Frames: {i}')
    print(f'Total Elapsed Time: {elapse_time:.3f} Seconds'.format())
    print(f'Final Estimated FPS: {i / (elapse_time):.2f}')


if __name__ == '__main__':
    args = parse_arguments("YoloV5 onnx demo")
    t1 = time.time()
    detect_onnx(src_path=args.input_path,
                media_type=args.media_type,
                threshold=args.threshold,
                official=True,  # official yolov5 post-processing
                onnx_path=args.onnx_path,
                output_dir=args.output_dir,
                num_classes=args.num_classes)
