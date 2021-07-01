# YOLOv5 CPU Export and OpenVINO Inference

Documentation on exporting YOLOv5 models for fast CPU inference using Intel's OpenVINO framework

## 1. Clone and set up the Official YOLOv5 GitHub repository

<details>
  <summary>Setup</summary>

  All package installations should be done in a virtualenv or conda env to prevent package conflict errors.

```bash
$ git clone https://github.com/ultralytics/yolov5                    # clone repo
$ cd yolov5
$ pip install -r requirements.txt                                    # base requirements
$ pip install "coremltools>=4.1" "onnx>=1.9.0" scikit-learn==0.19.2  # export requirements
```

</details>

## 2. Export a Trained YOLOv5 Model as ONNX

<details>
  <summary>Export</summary>

Export a pre-trained or custom trained YOLOv5 model to generate the respective ONNX, TorchScript and CoreML formats of the model. The pre-trained `yolov5s.pt` is the lightest and fastest model for CPU inference. Other slower but more accurate models include `yolov5m.pt, yolov5l.pt` and `yolov5x.pt`. All available model details at Ultralytics YOLOv5 [README](https://github.com/ultralytics/yolov5#pretrained-checkpoints).

A custom training checkpoint i.e. `runs/exp/weights/best.pt` can be used for conversion as well.

```bash
# export a pre-trained light yolov5s.pt model at 640x640 with batch size 1
$ python export.py --weights yolov5s.pt --img 640 --batch 1
# export a custom checkpoint for dynamic input shape {BATCH_SIZE, 3, HEIGHT, WIDTH}
# Note, for CPU inference mode, BATCH_SIZE must be set to 1
$ python export.py --weights runs/exp/weights/best.pt --dynamic --simplify
```

Move the onnx model to `models` directory

```bash
$ mv <PATH_TO_ONNX_MODEL> models/
```

Install required requirements for onnx and openvino Inference

```bash
$ pip install -r inf_requirements.txt
```

</details>

## 3. Test YOLOv5 ONNX model inference

<details>
  <summary>ONNX inference</summary>

```bash
$ python detect_onnx.py -m image -i <IMG_FILE_PATH/IMG_DIR_PATH>
$ python detect_onnx.py -m video -i <VID_PATH_FILE>
# python detect_onnx.py -h for more info
```

Optional: To convert the all frames in the `output` directory into a mp4 video using `ffmpeg`, use `ffmpeg -r 25 -start_number 00001 -i output/frame_onnx_%5d.jpg -vcodec libx264 -y -an onnx_result.mp4`

</details>

## 4. Download Docker and OpenVINO Docker Image

<details>
  <summary>OpenVINO Docker</summary>

[Install docker](https://docs.docker.com/get-docker/) in your system if not already installed.

Pass the docker run command below in a terminal which will automatically download the OpenVINO Docker Image and run it. The `models` directory containing the ONNX model must be in the current working directory.

```bash
docker run -it --rm \
            -v $PWD/models:/home/openvino/models \
            openvino/ubuntu18_dev:latest \
            /bin/bash -c "cd /home/openvino/; bash"
```

</details>

## 5. Export ONNX model to an OpenVINO IR representation

<details>
  <summary>OpenVINO IR Export</summary>

This will create the OpenVINO Intermediate Model Representation (IR) model files (xml and bin) in the directory `models/yolov5_openvino` which will be available in the host system outside the docker container.

```bash
# inside the openvino docker container
$ python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo.py \
          --progress \
          --input_model models/yolov5s.onnx \
          --output_dir models/yolov5_openvino \
          --data_type half # {FP16,FP32,half,float}
# exit openvino docker container
$ exit  
```

[Full OpenVINO export options](https://docs.openvinotoolkit.org/latest/openvino_docs_MO_DG_prepare_model_convert_model_Converting_Model_General.html)

</details>

## 6. Test YOLOv5 OpenVINO IR model CPU inference

<details>
  <summary>OpenVINO model inference</summary>

```bash
$ python detect_openvino.py -m image -i <IMG_FILE_PATH/IMG_DIR_PATH>
$ python detect_openvino.py -m video -i <VID_PATH_FILE>
# python detect_openvino.py -h for more info
```

Optional: To convert the all frames in the `output` directory into a mp4 video using `ffmpeg`, use `ffmpeg -r 25 -start_number 00001 -i output/frame_openvino_%5d.jpg -vcodec libx264 -y -an openvino_result.mp4`

</details>
