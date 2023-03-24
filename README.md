# YOLOv5 CPU Export and OpenVINO Inference

[![Codacy Badge](https://app.codacy.com/project/badge/Grade/55c3d2e474f14e7b8cb6c611504457d9)](https://www.codacy.com/gh/SamSamhuns/yolov5_export_cpu/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=SamSamhuns/yolov5_export_cpu&amp;utm_campaign=Badge_Grade)

Documentation on exporting YOLOv5 models for fast CPU inference using Intel's OpenVINO framework (Tested on commits up to June 6, 2022 in docker).

## Google Colab Conversion

Convert yolov5 model to IR format with Google Colab. [![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1K8gnZEka47Gbcp1eJbBaSe3GxngJdvio?usp=sharing) (Recommended)

## 1. Clone and set up the Official YOLOv5 GitHub repository

<details>
  <summary>Setup</summary>

  All package installations should be done in a virtualenv or conda env to prevent package conflict errors.

-   Install required requirements for onnx and openvino Inference

```bash
pip install --upgrade pip
pip install -r inf_requirements.txt
```

-   Clone and install requirements for yolov5 repository

```bash
git clone https://github.com/ultralytics/yolov5                    # clone repo
cd yolov5
pip install -r requirements.txt                                    # base requirements
```

</details>

## 2. Export a Trained YOLOv5 Model as ONNX

<details>
  <summary>Export</summary>

Export a pre-trained or custom trained YOLOv5 model to generate the respective ONNX, TorchScript and CoreML formats of the model. The pre-trained `yolov5s.pt` is the lightest and fastest model for CPU inference. Other slower but more accurate models include `yolov5m.pt, yolov5l.pt` and `yolov5x.pt`. All available model details at Ultralytics YOLOv5 [README](https://github.com/ultralytics/yolov5#pretrained-checkpoints).

A custom training checkpoint i.e. `runs/exp/weights/best.pt` can be used for conversion as well.

-   Export a pre-trained light yolov5s.pt model at 640x640 with batch size 1

```bash
python export.py --weights yolov5s.pt --include onnx --img 640 --batch 1
```

-   Export a custom checkpoint for dynamic input shape {BATCH_SIZE, 3, HEIGHT, WIDTH}. Note, for CPU inference mode, BATCH_SIZE must be set to 1. Install onnx-simplifier for simplifying onnx exports

```bash
pip install onnx-simplifier==0.3.10                                
python export.py --weights runs/exp/weights/best.pt --include onnx  --dynamic --simplify
```

-  Cd to `yolov5_export_cpu` dir and move the onnx model to `yolov5_export_cpu/models` directory

```bash
mv <PATH_TO_ONNX_MODEL> yolov5_export_cpu/models/
```

</details>

## 3. Test YOLOv5 ONNX model inference

<details>
  <summary>ONNX inference</summary>

```bash
python detect_onnx.py -m image -i <IMG_FILE_PATH/IMG_DIR_PATH>
python detect_onnx.py -m video -i <VID_PATH_FILE>
# python detect_onnx.py -h for more info
```

Optional: To convert the all frames in the `output` directory into a mp4 video using `ffmpeg`, use `ffmpeg -r 25 -start_number 00001 -i output/frame_onnx_%5d.jpg -vcodec libx264 -y -an onnx_result.mp4`

</details>

## 4. Export ONNX to OpenVINO

**Recommended Option A**

### Option A. Use OpenVINO's python dev library

<details>
  <summary> A1. Install OpenVINO python dev library</summary>

  Instructions for setting OpenVINO available [here](https://docs.openvino.ai/latest/openvino_docs_install_guides_install_dev_tools.html)

```bash
# install required OpenVINO lib to convert ONNX to OpenVINO IR
pip install openvino-dev[onnx]
```

</details>

<details>
  <summary> A2. Export ONNX to OpenVINO IR</summary>

This will create the OpenVINO Intermediate Model Representation (IR) model files (xml and bin) in the directory `models/yolov5_openvino`.

**Important Note:** --input_shape must be provided and match the img shape used to export ONNX model. Batching might not supported for CPU inference

```bash
# export onnx to OpenVINO IR
mo \
  --progress \
  --input_shape [1,3,640,640] \
  --input_model models/yolov5s.onnx \
  --output_dir models/yolov5_openvino \
  --data_type half # {FP16, FP32, half, float}
```

[Full OpenVINO export options](https://docs.openvinotoolkit.org/latest/openvino_docs_MO_DG_prepare_model_convert_model_Converting_Model_General.html)

</details>

### Option B. Use OpenVINO Docker

<details>
  <summary>B1. Download Docker and OpenVINO Docker Image</summary>

[Install docker](https://docs.docker.com/get-docker/) in your system if not already installed.

Pass the docker run command below in a terminal which will automatically download the OpenVINO Docker Image and run it. The `models` directory containing the ONNX model must be in the current working directory.

```bash
docker run -it --rm \
            -v $PWD/models:/home/openvino/models \
            openvino/ubuntu18_dev:latest \
            /bin/bash -c "cd /home/openvino/; bash"
```

</details>

<details>
  <summary>B2. Export ONNX model to an OpenVINO IR representation</summary>

This will create the OpenVINO Intermediate Model Representation (IR) model files (xml and bin) in the directory `models/yolov5_openvino` which will be available in the host system outside the docker container.

**Important Note:** --input_shape must be provided and match the img shape used to export ONNX model. Batching might not supported for CPU inference

```bash
# inside the OpenVINO docker container
mo \
  --progress \
  --input_shape [1,3,640,640] \
  --input_model models/yolov5s.onnx \
  --output_dir models/yolov5_openvino \
  --data_type half # {FP16, FP32, half, float}
# exit OpenVINO docker container
exit  
```

[Full OpenVINO export options](https://docs.openvinotoolkit.org/latest/openvino_docs_MO_DG_prepare_model_convert_model_Converting_Model_General.html)

</details>

## 5. Test YOLOv5 OpenVINO IR model CPU inference

<details>
  <summary>OpenVINO model inference</summary>

```bash
python detect_openvino.py -m image -i <IMG_FILE_PATH/IMG_DIR_PATH>
python detect_openvino.py -m video -i <VID_PATH_FILE>
# python detect_openvino.py -h for more info
```

Optional: To convert the all frames in the `output` directory into a mp4 video using `ffmpeg`, use `ffmpeg -r 25 -start_number 00001 -i output/frame_openvino_%5d.jpg -vcodec libx264 -y -an openvino_result.mp4`

</details>
