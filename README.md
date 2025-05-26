# netvlad_tensorrt
Netvlad model which is convertible to ONNX and Tensorrt engine. 
* This repo was written due to real-time performance constraints I have faced when I had a project regarding image retrieval problem. I was working on Jetson Xavier NX, which is a humble device, therefore I needed TRT engine version of Netvlad model. I had around 15 fps when I was inferencing with TRT engines, using 4096 features and speed.ini config file, on Jetson Xavier NX.

* Default config files are provided in config folder. I prefer speed.ini for the obvious concern. Default onnx and pth files are also provided in models folder.

* If the checkpoint file of Netvlad model is not found, it automatically downloads.

## 🛠️ Installation

### Quick Setup
```bash
# Clone the repository
git clone https://github.com/your-username/netvlad_tensorrt.git
cd netvlad_tensorrt

# Install in development mode
pip install -e .
```

### Environment Requirements
```bash
pip install torch==1.12.0 torchvision==0.13.0
pip install onnx==1.18.0 onnxruntime==1.12.0
pip install numpy==1.23.2 setuptools==45.2.0
pip install protobuf==3.6.1 onnxscript==0.2.6
```
> **Note**: The checkpoint file will be automatically downloaded if not found locally.

## 🔄 Model Conversion
### Step 1: To Onnx
To convert from pytorch to ONNX, please run given command,
##### Basic conversion with default paths

`python3 scripts/netvlad2onnx.py`
  
  
##### Specify output directory
```
python3 scripts/netvlad2onnx.py  \
    -m models/mapillary_WPCA4096.pth.tar \
    -o models/netvlad.onnx \
    -c config/speed.ini
```
These parameters are default. You may configure with respected to your models. If model path is not provided, it would download from [this link](https://cloudstor.aarnet.edu.au/plus/s/ZgW7DMEpeS47ELI/download)

### Step 2: ONNX To Tensorrt
```bash
/usr/src/tensorrt/bin/trtexec \
    --saveEngine=models/netvlad.trt \
    --onnx=models/netvlad.onnx
```

## 📋 Input Specifications

### Image Format Requirements
- **Color Space**: RGB (not BGR/OpenCV format)
- **Layout**: NHWC (Height-Width-Channel) format expected
- **Conversion**: Automatically converts NHWC → NCHW for TensorRT compatibility
- **Preprocessing**: Built-in normalization and resizing


### Example Input Preparation
```python
import cv2
import numpy as np
import torchvision.transforms as transforms

# Load and convert image
image = cv2.imread('input.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
image_rgb = Image.fromarray(image_rgb)
    
it = transforms.Compose(
    [
        # Torchscript does not support ToTensor()
        transforms.ToTensor(),

    ]
)
image_rgb = it(image_rgb).unsqueeze(0)

image_rgb = image_rgb.permute(0, 2, 3, 1)

```

This netvlad model is configured to work with specific image format for my use case. The input layer expects **RGB , NWHC (opencv format) image**. TensorRT requires NCHW image format so it intrinsically converts data from NWHC to NCHW in forward() method. Netvlad also expects RGB color space (not opencv format), so be aware of that.

> This conversions are tested in **Tensorrt 8.5.3** and **Tensorrt 8.4.1.5.** environment.

## TODO:
* I may add inference code.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.