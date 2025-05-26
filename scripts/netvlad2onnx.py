#! /usr/bin/python3

import argparse
from os.path import isfile
import configparser
import torch
import numpy as np
import torchvision.transforms as transforms
import torch.onnx
import os
import urllib.request

# import torchvision
from PIL import Image as Image

from src.netvlad_tensorrt.model_generic import NetvladGeneric

class ExportOnnx:
    """Export Onnx Class"""
    def __init__(self, netvlad_model_path, onnx_model_path, config_path):
        self.config_path = config_path
        self.netvlad_model_path = netvlad_model_path
        self.onnx_model_path = onnx_model_path
        self.config = configparser.ConfigParser()
        self.config.read(self.config_path)

        if torch.has_cuda:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        if not isfile(self.netvlad_model_path):
            self.download_all_models(ask_for_permission=False)

        if isfile(self.netvlad_model_path):
            print("=> loading checkpoint '{}'".format(self.netvlad_model_path))
            self.checkpoint = torch.load(
                self.netvlad_model_path, map_location=lambda storage, loc: storage
            )
            assert self.checkpoint["state_dict"]["WPCA.0.bias"].shape[0] == int(
                self.config["global_params"]["num_pcs"]
            )
            self.config["global_params"]["num_clusters"] = str(
                self.checkpoint["state_dict"]["pool.centroids"].shape[0]
            )

            self.model = NetvladGeneric(
                self.config, self.netvlad_model_path, self.checkpoint, append_pca_layer=True
            ).to(self.device)
            self.model.eval()
        
            dummy_input = np.ones((720, 1280, 3))
            realtime_image_pil = Image.fromarray((dummy_input * 255).astype(np.uint8))

            it = transforms.Compose(
                [
                    transforms.ToTensor(),
                ]
            )

            # Model expects RGB, 640x480, NWHC (opencv format)            
            tensor_image = it(realtime_image_pil).unsqueeze(0)
            tensor_image = tensor_image.to(self.device)
            
            tensor_image = tensor_image.permute(0, 2, 3, 1)

            torch.onnx.export(
                self.model,
                tensor_image,
                self.onnx_model_path, 
                True,
                # opset_version=11,  # the ONNX version to export the model to
            )
    
    def download_all_models(self, ask_for_permission=False):
        print("Downloading to: " + self.netvlad_model_path)

        if not os.path.isfile(self.netvlad_model_path):
            print("Downloading mapillary_WPCA4096.pth.tar")
            os.mkdir(self.netvlad_model_path)
            urllib.request.urlretrieve(
                "https://cloudstor.aarnet.edu.au/plus/s/ZgW7DMEpeS47ELI/download",
                self.netvlad_model_path
            )

        print("Downloaded all pretrained models.")

def main(args):
    ExportOnnx(args.model,
                args.onnx,
                args.config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a Netvlad model to ONNX format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "-m", "--model",
        default="../models/mapillary_WPCA4096.pth.tar",
        help="Path to the input PyTorch NetVLAD model checkpoint (.pth or .pt file)"
    )
    
    parser.add_argument(
        "-o", "--onnx", 
        default="../models/netvlad.onnx",
        help="Output path for the converted ONNX model file (.onnx extension)"
    )

    parser.add_argument(
        "-c", "--config", 
        default="../config/speed.ini",
        help="Path to NetVLAD configuration file (.ini or .yaml or .json) containing model parameters"
    )

    args = parser.parse_args()

    main(args)
