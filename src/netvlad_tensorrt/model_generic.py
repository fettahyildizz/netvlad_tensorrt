"""
MIT License

Copyright (c) 2021 Stephen Hausler, Sourav Garg, Ming Xu,
Michael Milford and Tobias Fischer

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


from torch.cuda import device_count
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms

from netvlad_tensorrt.netvlad import NetVLAD

class Flatten(nn.Module):
    def forward(self, input_data):
        return input_data.view(input_data.size(0), -1)


class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, input_data):
        return F.normalize(input_data, p=2, dim=self.dim)


# Expects RGB , NWHC (opencv format) image
class NetvladGeneric(nn.Module):
    def __init__(self, config, model_path, checkpoint, append_pca_layer=True):
        super().__init__()
        self.config = config
        encoder_dim = 512
        encoder = models.vgg16(pretrained=True)
        layers = list(encoder.features.children())[:-2]
        # only train conv5_1, conv5_2, and conv5_3
        # (leave rest same as Imagenet trained weights)
        for layer in layers[:-5]:
            for p in layer.parameters():
                p.requires_grad = False
        encoder = nn.Sequential(*layers)
        self.nn_model = nn.Module()
        self.nn_model.add_module("encoder", encoder)

        if config["global_params"]["pooling"].lower() == "netvlad":
            net_vlad = NetVLAD(
                num_clusters=int(config["global_params"]["num_clusters"]),
                dim=encoder_dim,
                vladv2=config["global_params"].getboolean("vladv2"),
            )
            self.nn_model.add_module("pool", net_vlad)
        elif config["global_params"]["pooling"].lower() == "max":
            global_pool = nn.AdaptiveMaxPool2d((1, 1))
            self.nn_model.add_module(
                "pool", nn.Sequential(*[global_pool, Flatten(), L2Norm()])
            )
        elif config["global_params"]["pooling"].pooling.lower() == "avg":
            global_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.nn_model.add_module(
                "pool", nn.Sequential(*[global_pool, Flatten(), L2Norm()])
            )
        else:
            raise ValueError(
                "Unknown pooling type: " + config["global_params"]["pooling"].lower()
            )

        if append_pca_layer:
            num_pcs = int(config["global_params"]["num_pcs"])
            netvlad_output_dim = encoder_dim
            if config["global_params"]["pooling"].lower() == "netvlad":
                netvlad_output_dim *= int(config["global_params"]["num_clusters"])

            pca_conv = nn.Conv2d(
                netvlad_output_dim, num_pcs, kernel_size=(1, 1), stride=1, padding=0
            )
            self.nn_model.add_module(
                "WPCA", nn.Sequential(*[pca_conv, Flatten(), L2Norm(dim=-1)])
            )

        self.nn_model.load_state_dict(checkpoint["state_dict"])
        print("=> succesfully loaded checkpoint '{}'".format(model_path))
        if int(config["global_params"]["nGPU"]) > 1 and device_count() > 1:
            self.nn_model.encoder = nn.DataParallel(self.nn_model.encoder)
            self.nn_model.pool = nn.DataParallel(self.nn_model.pool)

    # Expects RGB , NWHC (opencv format) image
    def forward(self, img):
        img = img.permute(0, 3, 1, 2) # Convert from NWHC to NCWH
        it = nn.Sequential(
            transforms.Resize((480, 640)),
            transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
        )
        img = it(img)
        image_encoding = self.nn_model.encoder(img)
        vlad_global = self.nn_model.pool(image_encoding)
        pca_encoding = self.nn_model.WPCA(vlad_global.unsqueeze(-1).unsqueeze(-1))
        return pca_encoding
