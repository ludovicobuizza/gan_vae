import torch.nn as nn


def make_module(conv_layer,
                 hyper_params,
                 activation=nn.ReLU):
    modules = []
    in_channels = hyper_params["in_channels"]
    for i in range(len(hyper_params["hidden_channels"])):
        module = nn.Sequential(
            conv_layer(
                in_channels=in_channels,
                out_channels=hyper_params["hidden_channels"][i],
                kernel_size=hyper_params["hidden_channels"][i],
                stride=hyper_params["strides"][i],
                padding=hyper_params["paddings"][i]
            ),
            nn.BatchNorm2d(num_features=hyper_params["hidden_channels"][i]),
            activation(),
        )
        in_channels = hyper_params["hidden_channels"][i]
        modules.append(module)
    return nn.Sequential(*modules)


def make_final_decoder_layer(decoder_hyper_params):
    final_layer = nn.Sequential(
        nn.ConvTranspose2d(
            in_channels=decoder_hyper_params["hidden_channels"][-1],
            out_channels=decoder_hyper_params["out_channels"],
            kernel_size=decoder_hyper_params["final_kernel"],
            stride=decoder_hyper_params["final_stride"],
            padding=decoder_hyper_params["final_padding"],
            output_padding=decoder_hyper_params["final_output_padding"],
        ),
        nn.BatchNorm2d(num_features=decoder_hyper_params["out_channels"]),
        nn.LeakyReLU(),
        nn.Conv2d(
            in_channels=decoder_hyper_params["out_channels"],
            out_channels=decoder_hyper_params["out_channels"],
            kernel_size=1,
            stride=1,
            padding=0,
        ),
        nn.Sigmoid(),
    )
    return final_layer