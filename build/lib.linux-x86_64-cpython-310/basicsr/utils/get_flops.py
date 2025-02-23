import torch
from basicsr.models.archs.restormer_arch import Restormer
# from basicsr.models.archs.dehazevmamba_convnext_arch import DehazeVmambaConvNext
from basicsr.models.archs.sota.res2tormer import Res2tormer
from utils_modelsummary import get_model_activation, get_model_flops


# Res2tormer
#     #Activations : 137.9205 [M]
#          #Conv2d : 173
#            FLOPs : 25.0840 [G]
#          #Params : 8.3500 [M]
# PVT
    #Activations : 90.5042 [M]
         #Conv2d : 113
           # FLOPs : 42.3132 [G]
         #Params : 15.1995 [M]
# GDA
    #Activations : 148.2424 [M]
         #Conv2d : 137
           # FLOPs : 24.4293 [G]
    #      #Params : 8.1419 [M]
# LeFF
    # #Activations : 89.1617 [M]
    #      #Conv2d : 95
    #        FLOPs : 23.1038 [G]
#          #Params : 1.9445 [M]
# GDFN
# #     #Activations : 196.5097 [M]
#            #Conv2d : 119
#              FLOPs : 9.3711 [G]
           #Params : 2.4555 [M]
# # MLP
#     #Activations : 56.1316 [M]
#          #Conv2d : 83
#            FLOPs : 3.3417 [G]
         #Params : 1.9042 [M]
# model = Restormer().cuda()
model = Restormer().cuda()
model = Res2tormer(
    out_chans=3,
    embed_dims=[24, 48, 96, 192, 96, 48, 24],
    num_heads=[1, 2, 4, 8, 4, 2, 1],
    depths=[1, 2, 2, 2, 2, 2, 1],
    bias=False,
    LayerNorm_type='WithBias',
    ffn_expansion_factor=[1, 4],
    num_blocks=[[1, 1], [1, 1], [1, 2], [1, 2], [1, 2], [1, 1], [1, 1]],
    att_types='Res2Net',
    mlp_types='MLP').cuda()
with torch.no_grad():
    input_dim = (3, 256, 256)  # set the input dimension
    activations, num_conv2d = get_model_activation(model, input_dim)
    print('{:>16s} : {:<.4f} [M]'.format('#Activations', activations/10**6))
    print('{:>16s} : {:<d}'.format('#Conv2d', num_conv2d))
    flops = get_model_flops(model, input_dim, False)
    print('{:>16s} : {:<.4f} [G]'.format('FLOPs', flops/10**9))
    num_parameters = sum(map(lambda x: x.numel(), model.parameters()))
    print('{:>16s} : {:<.4f} [M]'.format('#Params', num_parameters/10**6))