import torch
import torch.nn as nn
import torch.nn.functional as F
from trec.conv_layer import Conv2d_TREC

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, params_L=None, params_H=None, trec=None, layer_idx=0):
        super().__init__()
        self.conv1 = Conv2d_TREC(in_channels, out_channels, 3, stride=2, padding=1,
                                param_L=params_L[0], param_H=params_H[0], layer=layer_idx) if trec[0] else \
                    nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)
        
        self.conv2 = Conv2d_TREC(out_channels, out_channels, 3, stride=1, padding=1,
                                param_L=params_L[1], param_H=params_H[1], layer=layer_idx+1) if trec[1] else \
                    nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1)
        
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.norm2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        return F.relu(x)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, params_L=None, params_H=None, trec=None, layer_idx=0):
        super().__init__()
        self.conv1 = Conv2d_TREC(in_channels, out_channels, 3, stride=1, padding=1,
                                param_L=params_L[0], param_H=params_H[0], layer=layer_idx) if trec[0] else \
                    nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)
        
        self.conv2 = Conv2d_TREC(out_channels, out_channels, 3, stride=1, padding=1,
                                param_L=params_L[1], param_H=params_H[1], layer=layer_idx+1) if trec[1] else \
                    nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1)
        
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.norm2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv1(x)
        x = self.norm1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        return F.relu(x)

class AutoencoderTREC(nn.Module):
    def __init__(self, params_L=None, params_H=None, trec=None):
        super().__init__()
        ch_mult = [1, 2, 4, 8]  # Channel multipliers for each level
        base_ch = 64  # Base number of channels
        
        # Calculate number of TREC layers
        num_encoder_layers = len(ch_mult) * 2  # 2 convs per block
        num_decoder_layers = num_encoder_layers
        num_trec_layers = num_encoder_layers + num_decoder_layers
        
        # Validate TREC parameters
        if trec and any(trec):
            assert params_L is not None and params_H is not None, \
                "params_L and params_H must be provided when using TREC"
            assert len(params_L) == num_trec_layers, \
                f"params_L must have {num_trec_layers} elements"
            assert len(params_H) == num_trec_layers, \
                f"params_H must have {num_trec_layers} elements"
            assert len(trec) == num_trec_layers, \
                f"trec must have {num_trec_layers} elements"
        else:
            params_L = [None] * num_trec_layers
            params_H = [None] * num_trec_layers
            trec = [False] * num_trec_layers

        # Split parameters for encoder and decoder
        encoder_params = {
            'params_L': params_L[:num_encoder_layers],
            'params_H': params_H[:num_encoder_layers],
            'trec': trec[:num_encoder_layers]
        }
        
        decoder_params = {
            'params_L': params_L[num_encoder_layers:],
            'params_H': params_H[num_encoder_layers:],
            'trec': trec[num_encoder_layers:]
        }

        # Encoder
        self.encoder_blocks = nn.ModuleList()
        in_ch = 3
        layer_idx = 0
        for i, mult in enumerate(ch_mult):
            out_ch = base_ch * mult
            self.encoder_blocks.append(
                EncoderBlock(in_ch, out_ch, 
                            params_L=encoder_params['params_L'][i*2:(i+1)*2],
                            params_H=encoder_params['params_H'][i*2:(i+1)*2],
                            trec=encoder_params['trec'][i*2:(i+1)*2],
                            layer_idx=layer_idx)
            )
            in_ch = out_ch
            layer_idx += 2

        # Decoder
        self.decoder_blocks = nn.ModuleList()
        layer_idx = num_encoder_layers
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = base_ch * mult if i > 0 else base_ch
            self.decoder_blocks.append(
                DecoderBlock(in_ch, out_ch,
                            params_L=decoder_params['params_L'][i*2:(i+1)*2],
                            params_H=decoder_params['params_H'][i*2:(i+1)*2],
                            trec=decoder_params['trec'][i*2:(i+1)*2],
                            layer_idx=layer_idx)
            )
            in_ch = out_ch
            layer_idx += 2

        # Final convolution
        self.final_conv = nn.Conv2d(base_ch, 3, 3, padding=1)

    def encode(self, x):
        """Encode input to latent representation"""
        for block in self.encoder_blocks:
            x = block(x)
        return x
    
    def decode(self, x):
        """Decode latent representation to reconstruction"""
        for block in self.decoder_blocks:
            x = block(x)
        return torch.tanh(self.final_conv(x))
    
    def forward(self, x):
        """Full forward pass: encode then decode"""
        x = self.encode(x)
        x = self.decode(x)
        return x

    def get_last_layer(self):
        return self.final_conv.weight