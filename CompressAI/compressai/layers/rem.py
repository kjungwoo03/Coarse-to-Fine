import torch.nn as nn
from .layers import conv3x3, conv1x1
from torch.nn import Sequential as Seq
import torch

class ResidualBlockSmall(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch)

  
        self.nonlin = nn.LeakyReLU(inplace=True)

        
        #self.conv2 = conv3x3(out_ch, out_ch)

        if in_ch != out_ch:
            self.skip = conv1x1(in_ch, out_ch)
        else:
            self.skip = None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.nonlin(out)
        #out = self.conv2(out)
        #out = self.nonlin(out)

        if self.skip is not None:
            identity = self.skip(x)

        out = out + identity
        return out


class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch)

  
        self.nonlin = nn.LeakyReLU(inplace=True)

        
        self.conv2 = conv3x3(out_ch, out_ch)

        if in_ch != out_ch:
            self.skip = conv1x1(in_ch, out_ch)
        else:
            self.skip = None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.nonlin(out)
        out = self.conv2(out)
        out = self.nonlin(out)

        if self.skip is not None:
            identity = self.skip(x)

        out = out + identity
        return out
    


class LatentRateReduction(nn.Module):


        def __init__(self, dim_chunk = 32, mu_std = True, dimension = "middle" ):
            super().__init__()
            self.dim_block = dim_chunk

            self.mu_std = mu_std
            N = self.dim_block



            if dimension != "big": #dd
                self.enc_base_entropy_params = Seq(
                    ResidualBlock(2 * N ,  N),
                    ResidualBlock( N, N),
                    )
                
                self.enc_progressive_entropy_params = Seq(
                    ResidualBlock(2 * N if self.mu_std else N ,  N),
                    ResidualBlock( N, N),
                    )

                self.enc_base_rep = Seq(
                    ResidualBlock(N, N),
                    ResidualBlock(N, N),
                    )

                self.enc = Seq(
                    ResidualBlock(3 * N, 2 * N),
                    ResidualBlock(2 * N, 2 * N),
                    ResidualBlock(2 * N, 2 * N if self.mu_std else N),
                )
            else:
                self.enc_base_entropy_params = Seq(
                    ResidualBlock(2 * N ,  N),
                    ResidualBlock( N, N),
                    ResidualBlock( N, N),
                    )
                
                self.enc_progressive_entropy_params = Seq(
                    ResidualBlock(2 * N if self.mu_std else N ,  N),
                    ResidualBlock( N, N),
                    ResidualBlock( N, N),
                    )            

                self.enc_base_rep = Seq(
                    ResidualBlock(N, N),
                    ResidualBlock(N, N),
                    ResidualBlock(N, N),
                    )   

                self.enc = Seq(
                    ResidualBlock(3 * N, 2 * N),
                    ResidualBlock(2 * N, 2 * N),
                    ResidualBlock(2 * N, 2 * N),
                    ResidualBlock(2 * N, 2 * N if self.mu_std else N),
                )           



        def forward(self, x_base, entropy_params_base, entropy_params_prog, att_mask):

            identity = entropy_params_prog
            f_latent = self.enc_base_rep(x_base)
            f_ent_prog = self.enc_progressive_entropy_params(entropy_params_prog) 
            f_ent_base = self.enc_base_entropy_params(entropy_params_base)            

            ret = self.enc(torch.cat([f_latent, f_ent_base, f_ent_prog], dim=1)) #fff
            ret = ret*att_mask
            #ret = self.enc(torch.cat([f_latent, f_ent_prog], dim=1)) 
            res =  identity + ret
            return res