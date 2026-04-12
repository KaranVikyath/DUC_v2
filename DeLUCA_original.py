import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from custom_funcs import convert_nan
import torch.nn.init as init

class DeLUCA(nn.Module):
    def __init__(self, input_shape, flat_layer_size, enc_layer_size, deco_layer_size,  
                 kernel_size, output_padding, lr, K, rank, reg_const1=1.0, reg_const2=1.0,
                 batch_size=200, model_path=None, logs_path="", 
                 cluster_model= "CFS",device="CPU"):
        super(DeLUCA, self).__init__()

        # Initialize attributes
        self.input_shape = input_shape
        self.n_features = input_shape[1]
        self.model_path = model_path
        self.iter = 0
        self.batch_size = batch_size
        self.reg1 = reg_const1
        self.reg2 = reg_const2
        self.lr = lr
        self.kernel_size = kernel_size
        self.flat_layer_size = flat_layer_size
        self.enc_layer_size = enc_layer_size
        self.deco_layer_size = deco_layer_size
        self.cluster_model = cluster_model
        self.device = device
        self.rank = rank
        # Define layers
        self.pseudo = PseudoCompletion(input_shape, flat_layer_size)
        self.encoder = Encoder(input_shape, enc_layer_size, kernel_size)

        if self.cluster_model == "CFS": # If LLR is true, LLR module is implemented
            self.CFS_module = CFSModule(rank)
        elif self.cluster_model == "SSC":
            self.self_expressive_module = SelfExpressiveModule(batch_size)

        self.decoder = Decoder(enc_layer_size[-1], deco_layer_size, kernel_size, output_padding)

        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.9, patience=10, verbose=False)

        # Tensorboard Summary Writer
        self.summary_writer = SummaryWriter(logs_path)

    def forward(self, x):
        x = torch.tensor(x).to(self.device)
        Xc = self.pseudo(x)
        Z = self.encoder(Xc)
        Z_flat = Z.view(self.batch_size,-1)

        if self.cluster_model=="CFS":
            PZ, Coef = self.CFS_module(Z_flat)
        elif self.cluster_model=="SSC":
            PZ, Coef = self.self_expressive_module(Z_flat)

        PZ = PZ.view(Z.shape)
        decoded = self.decoder(PZ)
        
        # Compute reconstruction loss
        x_omega, mask_tensor = convert_nan(x)
        x_tilde_omega = Xc * mask_tensor
        x_omega_hat = decoded * mask_tensor

        reconstruction_loss= torch.norm((x_tilde_omega - x_omega), p='fro') + torch.norm((x_tilde_omega - x_omega_hat), p='fro') + torch.norm((x_omega_hat - x_omega), p='fro')
        # reconstruction_loss= 0.5 * torch.norm((x_omega_hat - x_omega), p='fro')
        autoencoder_loss = 0.5 * torch.norm(Z - PZ, p='fro')

        if self.cluster_model=="CFS":
            total_loss = reconstruction_loss
        else:
            coefficient_loss = torch.norm(Coef, p=2, dim=(0,1))
            total_loss = autoencoder_loss * self.reg2 + self.reg1 * coefficient_loss + reconstruction_loss
        
        return decoded, total_loss, autoencoder_loss, reconstruction_loss

    def finetune_fit(self, x):
        self.optimizer.zero_grad()
        decoded, total_loss, autoencoder_loss, reconstruction_loss = self.forward(x)
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        total_loss.backward()
        self.optimizer.step()
        
        # Step the scheduler
        self.scheduler.step(total_loss)
        # Logging
        with torch.no_grad():
            if self.cluster_model=="CFS":
                C = self.CFS_module.Coef.detach().cpu().numpy()
            elif self.cluster_model == "SSC":
                C = self.self_expressive_module.Coef.detach().cpu().numpy()

            reconstruction_loss = reconstruction_loss.item()
            autoencoder_loss = autoencoder_loss.item()
            total_loss = total_loss.item()
            complete_data = decoded.detach().cpu().numpy()
            lr = self.optimizer.param_groups[0]['lr']
            all_params = {name: param.detach().cpu().numpy() for name, param in self.named_parameters()}

            # Logging all parameters
            # for name, param in all_params.items():
            #     self.summary_writer.add_histogram(name, param, global_step=self.iter)

        self.iter += 1
        self.summary_writer.add_scalar('Reconstruction Loss', reconstruction_loss, self.iter)
        self.summary_writer.add_scalar('Autoencoder Loss', autoencoder_loss, self.iter)
        self.summary_writer.add_scalar('Total Loss', total_loss, self.iter)

        return C, total_loss, complete_data, lr
    
class PseudoCompletion(nn.Module):
    def __init__(self, input_shape, flat_layer_size):
        super(PseudoCompletion, self).__init__()

        self.input_shape = input_shape
        self.flat_layer_size = flat_layer_size
        self.feature_size = np.product(self.input_shape[1:])
        self.fc_layers = nn.ModuleList()
        for i in range(self.feature_size):
            # layer = nn.Linear(flat_layer_size[0], flat_layer_size[0],bias = True)
            # # init.xavier_normal_(layer.weight)
            # self.fc_layers.append(layer)

            layer = nn.Linear(flat_layer_size[0], flat_layer_size[0], bias=True)
            # Initialize PReLU with 1 initial parameter per channel
            prelu = nn.PReLU(num_parameters=1, init=0.25)  # You can adjust init value
            self.fc_layers.append(nn.Sequential(layer, prelu))

    def forward(self, x):
        x = x.reshape(self.flat_layer_size[0], -1)
        stacked_tensors_list = []
        for i in range(self.feature_size):
            input_flat = x[:, i].view(1, self.flat_layer_size[0])
            input_missing_flat , mask_flat_tensor = convert_nan(input_flat)
            input_missing_flat = input_missing_flat.float()
            
            flat_layer_output = self.fc_layers[i](input_missing_flat)
            stacked_tensors_list.append(flat_layer_output.t())

        stacked_flat_tensor = torch.cat(stacked_tensors_list, dim=1)
        return stacked_flat_tensor.view(self.input_shape)
class Encoder(nn.Module):
    def __init__(self, input_shape, enc_layer_size,kernel_size):
        super(Encoder, self).__init__()

        self.input_shape = input_shape
        self.enc_layer_size = enc_layer_size
        self.kernel_size = kernel_size

        self.fc_layers = nn.ModuleList()
        if not self.kernel_size:
            for i in range(len(enc_layer_size)):
                if i == 0:
                    layer = nn.Linear(input_shape[1], enc_layer_size[i],bias = True)
                else:
                    layer = nn.Linear(enc_layer_size[i - 1], enc_layer_size[i],bias = True)

                self.fc_layers.append(nn.Sequential(layer, nn.PReLU()))
        else:
            for i in range(len(enc_layer_size)):
                in_channels = self.input_shape[-1] if i == 0 else enc_layer_size[i - 1]  # Determine input channels
                out_channels = enc_layer_size[i]  # Determine output channels

                layer = nn.Conv2d(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=(self.kernel_size[i],self.kernel_size[i]),
                                stride=(2,2),
                                padding=(1,1),bias = True)
                init.xavier_normal_(layer.weight)
                self.fc_layers.append(nn.Sequential(layer, nn.PReLU()))

    def forward(self, x):

        x , mask_flat_tensor = convert_nan(x)
        encoder_layer = x.float()
        
        if not self.kernel_size:
            for layer in self.fc_layers:
                encoder_layer = layer(encoder_layer)
        else:
            encoder_layer = encoder_layer.permute(0,3,1,2)
            for layer in self.fc_layers:
                encoder_layer = layer(encoder_layer)
            encoder_layer = encoder_layer.permute(0,2,3,1)
        return encoder_layer


class Decoder(nn.Module):
    def __init__(self, encoded_shape,deco_layer_size,kernel_size,output_padding):
        super(Decoder, self).__init__()
        self.encoded_shape = encoded_shape
        self.deco_layer_size = deco_layer_size
        self.kernel_size = kernel_size
        self.kernel_length = len(kernel_size) - 1 if kernel_size else None
        self.output_padding = output_padding

        self.fc_layers = nn.ModuleList()
        if not self.kernel_size:
            for i in range(len(deco_layer_size)):
                if i == 0:
                    layer = nn.Linear(encoded_shape, deco_layer_size[i],bias = True)
                else:
                    layer = nn.Linear(deco_layer_size[i-1], deco_layer_size[i],bias = True)

                self.fc_layers.append(nn.Sequential(layer, nn.PReLU()))
        else:
            for i in range(len(deco_layer_size)):
                in_channels = self.encoded_shape if i == 0 else deco_layer_size[i - 1]  # Determine input channels
                out_channels = deco_layer_size[i]  # Determine output channels

                layer = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, 
                                           kernel_size=(self.kernel_size[self.kernel_length - i],self.kernel_size[self.kernel_length - i]), 
                                           stride=(2,2), padding=(1,1), 
                                           output_padding=output_padding[i],bias = True)

                init.xavier_normal_(layer.weight)
                
                self.fc_layers.append(nn.Sequential(layer, nn.PReLU()))

    def forward(self, z):
        decoder_layer = z
        if not self.kernel_size:
            for layer in self.fc_layers:
                decoder_layer = layer(decoder_layer)
            
        else:
            decoder_layer = decoder_layer.permute(0,3,1,2)
            for layer in self.fc_layers:
                decoder_layer = layer(decoder_layer)
            decoder_layer = decoder_layer.permute(0,2,3,1)
        return decoder_layer
    
class SelfExpressiveModule(nn.Module):
    def __init__(self, batch_size):
        super(SelfExpressiveModule, self).__init__()

        self.batch_size = batch_size

        # Initialize Coef
        self.Coef = nn.Parameter(1.0e-4 * torch.ones(batch_size, batch_size))

    def forward(self, Z):

        theta_Z = torch.matmul(self.Coef, Z)

        return theta_Z, self.Coef

# class CFSModule(nn.Module):
#     def __init__(self,rank):
#         super(CFSModule, self).__init__()
#         self.rank = rank

#     def forward(self, Z):
#         Zt = Z.t()
#         U, S, V = torch.svd(Zt)
#         V_rank = V[:, :self.rank]
#         P = torch.mm(V_rank, V_rank.t())
#         self.Coef = P.t()
#         PZ = torch.mm(self.Coef, Z)

#         return PZ, self.Coef
    

class CFSModule(nn.Module):
    """Coefficient-Feature Self-expressive module using stable SVD."""
    def __init__(self, rank):
        super().__init__()
        self.rank = rank

    def forward(self, Z):
        # Z: batch_size x features
        Zt = Z.t()  # features x batch_size
        # stable SVD: try torch.linalg.svd, fallback to NumPy
        try:
            U, S, Vh = torch.linalg.svd(Zt, full_matrices=False)
            V = Vh.transpose(-2, -1)
        except RuntimeError:
            import numpy as _np
            Zt_cpu = Zt.detach().cpu().numpy()
            U_np, S_np, Vt_np = _np.linalg.svd(Zt_cpu, full_matrices=False)
            U = torch.from_numpy(U_np).to(Zt.device)
            S = torch.from_numpy(S_np).to(Zt.device)
            V = torch.from_numpy(Vt_np.T).to(Zt.device)
        # select top-rank singular vectors
        V_rank = V[:, :self.rank]
        P = V_rank @ V_rank.t()
        self.Coef = P.t()
        PZ = self.Coef @ Z
        return PZ, self.Coef