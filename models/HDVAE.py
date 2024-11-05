# 
# HDVAE = VanillaVAE3
# Inspired from https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py
# Last Updated: 26 Sep 2024
# Developed by Mohammad Reza Hasanabadi (MRH)
------------------------------
import torch
from models import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *
import math

# MRH:
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture


class VanillaVAE3(BaseVAE):
    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(VanillaVAE3, self).__init__()

        self.latent_dim = latent_dim

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim)

        # Encoder for Level 2
        hidden_dim=hidden_dims[-1]*4
        self.fc2 = nn.Linear(latent_dim, hidden_dim)
        self.fc_mu2 = nn.Linear(hidden_dim, latent_dim)
        self.fc_log_var2 = nn.Linear(hidden_dim, latent_dim)

        # Encoder for Level 3
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc_mu3 = nn.Linear(hidden_dim, latent_dim)
        self.fc_log_var3 = nn.Linear(hidden_dim, latent_dim)

        # MRH:
        # Convolutional Layers to increase dimensions
        self.mrh_conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, stride=2, padding=2)
        self.mrh_conv1_v2 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=5, stride=2, padding=2)
        self.mrh_conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=2)
        self.mrh_conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2)
        self.mrh_conv4 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2)
        self.mrh_conv4_v2 = nn.Conv1d(in_channels=64, out_channels=512, kernel_size=5, stride=2, padding=2)
        self.mrh_conv5 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=2)
        self.mrh_conv6 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=5, stride=2, padding=2)
        self.mrh_conv7 = nn.Conv1d(in_channels=512, out_channels=1024, kernel_size=5, stride=2, padding=2)
        self.mrh_conv7_v2 = nn.Conv1d(in_channels=512, out_channels=2048, kernel_size=5, stride=2, padding=2)
        self.mrh_conv8 = nn.Conv1d(in_channels=1024, out_channels=2048, kernel_size=5, stride=2, padding=2)
        #
        self.mrh_convs = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=2, padding=2),  # Output: (batch_size, 32, 64)
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2),  # Output: (batch_size, 64, 32)
            nn.LeakyReLU()
        )
        self.fc_mu_4 = nn.Linear(64 * 32, 128)  # Output from conv block to latent_dim
        self.fc_log_var_4= nn.Linear(64 * 32, 128)  # Output from conv block to latent_dim
        self.mrh_leaky_relu = nn.LeakyReLU(negative_slope=0.01)

        # # Transposed Convolutional Layers to reduce dimensions back
        # self.mrh_deconv1 = nn.ConvTranspose1d(in_channels=2048, out_channels=1024, kernel_size=5, stride=2, padding=2, output_padding=1)
        # self.mrh_deconv2 = nn.ConvTranspose1d(in_channels=1024, out_channels=512, kernel_size=5, stride=2, padding=2, output_padding=1)
        # self.mrh_deconv3 = nn.ConvTranspose1d(in_channels=512, out_channels=256, kernel_size=5, stride=2, padding=2, output_padding=1)
        # self.mrh_deconv4 = nn.ConvTranspose1d(in_channels=256, out_channels=self.latent_dim, kernel_size=5, stride=2, padding=2, output_padding=1)


        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= 3,
                                      kernel_size= 3, padding= 1),
                            nn.Tanh())

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def mrh_encode_z2(self, input: Tensor) -> List[Tensor]:

        # Reshape input to include channel dimension
        input = input.unsqueeze(1)  # shape (batch_size, 1, latent_dim)
        #print('mrh inside mrh_encode_z2 input= ',input.shape)
        # Convolutional forward pass
        z2 = self.mrh_leaky_relu(self.mrh_conv1(input))
        z2 = self.mrh_leaky_relu(self.mrh_conv2(z2))
        z2 = self.mrh_leaky_relu(self.mrh_conv3(z2))
        z2 = self.mrh_leaky_relu(self.mrh_conv4(z2))
        z2 = self.mrh_leaky_relu(self.mrh_conv5(z2))
        z2 = self.mrh_leaky_relu(self.mrh_conv6(z2))
        z2 = self.mrh_leaky_relu(self.mrh_conv7(z2))
        z2 = self.mrh_leaky_relu(self.mrh_conv8(z2))
        #print('mrh inside mrh_encode_z2 after covolutional layers= ',z2.shape)

        # Flatten the output before the fully connected layers
        z2 = z2.view(z2.size(0), -1)  # shape (batch_size, -1)
        #print('mrh inside mrh_encode_z2 flatten= ',z2.shape)
        mu2 = self.fc_mu2(z2)
        #print('mrh inside mrh_encode_z2 after fc_mu2 = ',z2.shape)
        log_var2 = self.fc_log_var2(z2)
        #print('mrh inside mrh_encode_z2 after log_var2 ',z2.shape)
        return [mu2, log_var2]

    def mrh_encode_z3(self, input: Tensor) -> List[Tensor]:

        # Reshape input to include channel dimension
        input = input.unsqueeze(1)  # shape (batch_size, 1, latent_dim)
        #print('mrh inside mrh_encode_z2 input= ',input.shape)
        # Convolutional forward pass
        z2 = self.mrh_leaky_relu(self.mrh_conv1(input))
        z2 = self.mrh_leaky_relu(self.mrh_conv2(z2))
        z2 = self.mrh_leaky_relu(self.mrh_conv3(z2))
        z2 = self.mrh_leaky_relu(self.mrh_conv4(z2))
        z2 = self.mrh_leaky_relu(self.mrh_conv5(z2))
        z2 = self.mrh_leaky_relu(self.mrh_conv6(z2))
        z2 = self.mrh_leaky_relu(self.mrh_conv7(z2))
        z2 = self.mrh_leaky_relu(self.mrh_conv8(z2))
        #print('mrh inside mrh_encode_z2 after covolutional layers= ',z2.shape)

        # Flatten the output before the fully connected layers
        z3 = z2.view(z2.size(0), -1)  # shape (batch_size, -1)
        #print('mrh inside mrh_encode_z2 flatten= ',z2.shape)
        mu3 = self.fc_mu3(z3)
        #print('mrh inside mrh_encode_z2 after fc_mu2 = ',z2.shape)
        log_var3 = self.fc_log_var3(z3)
        #print('mrh inside mrh_encode_z2 after log_var2 ',z2.shape)
        return [mu3, log_var3]

    def mrh_encode_z_v2(self, input: Tensor) -> List[Tensor]:

        # Reshape input to include channel dimension
        input = input.unsqueeze(1)  # shape (batch_size, 1, latent_dim)
        #print('mrh inside mrh_encode_z2 input= ',input.shape)
        # Convolutional forward pass
        z2=self.mrh_convs(input)
        #print('mrh inside mrh_encode_z2 after covolutional layers= ',z2.shape)
        # Flatten the output before the fully connected layer
        z2 = z2.view(z2.size(0), -1)  # shape (64, 64*32)

        #print('mrh inside mrh_encode_z2 flatten= ',z2.shape)
        mu2 = self.fc_mu_4(z2)
        #print('mrh inside mrh_encode_z2 after fc_mu2 = ',z2.shape)
        log_var2 = self.fc_log_var_4(z2)
        #print('mrh inside mrh_encode_z2 after log_var2 ',z2.shape)
        return [mu2, log_var2]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result


    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        # MRH:
        #std = torch.exp(0.5 * logvar)
        #print('\nmrh inide reparameterize logvar before= ',logvar[0][0:3])
        std = torch.clamp(logvar, min=-10, max=2)  # Adjust min/max as needed
        #print('mrh inide reparameterize logvar after= ',std[0][0:3])
        eps = torch.randn_like(std)
        return eps * std + mu

    ### MRH:
    #def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
    def forward(self, input: Tensor,  **kwargs) -> List[Tensor]:
        #print('\nmrh inside model forward: shape= ',input.shape) # shape=  torch.Size([64, 3, 64, 64])
        #raise Exception("Mohammad Reza Hasanabadi")
        #print('\nmrh inside model forward start')

        # Assume prior means and log variances for each layer are defined as follows
        batch=input.size(0)
        mu_p_list = [torch.zeros(batch, self.latent_dim) for _ in range(3)]  # Three layer, standard normal
        log_var_p_list = [torch.zeros(batch, self.latent_dim) for _ in range(3)]
        #
        mu1, log_var1 = self.encode(input)
        z1 = self.reparameterize(mu1, log_var1)

        ############### MRH:
        # h2 = F.relu(self.fc2(z1))
        # mu2 = self.fc_mu2(h2)
        # log_var2 = self.fc_log_var2(h2)
        # z2 = self.reparameterize(mu2, log_var2)
        # #
        # h3 = F.relu(self.fc2(z2))
        # mu3 = self.fc_mu2(h3)
        # log_var3 = self.fc_log_var2(h3)
        # z3 = self.reparameterize(mu3, log_var3)
        ###############
        # h2 = F.relu(self.fc2(z1))
        # mu2 = self.fc_mu2(h2)
        # log_var2 = self.fc_log_var2(h2)
        # z2 = self.reparameterize(mu2, log_var2)
        # #
        # h3 = F.relu(self.fc3(z2))
        # mu3 = self.fc_mu3(h3)
        # log_var3 = self.fc_log_var3(h3)
        # z3 = self.reparameterize(mu3, log_var3)
        ############################
        # mu2, log_var2 = self.mrh_encode_z2(z1)
        mu2, log_var2 = self.mrh_encode_z_v2(z1)
        z2 = self.reparameterize(mu2, log_var2)
        #
        # mu3, log_var3 = self.mrh_encode_z2(z2)
        mu3, log_var3 = self.mrh_encode_z_v2(z2)
        z3 = self.reparameterize(mu3, log_var3)

        #
        z=[z1,z2,z3]
        #
        mu_q_list = [mu1, mu2, mu3]
        log_var_q_list = [log_var1, log_var2, log_var3]
        #print('mrh inside forward= ','mu_q_list= ',mu_q_list, 'log_var_q_list = ',log_var_q_list )
        #

        #
        recon_z1 = self.decode(z1)
        recon_z2 = self.decode(z2)
        recon_z3 = self.decode(z3)

        ############### MRH:
        # mu2 , _ =self.mrh_encode_2(mu1)
        # _ , log_var2=self.mrh_encode_2(log_var1)
        # z2 = self.reparameterize(mu2, log_var2)
        # z3= torch.cat((z1, z2), 1)
        # z = self.mrh_decoder_2(z3)
        quality_metrics_z1 = self.assess_quality(recon_z1, input)
        quality_metrics_z2 = self.assess_quality(recon_z2, input)
        quality_metrics_z3 = self.assess_quality(recon_z3, input)
        #print(f"PSNR: {quality_metrics['PSNR']}, SSIM: {quality_metrics['SSIM']}")
        #print('mrh inside  loss= ','mu_q= ',mu_q_list[0].shape,'log_var_q= ',log_var_q_list[0].shape,'mu_p= ',mu_p_list[0].shape,'log_var_p= ',log_var_p_list[0].shape)
        ################

        # MRH:
        #return  [self.decode(z), input, mu, log_var]
        #print('MRH inside Forward = ',"z1= ", z1[0] ,"z2= ", z2[0],"z1_recon= ",z1_recon[0] )
        return  [recon_z3, input, mu_q_list, log_var_q_list, mu_p_list, log_var_p_list, z, quality_metrics_z1, quality_metrics_z2, quality_metrics_z3, recon_z1, recon_z2 ]

    ### MRH:

    def loss_function(self, *args,**kwargs) -> dict:
        """
        recons, input, mu_q, log_var_q, mu_p, log_var_p, kwargs
        """
        recon_z3 = args[0]
        input = args[1]
        mu_q_list = args[2]
        log_var_q_list = args[3]
        mu_p_list = args[4]
        log_var_p_list = args[5]
        z=args[6]
        z1=z[0]
        z2=z[1]
        z3=z[2]
        #
        quality_metrics_z1=args[7]
        quality_metrics_z2=args[8]
        quality_metrics_z3=args[9]
        total_PSNR = (quality_metrics_z1['PSNR'] + quality_metrics_z2['PSNR'] + quality_metrics_z3['PSNR'])/3
        total_SSIM = (quality_metrics_z1['SSIM'] + quality_metrics_z2['SSIM'] + quality_metrics_z3['SSIM'])/3
        #print('mrh inside mutual_information z1 & z2 shape= ',type(z1), type(z2))   # Torch tensor
        #print('mrh inside mutual_information z1 & z2 shape= ',z1.shape, z2.shape) # torch.Size([64, 128]) torch.Size([64, 128])

        ### MRH:
        recon_z1=args[10]
        recon_z2=args[11]

        # MRH: KLD
        kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        recon_z3_loss = F.mse_loss(recon_z3, input)
        # MRH:
        recon_z1_loss = F.mse_loss(recon_z1, input)
        recon_z2_loss = F.mse_loss(recon_z2, input)
        total_Reconstruction_Loss= recon_z3_loss #+ recon_z2_loss + recon_z1_loss
        #
        total_kld_loss, kld_loss_list = self.hierarchical_kl(mu_q_list, log_var_q_list, mu_p_list, log_var_p_list)
        kld_loss_z1 = kld_loss_list[0]
        kld_loss_z2 = kld_loss_list[1]
        kld_loss_z3 = kld_loss_list[2]


        # ### MRH: Mutual Information
        mi_loss12 = self.mutual_information(z1, z2)
        mi_loss13 = self.mutual_information(z1, z3)
        mi_loss23 = self.mutual_information(z2, z3)
        # mi_loss12 = self.mutual_information_kde(z1, z2)
        # mi_loss13 = self.mutual_information_kde(z1, z3)
        # mi_loss23 = self.mutual_information_kde(z2, z3)
        mi_loss=mi_loss12 + mi_loss13 + mi_loss23
        #print('mi_loss= ', mi_loss,'mi_loss12= ', mi_loss12,' mi_loss13= ',  mi_loss13, 'mi_loss23= ', mi_loss23)
        # lambda_mi = 0.1
        lambda_mi = 10
        MMI_loss= lambda_mi * mi_loss
        #print('MMI_loss= ',MMI_loss)  # 0.4825816304871043
        # ###

        total_loss = total_Reconstruction_Loss + kld_weight * total_kld_loss  + MMI_loss
        return {
            'total_loss': total_loss,
            'total_Reconstruction_Loss': total_Reconstruction_Loss.detach(),
            'recon_z1_loss':  recon_z1_loss.detach(),
            'recon_z2_loss':  recon_z2_loss.detach(),
            'recon_z3_loss':  recon_z3_loss.detach(),
            'hierarchical_KLD': -total_kld_loss.detach(),
            'kld_loss_z1': -kld_loss_z1.detach(),
            'kld_loss_z2': -kld_loss_z2.detach(),
            'kld_loss_z3': -kld_loss_z3.detach(),
            'mi_loss': mi_loss,
            'mi_loss12': mi_loss12,
            'mi_loss13': mi_loss13,
            'mi_loss23': mi_loss23,
            'total_PSNR': total_PSNR,
            'PSNR_z1': quality_metrics_z1['PSNR'],
            'PSNR_z2': quality_metrics_z2['PSNR'],
            'PSNR_z3': quality_metrics_z3['PSNR'],
            'total_SSIM': total_SSIM,
            'SSIM_z1': quality_metrics_z1['SSIM'],
            'SSIM_z2': quality_metrics_z2['SSIM'],
            'SSIM_z3': quality_metrics_z3['SSIM'],
        }

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]


    ### MRH:
    def psnr(self, img1: Tensor, img2: Tensor) -> float:
        """
        Compute the PSNR between two images.
        :param img1: (Tensor) First image [B x C x H x W]
        :param img2: (Tensor) Second image [B x C x H x W]
        :return: (float) PSNR value
        """
        img1 = img1.detach().cpu().numpy()
        img2 = img2.detach().cpu().numpy()

        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return float('inf')  # Identical images have infinite PSNR

        max_pixel = 1.0  # Assuming output is normalized between -1 and 1
        psnr_value = 20 * np.log10(max_pixel / np.sqrt(mse))
        return psnr_value

    def compute_ssim(self, img1: Tensor, img2: Tensor) -> float:
        """
        Compute the SSIM between two images.
        :param img1: (Tensor) First image [B x C x H x W]
        :param img2: (Tensor) Second image [B x C x H x W]
        :return: (float) SSIM value
        """
        img1 = img1.detach().cpu().numpy()
        img2 = img2.detach().cpu().numpy()

        ssim_value = 0
        # Compute SSIM for each channel and average
        for c in range(img1.shape[1]):  # Iterate through color channels
            ssim_value += ssim(img1[0, c], img2[0, c], data_range=img2.max() - img2.min())

        ssim_value /= img1.shape[1]  # Average SSIM for all channels
        return ssim_value

    def assess_quality(self, recons: Tensor, input: Tensor) -> dict:
        """
        Assess the reconstruction quality using PSNR and SSIM.
        :param recons: (Tensor) Reconstructed images [B x C x H x W]
        :param input: (Tensor) Original images [B x C x H x W]
        :return: (dict) Dictionary containing PSNR and SSIM values
        """
        psnr_value = self.psnr(recons, input).item()
        ssim_value = self.compute_ssim(recons, input).item()

        return {
            'PSNR': psnr_value,
            'SSIM': ssim_value
        }
    ### MRH:
    def kl_divergence(self, mu_q, log_var_q, mu_p, log_var_p):
        """
        Computes the KL divergence between two Gaussian distributions.

        :param mu_q: (Tensor) Mean of the variational distribution q
        :param log_var_q: (Tensor) Log variance of the variational distribution q
        :param mu_p: (Tensor) Mean of the prior distribution p
        :param log_var_p: (Tensor) Log variance of the prior distribution p
        :return: (Tensor) The KL divergence value
        """

        log_var_p=log_var_p.cuda()
        #print('mrh device= ', mu_q.get_device(), log_var_q.get_device(),log_var_p.get_device() )
        return -0.5 * torch.sum(1 + log_var_q - mu_q.pow(2) - log_var_q.exp() / log_var_p.exp(), dim=1)

    def hierarchical_kl(self, mu_q_list, log_var_q_list, mu_p_list, log_var_p_list):
        """
        Computes the combined KL divergence for the hierarchical ELBO structure.

        :param mu_q: (List of Tensors) Means of the variational distributions for all levels
        :param log_var_q: (List of Tensors) Log variances of the variational distributions for all levels
        :param mu_p: (List of Tensors) Means of the prior distributions for all levels
        :param log_var_p: (List of Tensors) Log variances of the prior distributions for all levels
        :return: (Tensor) Total KL divergence value
        """
        kl_total = 0
        kld_list=[]
        for i in range(len(mu_q_list)):
            value= self.kl_divergence(mu_q_list[i], log_var_q_list[i], mu_p_list[i], log_var_p_list[i])
            kl_total = kl_total + value
            kld_list.append(value.mean())

        #print('kl_total= ',kl_total)
        kl_total=kl_total.mean()
        return kl_total, kld_list

###
    def mutual_information(self, z1, z2, k=10): # MRH: k=5
        # Stack the latent vectors together
        #print('mrh inside mutual_information z1 & z2 shape= ',type(z1), type(z2))  # torch tensor
        #print('mrh inside mutual_information z1 & z2 shape= ',z1.shape, z2.shape)  # torch.Size([64, 128]) torch.Size([64, 128])
        # Normaization
        z1 = (z1 - z1.mean(dim=0)) / (z1.std(dim=0) + 1e-10)
        z2 = (z2 - z2.mean(dim=0)) / (z2.std(dim=0) + 1e-10)
        #
        z = torch.cat([z1, z2], dim=0).detach().cpu().numpy()  # Convert to numpy for sklearn
        n_samples = z.shape[0]
        #print('n_samples= ',n_samples)  # int 128

        # Nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=k + 1).fit(z)
        distances, indices = nbrs.kneighbors(z)

        # Exclude the first neighbor since it's the point itself
        distances = distances[:, 1:]

        # Calculate entropy for each latent variable
        def entropy(distances):
            # Convert to density
            density = np.exp(-distances)
            prob = density / np.sum(density, axis=1, keepdims=True)
            ent = -np.sum(prob * np.log(prob + 1e-10), axis=1)  # Avoid log(0)
            return np.mean(ent)

        h_z1 = entropy(distances[:n_samples//2])  # Entropy of z1
        h_z2 = entropy(distances[n_samples//2:])  # Entropy of z2

        # Joint distribution entropy
        joint = np.vstack((z1.detach().cpu().numpy(), z2.detach().cpu().numpy()))
        joint_nbrs = NearestNeighbors(n_neighbors=k + 1).fit(joint)
        joint_distances, _ = joint_nbrs.kneighbors(joint)
        joint_entropy = entropy(joint_distances[:, 1:])  # Without self

        # Mutual Information
        mi = h_z1 + h_z2 - joint_entropy
        return mi
##

    def mutual_information_kde(self, z1, z2, bandwidth=10):  # MRH=1
        # Normalize the inputs
        #print('\nz1= ', z1)
        #print('\ntype z1= ', type(z1))
        z1 = (z1 - z1.mean(dim=0)) / (z1.std(dim=0) + 1e-10)
        z2 = (z2 - z2.mean(dim=0)) / (z2.std(dim=0) + 1e-10)
        #
        # Standardization (mean=0, std=1)
        # z1 = (z1 - z1.mean(dim=0)) / (z1.std(dim=0, unbiased=False) + 1e-10)
        # z2 = (z2 - z2.mean(dim=0)) / (z2.std(dim=0, unbiased=False) + 1e-10)
        #
        # z1 = (z1 - z1.min(dim=0)[0]) / (z1.max(dim=0)[0] - z1.min(dim=0)[0] + 1e-10)
        # z2 = (z2 - z2.min(dim=0)[0]) / (z2.max(dim=0)[0] - z2.min(dim=0)[0] + 1e-10)

        # Stack the latent vectors together
        z = torch.cat([z1, z2], dim=0).detach().cpu().numpy()
        n_samples = z.shape[0]

        # Function to estimate entropy using KDE
        def entropy_kde(data):
            kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(data)
            log_density = kde.score_samples(data)
            return -np.mean(log_density)

        def entropy_gmm(data, n_components=10):
            gmm = GaussianMixture(n_components=n_components).fit(data)
            log_probs = gmm.score_samples(data)
            return -np.mean(log_probs)

        # Calculate individual entropies
        h_z1 = entropy_kde(z1.detach().cpu().numpy())
        h_z2 = entropy_kde(z2.detach().cpu().numpy())
        # h_z1 = entropy_gmm(z1.detach().cpu().numpy())
        # h_z2 = entropy_gmm(z2.detach().cpu().numpy())
        # Calculate joint entropy
        joint = np.vstack((z1.detach().cpu().numpy(), z2.detach().cpu().numpy()))
        h_joint = entropy_kde(joint)

        # Mutual Information
        mi = h_z1 + h_z2 - h_joint
        return mi