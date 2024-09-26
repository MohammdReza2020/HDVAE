**Hierarchical Disentanglement VAE**

**Hierarchical Disentangled Variational Autoencoder (HDVAE)**, a novel framework that utilizes the **Hierarchical Evidence Lower Bound (HELBO)** to enhance representation learning in complex data spaces.The HDVAE features hierarchical disentangled stages that enable the modeling of diverse attributes, facilitating smooth transfer learning across domains. Our approach supports unsupervised latent factor learning, allowing automatic discovery of meaningful latent representations without labeled data, thus broadening applicability across various tasks. We employ strategies for both intra-latent and inter-latent disentanglement, achieving effective separation of latent variables through the integration of mutual information in the loss function. 

By Running mrh_run.py You can follow implementation
Remember to extract .rar files first
Training results:

![image](https://github.com/user-attachments/assets/34377670-a21e-416b-91ce-66b77e4b7300)

Unsupervised disentangled features:

![image](https://github.com/user-attachments/assets/3f1d13db-f0b4-42b1-9e10-8a7a893a278f)

Reconstruction Quality Metrics (PSNR & SSIM):

![image](https://github.com/user-attachments/assets/e2c6fbd7-52c0-48b2-acb2-21b7dc8b913d)

Visulization of Disentangled Hierarchies:

![image](https://github.com/user-attachments/assets/df2c9b7e-0f71-4590-a4e7-af427d0fcfaf)




