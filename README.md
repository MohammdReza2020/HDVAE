**HDVAE: A Hierarchical Disentanglement Representation Based on Hierarchical Evidence Lower Bound (HELBO)**

**Hierarchical Disentangled Variational Autoencoder (HDVAE)**, a novel framework that utilizes the **Hierarchical Evidence Lower Bound (HELBO)** 
This paper introduces the Hierarchical Disentangled Variational Autoencoder (HDVAE), a novel framework that leverages the Hierarchical Evidence Lower Bound (HELBO) to advance representation learning in complex data landscapes. The HDVAE facilitates unsupervised latent factor learning, enabling the automatic discovery of meaningful latent representations without dependence on labeled data, thereby broadening its applicability across diverse tasks. A rigorous mathematical framework is proposed, augmented by information disentanglement achieved through both Intra-Latent Disentanglement (within each hierarchical level) and Inter-Latent Disentanglement (between hierarchical levels). The hierarchical representation's efficacy is initially evaluated on the MNIST dataset using UMAP for visualization. Subsequently, the disentanglement capabilities are further assessed on more complex and challenging datasets, including CIFAR10, SVHN, and STL-10. To quantify the effectiveness of the HDVAE in disentangling latent factors, three established metrics—MIG, DCI metric, and the average R2 disentanglement score—are employed on the Dsprite dataset. Simultaneously, reconstruction quality is evaluated using PSNR and SSIM, with comparisons drawn against prominent models such as Beta-VAE, FactorVAE, and DIP-VAE-I, to underscore the advantages of the proposed approach. The results indicate that, under certain conditions, the HDVAE outperforms other models, while maintaining competitive performance in alternative scenarios. Furthermore, latent space traversals on the CelebA dataset confirm that the HDVAE effectively encodes disentangled features, facilitating smooth and interpretable manipulations within the latent space, a crucial aspect for understanding and controlling individual attributes in generative models.

By Running following *.ipynb  to  You can follow implementation
HELBO_v01_2LayerHierarchy.ipynb
HELBO_v02_3LayerHierarchy.ipynb
HELBO_v03_3LayerHierarchy_UMAP.ipynb
HELBO_v04_3LayerHierarchy_UMAP_BetaKLD.ipynb
HELBO_v04_3LayerHierarchy_UMAP_BetaKLD_CIFAR_10.ipynb
HELBO_v04_3LayerHierarchy_UMAP_BetaKLD_STL10.ipynb
HELBO_v04_3LayerHierarchy_UMAP_BetaKLD_SVHN.ipynb
HELBO_v05_3LayerHierarchy_UMAP_BetaKLD_MNIST_Comparison_PSNR_SSIM_LOSS.ipynb
HELBO_v06_3LayerHierarchy_UMAP_BetaKLD_MNIST_Comparison_disentanglementMetrics.ipynb
----------------------------
also for latent Traveral on CelebA, you can follow HDVAE_run.py

Training results:
*Hierarchical Representation*

![1](https://github.com/user-attachments/assets/c3c8a6e4-e34d-494d-a4fa-a1451a3f5fe3)
*Generalization*

![2](https://github.com/user-attachments/assets/cbcb8a26-14e0-49e0-9476-e15049f6cae1)
*Disentanglement*

![3](https://github.com/user-attachments/assets/dc4ed554-57c2-40b7-b8ea-f13c3d9418f9)
![4](https://github.com/user-attachments/assets/03e75637-fba8-4b37-ab97-1556438b1285)
![5](https://github.com/user-attachments/assets/ce052e01-cf34-49fd-854b-400ded086392)

*Information Disentanglement*

![image](https://github.com/user-attachments/assets/34377670-a21e-416b-91ce-66b77e4b7300)

*Latent Space Traversal*

![image](https://github.com/user-attachments/assets/3f1d13db-f0b4-42b1-9e10-8a7a893a278f)

**Contact**
If you have any question about the paper or the code, feel free to email me at m_hasanabadi@sbu.ac.ir





