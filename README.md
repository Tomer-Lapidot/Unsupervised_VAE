# Unsupervised_VAE
Exploration of unsupervised variational autoencoder architectures for cell phenotype clustering.
Objective of the excersize is to see if an unsupervised VAE can effectively classify single cell images.
Project is still under development

Currently exloring the following architechtures,
1. ResNet18 VAE
2. ResNet34 VAE

Training data upon request

Learding results:
ResNet34 with 256 parameter latent space, training on 4X augmented 200,000 single cell images.

Latent space UMAP,
![]https://github.com/Tomer-Lapidot/Unsupervised_VAE/blob/main/Localization_UMAP.png

Density of localizaton plots,
![]https://github.com/Tomer-Lapidot/Unsupervised_VAE/blob/main/Localization_Density_UMAP.png

Reconstruction plots
![]https://github.com/Tomer-Lapidot/Unsupervised_VAE/blob/main/Reconstruction_Plot.png
