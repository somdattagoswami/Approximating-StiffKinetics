
</div>

## Table of contents
* [General info](#general-info)
* [Method](#method)
* [Contact](#contact)

<font color="red">{abc}
## General info

This Git repository contains codes for the **'Learning stiff chemical kinetics using extended deep neural operators'** paper which can be found here: [Link](https://www.sciencedirect.com/science/article/pii/S0045782523007971?dgcid=coauthor).

Authors: [Somdatta Goswami](https://scholar.google.com/citations?user=GaKrpSkAAAAJ&hl=en&oi=sra), [Ameya D. Jagtap](https://scholar.google.com/citations?user=Rh2Ka0gAAAAJ&hl=en&oi=ao), [Hessam Babaee](https://scholar.google.com/citations?hl=en&user=GvQ9aq8AAAAJ), Bryan T. Susi, [George Em Karniadakis](https://scholar.google.com/citations?user=yZ0-ywkAAAAJ&hl=en)

## Method
Stiff chemical kinetics are computationally expensive to solve, thus, this work aims to develop a neural operator-based surrogate model to efficiently solve such problems. 
Additionally, we demonstrate that DeepONet can behave like a solution propagator with large time steps. To that end, we employ Autoencoder (AE)-integrated DeepONet. With AE, a more compact latent representation can be achieved to mitigate the negative effects induced by the acceptable and prevalent circumstance that many species’ mass fractions are zero during reaction system integration. The fact that most of the potential thermochemical states in a reacting system reside on or near a lower-dimensional manifold in spite of the vast number of species in detailed chemical kinetic models serves as the impetus for using an AE to resolve the highly nonlinear chemical kinetics.

In this work, we have employed multi-layer autoencoders to obtain a compact latent representation of the chemical kinetics model for a given time step. The DeepONet is then trained to learn the evolving kinetics of the latent space.

The AE-DeepONet code is written in TF2, while the standalone DeepONet code is in TF1.15.

## Citation

If you find this GitHub repository useful for your work, please consider citing this work:

```
@article{goswami2024learning,
  title={Learning stiff chemical kinetics using extended deep neural operators},
  author={Goswami, Somdatta and Jagtap, Ameya D and Babaee, Hessam and Susi, Bryan T and Karniadakis, George Em},
  journal={Computer Methods in Applied Mechanics and Engineering},
  volume={419},
  pages={116674},
  year={2024},
  publisher={Elsevier}
}
```
______________________

## Contact
For more information or questions please contact us at:   
* somdatta_goswami@brown.edu
* ameya_jagtap@brown.edu 
