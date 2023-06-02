# Stochastic diffusions in H2 Sobolev spaces

We describe 2-dimensional shapes in the euclidean space $\mathbb{R}^3$ as elements of $H^2(\mathbb{S}^2, \mathbb{R}^3)$ and so, deformations of the sphere with extra regularity constraints. This approach is particularly inspired by Bauer et al. (2022) and Vialard (2013).


<p align="center">
  <img src="https://github.com/tbesnier/bm-shapes/blob/main/examples/tests/wiener_process/Q_identity_49coeffs.gif" alt="animated_sphere" style="width:100%"/>
  <img src="https://github.com/tbesnier/bm-shapes/blob/main/examples/tests/wiener_process_bunny/Q_identity_25coeffs.gif" alt="animated_bunny" style="width:100%"/>
</p>
  
## Description
This Python package aims at computing and visualizing implicit stochastic diffusion processes between shapes. It provides comprehensive functions to easily compute random processes acting on the spherical harmonic decomposition of 3-dimensional shapes.

## Setup and dependencies

The code involves the following libraries:
- numpy
- scipy
- trimesh
- open3d
- imageio
- matplotlib
- pyssht

## References

    @InProceedings{10.1007/978-3-031-31438-4_19,
    author="Baker, Elizabeth
    and Besnier, Thomas
    and Sommer, Stefan",
    title="A Function Space Perspective on Stochastic Shape Evolution",
    booktitle="Image Analysis",
    year="2023",
    publisher="Springer Nature Switzerland",
    pages="278--292",
    abstract="Modelling randomness in shape data, for example, the evolution of shapes of organisms in biology, requires stochastic models of shapes. This paper presents a new       stochastic shape model based on a description of shapes as functions in a Sobolev space. Using an explicit orthonormal basis as a reference frame for the noise, the model is     independent of the parameterisation of the mesh. We define the stochastic model, explore its properties, and illustrate examples of stochastic shape evolutions using the         resulting numerical framework."
    }

Please cite this paper if you use it in your work.

## Contacts

    Elizabeth Louise Baker: elba@di.ku.dk

    Thomas Besnier: thomas.besnier@univ-lille.fr

    Stefan Sommer: sommer@di.ku.dk
