# MultiRegionsDeformable
Multi Region Image Segmentation with a Single Level Set Function. This repository contains a python implementation of the [paper](https://ieeexplore.ieee.org/document/6995952) based mostly on numpy. 

## Installation
You need:
- standard python libraires numpy, scikit-image, scipy, PIL
- opencv (for visualization purposes)
- scikit-fmm (you can install it via the command `pip install scikit-fmm`). 

To reproduce the plots :
- matplotlib
- seaborn

In this repository, we used the nt-toolbox library, developped by Gabriel Peyré, and transposed to Python language by Theo Bertrand (see : [nt-toolbox](https://github.com/TheoBertrand-Dauphine/MVA_NT_geodesic_methods))


## Usage
To execute the pipeline, you can run the following command:
```
python multi_regions_segmentation.py \
       --img_path="path/to/mypicture.jpg \
       --resize=200 \
       --n_regions_init=3 \
       --initialization="kmeans" \
       --n_iter=30 \
       --mu=1 \
       --dt=20 \
       --thresh=0.1 \
       --it_view=10 \
       --dilat_size=3 \
       --save_video="path/to/myvideo.mp4"
```
You can play around with the arguments listed above. For example, replace the initialization parameter by "circles" to initialize your regions with evenly spaced circles across the image. You can also vary the different thresholds.
If you need any information about the parameters, just execute `python multi_regions_segmentation.py -h`.

Finally, the argument `--save_video` allows you to visualize the evolution of the contours after having run the simulation. If you don't want to save a video, just remove the argument (it is not required)
