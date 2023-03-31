import argparse
import cv2
import numpy as np
import scipy
from scipy.ndimage import gaussian_filter

from PIL import Image
from sklearn.cluster import KMeans

# nt-toolbox
from nt_toolbox.grad import grad
from nt_toolbox.perform_fast_marching import perform_fast_marching_isotropic

# Local lib
from utils_contours import get_contours_regions, get_voronoi_cij
from utils_levelsets import update_Phi, get_eps_levels, new_contours_from_eps_levels, merge_regions
from utils_plot import plot_img_with_contours, two_plot

def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', help="Image path")
    parser.add_argument('--resize', type=int, default=200, help="Size to which transform the image")
    parser.add_argument('--initialization', default="circles", help="Choose initalization method (between circles and kmeans)")
    parser.add_argument('--n_regions_init', type=int, default=6, help="Number of regions for kmeans/circles horizontally")
    parser.add_argument('--n_iter', type=int, default=200, help="Number of iterations (max)")
    parser.add_argument('--mu', type=float, default=1, help="Regularization param mu")
    parser.add_argument('--dt', type=float, default=1, help="Time step")
    parser.add_argument('--dilat_size', type=int, default=3, help="Dilation factor")
    parser.add_argument('--thresh', type=float, default=.1, help="Threshold for region merging")
    parser.add_argument('--it_view', type=int, default=10, help="Interval to plot results")
    parser.add_argument('--save_video', required=False, help="If given, will save the video to required path")

    args = parser.parse_args()
    return args

def load_img(img_path, new_size):
    I = np.array(Image.open(img_path)).astype('uint16')
    I = cv2.resize(I, (new_size, new_size))
    I = I.astype(float)

    I /= 255.
    return I

def initialize_regions_circles(I, n_circles=6):
    regions = np.ones_like(I)
    n = I.shape[0]
    it_regions = 2
    for i in range(0, n_circles):
        for j in range(0, n_circles):
            regions = cv2.circle(regions, ((n*i)//n_circles + n//(n_circles*2), (n*j)//n_circles + n//(n_circles*2)), n//(n_circles*2 +2), color=it_regions, thickness=-1)
            it_regions += 1
    return regions

def initialize_kmeans(I, n_clusters, new_size):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(I.ravel()[:, None])
    regions = (1 + (kmeans.predict(I.ravel()[:, None])).reshape(new_size,new_size))
    return regions


def save_evolution(I, video_seq, save_path):
    fourcc = cv2.VideoWriter_fourcc(*"MP4V")
    fps = 5
    height, width = I.shape

    video_writer = cv2.VideoWriter(save_path,
                                   fourcc,
                                   fps,
                                   (width, height))

    for im in video_seq:
        im_new = im*255.
        im_new = cv2.cvtColor(im_new.astype('uint8'), cv2.COLOR_GRAY2BGR)
        video_writer.write(im_new)

    video_writer.release()

if __name__ == '__main__':
    args = get_params()

    # Define additional params from args
    resize = args.resize
    dilat_size = args.dilat_size
    dt = args.dt
    mu = args.mu
    eps_ls = 3*dilat_size
    it_view = args.it_view
    eps = 1e-15
    n_regions_init = args.n_regions_init
    thresh_merge = args.thresh

    # Load image
    I = load_img(args.img_path, resize)

    # Initialize regions
    if args.initialization == "circles":
        print("Initializing regions with evenly spaced circles")
        regions = initialize_regions_circles(I, n_regions_init)
    else:
        print("Initalizing regions with kmeans")
        regions = initialize_kmeans(I, n_regions_init, resize)
    
    # Get initial contours
    contours, contours_regions = get_contours_regions(regions)

    # Plot initial regions/contours
    plot_img_with_contours(I, regions, contours)

    # Dilat image
    I_dilat = scipy.ndimage.zoom(I, dilat_size, order=0)
    
    # Get inverse edge indicator function
    g = 1/(1 + np.sum(grad(gaussian_filter(I_dilat, sigma=5), order=2)**2, axis=2))

    # Initialize Phi
    Phi = perform_fast_marching_isotropic(np.ones_like(I_dilat), dilat_size*contours.T)

    # Initalize video
    if args.save_video:
        video_seq = [cv2.addWeighted(regions, 0.7, I, 0.2, 0)]
    
    for it in range(args.n_iter):
        #### STEP 1
        # Get unique regions
        unique_regions = np.unique(regions)
        
        # Get mean intensity per region
        ci = {}
        for i in unique_regions:
            ci[i] = np.mean(I[regions == i])

        if it % it_view == 0:
            print('STEP 1')
            print(f'mean intensities: {ci}')
      
        # Dilate original regions
        regions_dilat = scipy.ndimage.zoom(regions, dilat_size, order=0)
        contours_dilat, contours_regions_dilat = get_contours_regions(regions_dilat)

        # Get Voronoi regions from C_ij
        unique_neighbors = np.unique(contours_regions, axis=0)
        Voronoi, _ = get_voronoi_cij(contours_dilat, contours_regions_dilat, unique_neighbors, shape_vor=I_dilat.shape)


        # Compute new Phi (thanks to extended velocity)
        Phi_new, norm_grad_phi = update_Phi(I_dilat, Phi, g, regions_dilat, unique_regions, Voronoi, unique_neighbors, dt, ci, mu)


        if it % it_view == 0:
            print('Phi OLD & NEW')
            two_plot(Phi, Phi_new)
            print('--------------')
            print('STEP 2')
        
        #### STEP 2
        ## eps-level sets
        eps_level_sets, global_ls_mask, mask_per_reg = get_eps_levels(regions_dilat, unique_regions, Phi_new, eps_ls)

        ## new contours from eps level sets
        regions_new, Indices, Indices_dilated = new_contours_from_eps_levels(eps_level_sets, I_dilat, unique_regions, dilat_size)

        # Update contours
        contours, contours_regions = get_contours_regions(regions_new)

        # merge neighboring regions that are too similar
        regions_new = merge_regions(regions_new, contours_regions, ci, thresh_merge)

        # Re-Update contours
        contours, contours_regions = get_contours_regions(regions_new)
        
        # Update Regions & Phi
        diff_regions = np.sum(regions != regions_new)
        print(f'Are regions diff : {diff_regions}')
        regions = regions_new
        Phi = Phi_new

        if diff_regions == 0:
            print("Regions are not evolving : stopping")
            break
        
        if args.save_video:
            dst = cv2.addWeighted(regions, 0.7, I, 0.2, 0)
            video_seq.append(dst)
        
        if it % it_view == 0:
            plot_img_with_contours(I, regions, contours)

        #redistance Phi
        Phi = perform_fast_marching_isotropic(np.ones_like(I_dilat), dilat_size*contours.T)
    
    print("Final Segmentation")
    plot_img_with_contours(I, regions, contours)

    if args.save_video:
        print("Saving video")
        save_evolution(I, video_seq, args.save_video)

