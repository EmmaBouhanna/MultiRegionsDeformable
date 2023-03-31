import numpy as np
import skimage
from nt_toolbox.div import div
from nt_toolbox.grad import grad
from utils_contours import get_contour_eps_levels
from nt_toolbox.perform_fast_marching import perform_fast_marching_isotropic


def compute_freg(Phi, g, eps=1e-15):
    """
    Compute regularizing extended velocity

    Input :
            - Phi : level set function
            - g : inverse edge indicator function (regularization term)

    Output :
            - freg : geodesic regularization
            - norm_grad_phi : gradient norm of the level set function
    """
    grad_phi = grad(Phi, order=2)
    norm_grad_phi = np.sqrt(np.sum(grad_phi**2, 2))
    normalized_gradient = grad_phi / \
        (np.repeat(norm_grad_phi[:, :, np.newaxis], 2, axis=-1) + eps)
    normalized_gradient = normalized_gradient*g[:, :, np.newaxis]
    freg = div(normalized_gradient[:, :, 0],
               normalized_gradient[:, :, 1], order=2)
    return freg, norm_grad_phi


def update_Phi(I,
               Phi,
               g,
               regions,
               unique_regions,
               Voronoi,
               unique_neighbors,
               dt,
               ci,
               mu):
    """
    Compute extended velocities and new Phi function

    Input :
            - I : the image (nxp)
            - Phi : the level set map (nxp)
            - g : the edge indicator function
            - regions : the region map (nxp)
            - unique_regions : list of unique region ids
            - Voronoi : voronoi regions of each contour portion (nxpx2)
            - unique_neighbors : list of unique adjacent regions (dx2)
            - dt : time step
            - ci : mean intensity per region (dict)
            - mu : regularization parameter

    Output :
            - new_phi function
            - norm_grad_phi : gradient norm

    """
    new_phi = np.zeros_like(I)
    freg, norm_grad_phi = compute_freg(Phi, g)
    for idx, reg in enumerate(unique_regions):
        mask_reg = regions == reg
        for neigh in unique_neighbors:
            if neigh[0] == reg or neigh[1] == reg:
                mask_vor = (Voronoi[:, :, 0] == neigh[0]) * \
                    (Voronoi[:, :, 1] == neigh[1])
                mask_inter = mask_reg*mask_vor

                Fi = (I - ci[reg])**2
                Fi = Fi * mask_inter

                j = np.setdiff1d(neigh, reg)[0]
                Fj = (I - ci[j])**2
                Fj = Fj * mask_inter

                #new_phi = new_phi + (Phi + dt*(np.abs(Fi - Fj))*norm_grad_phi)*mask_inter
                new_phi = new_phi + (Phi + dt*(Fj - Fi) *
                                     norm_grad_phi)*mask_inter
    new_phi = new_phi + mu*freg*norm_grad_phi
    return new_phi, norm_grad_phi


def get_eps_levels(regions_dilat, unique_regions, new_phi, eps_ls):
    """
    Get epsilon level sets for dilated regions from new computation
    of the Phi distance function.
    Contours are obtained by taking the border of new_phi <= eps_ls

    Input : 
            - regions_dilat : regions map (dilated with certain factor)
            - unique_regions : list of unique regions
            - new_phi : level set function
            - eps_ls : level

    Output :
            - eps_level_sets : the coordinates if points such that Phi(x) = eps_ls,
                               for each region
            - global_ls_mask : global mask where Phi(x) = eps_ls (nxp)
            - mask_per_reg : Phi(x) = eps_ls such that x in region i (dict of masks)

    """
    eps_level_sets = {}
    mask_per_reg = {}

    mask_band = (new_phi <= eps_ls)
    mask_band = mask_band - 0.5

    global_ls = get_contour_eps_levels(mask_band)

    global_ls_mask = np.zeros_like(new_phi)
    global_ls_mask[global_ls[0, :], global_ls[1, :]] = 1

    for reg in unique_regions:
        mask_ = global_ls_mask*(regions_dilat == reg)
        mask_per_reg[reg] = mask_
        eps_level_sets[reg] = np.argwhere(mask_)

    return eps_level_sets, global_ls_mask, mask_per_reg


def reduce_masks(mask_per_reg, dilat_size):
    """
    Reduce the mask to go back to original size (revert dilation)

    Input : 
            - mask_per_reg : dictionnary of mask of epsilon level sets per region
            - dilat_size : dilation factor
    Output : reduced masks for each region
    """
    new_masks = {}
    for reg, mask in mask_per_reg.items():
        mask_reduced = skimage.measure.block_reduce(
            mask, (dilat_size, dilat_size), np.max).astype(bool)
    new_masks[reg] = mask_reduced
    return new_masks


def new_contours_from_eps_levels(eps_level_sets,
                                 I_dilat,
                                 unique_regions,
                                 dilat_size):
    """
    Attributes new regions based on eps level sets

    Input : 
            - eps_level_sets : dict of level sets for each region
            - I_dilat : dilated images dilat_size*(nxp)
            - unique_regions : list of unique regions
            - dilat_size : dilation factor
    Output :
            - attrib_regions : new regions
    """
    distance_C_eps = {}

    for reg, C_eps in eps_level_sets.items():
        D_ieps = perform_fast_marching_isotropic(
            np.ones_like(I_dilat), C_eps.T)
        distance_C_eps[reg] = D_ieps

    Distance = np.stack(list(distance_C_eps.values()), axis=-1)
    Indices = np.argmin(Distance, axis=-1)
    Indices_reduced = skimage.measure.block_reduce(
        Indices, (dilat_size, dilat_size), np.median)

    attrib_regions = Indices_reduced.copy()
    for i, reg in enumerate(unique_regions):
        attrib_regions[Indices_reduced == i] = reg

    return attrib_regions, Indices_reduced, Indices


def merge_regions(regions, contours_regions, ci, thresh_merge):
    """
    Merges regions that have close mean intensity

    Input :
            - regions : region map nxp
            - contours_regions : list of adjacent regions
            - ci : mean intensity per region (dict)
            - thresh_merge : threshold under which we merge (float)
    """
    for neigh in np.unique(contours_regions, axis=0):
        if np.abs(ci[neigh[0]] - ci[neigh[1]]) <= thresh_merge:
            print(f"Merging regions {neigh[0]} and {neigh[1]}")
            regions[regions == neigh[1]] = neigh[0]
    return regions
