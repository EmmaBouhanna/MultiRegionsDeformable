import numpy as np
import itertools 

from nt_toolbox.perform_fast_marching import perform_fast_marching_isotropic


def get_contour_eps_levels(mask, eps=1e-15):
    """
    Gets contours from binary regions (inside/outside) via an unbiaised method

    Input : mask = (n x p) array
    Output : contours (list of (x,y) coordinates)
    """
    n, p = mask.shape
    D = mask.copy()
    #horizontal
    P1 = D[0:(n-1),:]
    P2 = D[1:,:]
    P = ((P1*P2) < 0)

    d = abs(P1-P2)
    d[d < eps] = 1
    v1 = abs(P1)/d
    v2 = abs(P2)/d
    Ah = ((np.vstack((P,np.zeros([1,n]))) + np.vstack((np.zeros([1,n]),P))) > 0)
    Vh = np.maximum(np.vstack((v1,np.zeros([1,n]))), np.vstack((np.zeros([1,n]),v2)))

    # vertical
    P1 = D[:,0:(p-1)]
    P2 = D[:,1:]
    P = ((P1*P2) < 0)

    d = abs(P1-P2)
    d[d < eps] = 1
    v1 = abs(P1)/d
    v2 = abs(P2)/d
    Av = ((np.hstack((P,np.zeros([p,1]))) + np.hstack((np.zeros([p,1]),P))) > 0)
    Vv = np.maximum(np.hstack((v1,np.zeros([p,1]))), np.hstack((np.zeros([p,1]),v2)))

    V = np.zeros([n,p])
    I = np.where(Ah > 0)
    V[I] = Vh[I]
    I = np.where(Av > 0)
    V[I] = np.maximum(V[I],Vv[I])

    I = np.where(V != 0)
    x,y = I[0],I[1]
    start_points = np.vstack((x,y))

    return start_points

def get_contours_regions(regions):
    """
    Gets contours from a mask with multiple regions

    Input : regions = (n x p) array, type int
    Output : 
            - contours = list of (x,y) coordinates
            - contours_regions = list of (region1, region2) coords,
              with adjacent regions for each contour tuple 
    """
    unique_regions = np.unique(regions)
    new_contours = []
    contour_regions = []

    for comb in list(itertools.combinations(unique_regions, 2)):
        mask1 = (regions == comb[0])
        mask2 = (regions == comb[1])

        comb_mask = np.zeros_like(regions)
        comb_mask[mask1] = 1
        comb_mask[mask2] = -1

        conts = get_contour_eps_levels(comb_mask)

        if conts.any():
            new_contours.append(conts)

            cont_regions = [list(comb) for _ in range(conts.shape[1])]
            contour_regions.append(cont_regions)


    return np.hstack(new_contours).T, np.vstack(contour_regions)

def get_voronoi_cij(contours, 
                    contours_regions, 
                    unique_neighbors, 
                    shape_vor):
    """
    Gets Voronoi regions of each portion of the contour separating
    two regions i and j.

    Input :
            - contours : list of (x,y) coordinates
            - contours_regions : list of (i, j) adjacent regions
              for each contour tuple
            - unique_neighbors : unique combination of adjacent
              regions (array p x 2)
            - shape_vor : the shape of the voronoi mask
    
    Output : 
            - Voronoi : mask of voronoi influences for each contour Cij,
              array with shape shape_vor x 2
    """
    ## obtain region segments
    Cij = {}

    for neighbors in unique_neighbors:
        mask_ = contours_regions == neighbors
        mask_ = mask_[:, 0] * mask_[:, 1]
        pos_Cij = contours[mask_]
        Cij[tuple(neighbors.astype(int))] = pos_Cij
  
    ## obtain voronoi regions for each region segment using FM
    distance_Cij = {}

    for neighbors, pos_Cij in Cij.items():
        D_ij = perform_fast_marching_isotropic(np.ones(shape_vor), pos_Cij.T)
        distance_Cij[neighbors] = D_ij

    ## voronoi masks
    Distance = np.stack(list(distance_Cij.values()), axis=-1)
    Indices = np.argmin(Distance, axis=-1)
    Voronoi = np.zeros((shape_vor[0], shape_vor[1], 2))

    for i, neigh in enumerate(unique_neighbors):
        Voronoi[Indices == i] = neigh

    return Voronoi, Indices