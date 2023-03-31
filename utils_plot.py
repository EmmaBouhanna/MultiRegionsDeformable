import matplotlib.pyplot as plt

def two_plot(p1, p2):
    """
    Plots two images next to each other
    """
    plt.figure(figsize=(8,8))
    plt.subplot(1, 2, 1)
    plt.imshow(p1)
    plt.subplot(1, 2, 2)
    plt.imshow(p2)
    plt.show()

def plot_with_borders(p1, p2):
    plt.figure(figsize=(8,8))
    plt.subplot(1, 2, 1)
    plt.imshow(p1)
    plt.subplot(1, 2, 2)
    plt.imshow(p2)
    plt.show()

def plot_iteration(Phi, 
                   Voronoi, 
                   Phi_new, 
                   regions, 
                   regions_new, 
                   contours, 
                   I, 
                   Indices):
    plt.figure(figsize=(10,10))
    plt.subplot(2,3,1)
    plt.imshow(Phi)
    plt.title("Original Phi")
    plt.axis("off")

    plt.subplot(2,3,2)
    plt.imshow(Voronoi[:,:,0]*10 + Voronoi[:,:,1])
    plt.title("Voronoi regions Cij")
    plt.axis("off")

    plt.subplot(2,3,3)
    plt.imshow(Phi_new)
    plt.title("Updated Phi (with extended velocities)")
    plt.axis("off")

    plt.subplot(2,3,4)
    plt.imshow(regions)
    plt.imshow(I, alpha=0.2)
    plt.title("Original Regions")
    plt.axis("off")

    plt.subplot(2,3,5)
    plt.imshow(regions_new)
    plt.scatter(contours[:,1], contours[:,0], c="pink")
    plt.title("New Regions")
    plt.axis("off")


    plt.subplot(2,3,6)
    plt.imshow(Indices)
    plt.title("Indices new regions")
    plt.axis("off")

    plt.show()