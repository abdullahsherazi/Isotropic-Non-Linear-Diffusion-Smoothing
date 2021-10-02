from PIL import Image
from numpy import asarray   
from medpy.filter.smoothing import anisotropic_diffusion
import os 

# Opening the image 
folder_path = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(folder_path, 'office_noisy.png')
image = Image.open(image_path)

# asarray() class is used to convert
# PIL images into NumPy arrays
data = asarray(image)

# isotropic_diffusion function Parameters:	
# img ==> array_like(Input image (will be cast to numpy.float))

# niter ==> integer(Number of iterations) or Diffusion time

# kappa ==> integer(Conduction coefficient, e.g. 20-100. kappa controls conduction as a 
# function of the gradient. If kappa is low small intensity gradients are able to block 
# conduction and hence diffusion across steep edges. A large value reduces the influence 
# of intensity gradients on conduction. 
# (kappa ==> large value ==> λ ==> large value ==> D large value)

# gamma ==> float(Controls the speed of diffusion). Pick a value <=.25 for stability.

# voxelspacing ==> tuple of floats or array_like(The distance between adjacent 
# pixels in all img.ndim directions)

# option ==> {1, 2, 3} (Whether to use the Perona Malik diffusion equation 
# No. 1 or No. 2, or Tukey’s biweight function. Equation 1 favours high 
# contrast edges over low contrast ones, while equation 2 favours wide 
# regions over smaller ones. Equation 3 preserves 
# sharper boundaries than previous formulations and improves the automatic 
# stopping of the diffusion)

# Returns==> isotropic_diffusion : Diffused image. ndarray

img_filtered = anisotropic_diffusion(data, niter=10, kappa=20, gamma=0.25, voxelspacing=None, option=1)

# creating Pillow image from our numpyarray
img = Image.fromarray(img_filtered)

# Displaying the image
img.show()





