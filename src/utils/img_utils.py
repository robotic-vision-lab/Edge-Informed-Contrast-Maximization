import sys
from typing import Any
from typing import Optional
from typing import Union

import cv2 as cv
import jax
import jax.image as jim
import jax.numpy as jnp
import numpy as onp
from jax import Array as JaxArray
from jax.typing import ArrayLike as JaxArrayLike
from numpy.typing import NDArray
from scipy import ndimage
from scipy.stats import norm


EPSN = sys.float_info.epsilon
INT_BIG = 2**31-1
DIST_SURF_SATURATION_DISTANCE = 6
ALPHA_IEDT = DIST_SURF_SATURATION_DISTANCE/5.541


def normalize_to_unit_range(arr: JaxArray | NDArray) -> JaxArray | NDArray:
    return (arr - arr.min()) / (arr.max() - arr.min() + EPSN)


def convert_to_grayscale(
        img: NDArray[Union[onp.uint8, onp.float32]]
) -> NDArray[Union[onp.uint8, onp.float32]]:
    """ returns grayscale of the input image """
    # check if image shape has 3 dimensions before converting
    if len(img.shape) == 3:
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    else:
        img_gray = img

    return img_gray


def convert_grayscale_to_BGR(
        img: NDArray[Union[onp.uint8, onp.float32]]
) -> NDArray[Union[onp.uint8, onp.float32]]:
    """ returns BGR image given a grayscale image """
    # check if image shape has 2 dimensions before converting
    if len(img.shape) == 2:
        img_BGR = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    else:
        img_BGR = img

    return img_BGR


def convert_BGR_to_RGB(
        img: NDArray[Union[onp.uint8, onp.float32]]
) -> NDArray[Union[onp.uint8, onp.float32]]:
    """ converts the image from BGR (OpenCV) to RGB """
    return cv.cvtColor(img, cv.COLOR_BGR2RGB)


def ocv_normalize_to_unit_range(img: NDArray[Union[onp.uint8, onp.float32]]) -> NDArray[onp.float32]:
    """ takes in an image and normalizes to range 0.0 to 1.0  """
    return cv.normalize(img.astype('float32'), None, 0.0, 1.0, norm_type=cv.NORM_MINMAX)


def ocv_normalize_to_255_range(img: NDArray[Union[onp.uint8, onp.float32]]) -> NDArray[onp.uint8]:
    """ takes in an image and normalizes to range 0 to 255  """
    return cv.normalize(img, None, 0, 255, norm_type=cv.NORM_MINMAX).astype('uint8') 


def jnp_to_onp(
        x_jnp: JaxArrayLike,
        dtype: Optional[Any] = onp.float64
) -> NDArray:
    x_onp = onp.array(x_jnp).astype(dtype)
    return x_onp


def jnp_to_ocv(
        x_jnp: JaxArrayLike,
        dtype: Optional[Any] = onp.float32
) -> NDArray:
    x_onp = jnp_to_onp(x_jnp, dtype)
    return x_onp


def jnp_to_ocv_n255(
        x_jnp: JaxArrayLike,
        dtype: Optional[Any] = onp.uint8
) -> NDArray[onp.uint8]:
    x_onp = onp.array(x_jnp).astype(onp.float32)
    x_onp = cv.normalize(x_onp, None, 0, 255, norm_type=cv.NORM_MINMAX).astype(dtype)
    return x_onp


def jnp_to_ocv_n1(
        x_jnp: JaxArrayLike,
        dtype: Optional[Any] = onp.float32
) -> NDArray[onp.float32]:
    x_onp = onp.array(x_jnp).astype(onp.float32)
    x_onp = cv.normalize(x_onp, None, 0.0, 1.0, norm_type=cv.NORM_MINMAX).astype(dtype)
    return x_onp


def extract_tiles(arr: JaxArrayLike, tile_h: int, tile_w: int) -> JaxArray:
    arr_h, arr_w = arr.shape
    n_row_tiles = arr_h // tile_h
    n_col_tiles = arr_w // tile_w

    tiles = []

    for row_i in range(n_row_tiles):
        for col_i in range(n_col_tiles):
            tile = arr[row_i * tile_h: (row_i + 1) * tile_h, col_i * tile_w : (col_i + 1) * tile_w]
            tiles.append(tile)

    # stack tiles
    stacked_tiles = jnp.stack(tiles)

    return stacked_tiles


def blend_two_imgs(img1, img2, img1_alpha=0.5):
    blended_img = cv.addWeighted(img1, img1_alpha,
                                 img2, 1 - img1_alpha,
                                 0)
    
    return blended_img


def preprocess_image(img: JaxArrayLike,
                     denoise_h=4,
                     denoise_template_win_size=3,
                     denoise_search_win_size=11,
                     clahe_clip_limit=5,
                     clahe_tile_grid_size=(10, 10),
                     sharpen_kernel_size=3,
                     sharpen_sigma_x=2,
                     sharpen_alpha=1.5,
                     sharpen_beta=-0.5,
                     bilateral_filter_neigh_diameter=5,
                     bilateral_filter_sigma_color=15,
                     bilateral_filter_sigma_space=15) -> NDArray[onp.float64]:
    if not isinstance(img, onp.ndarray) or (hasattr(img, 'dtype') and not img.dtype == onp.uint8):
        img = jnp_to_ocv_n255(img)
    
    # non-local means denoise (src, dst, h, templateWindowSize, searchWindowSize, normType)
    h = denoise_h                                   # large h will completely remove noise but also remove details
    template_win_size = denoise_template_win_size   # template patch size to compute weight
    search_win_size = denoise_search_win_size       # window size to compute average weight
    norm_type = cv.NORM_L2                          # NORM_L2 or NORM_L1, norm used for weight calc
    d_img = cv.fastNlMeansDenoising(img, 
                                    None, 
                                    h, 
                                    template_win_size, 
                                    search_win_size)
        
    # contrast limited adaptive histogram eq
    clahe = cv.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_tile_grid_size)
    clahe_img = clahe.apply(d_img)

    # sharpen
    kernel_size = sharpen_kernel_size # Gaussian kernel size
    sigma_x = sharpen_sigma_x         # standard deviation in X direction
    sigma_y = 0                       # standard deviation in Y direction (equal to sigmaX if 0)
    blur_clahe_img = cv.GaussianBlur(clahe_img, 
                                     None, 
                                     kernel_size, 
                                     sigma_x, 
                                     sigma_y)

    alpha = sharpen_alpha   # weight of first image
    beta = sharpen_beta     # weight of second image
    gamma = 0               # scalar added to each term 
    sharp_clahe_img = cv.addWeighted(clahe_img, 
                                     alpha, 
                                     blur_clahe_img, 
                                     beta, gamma)

    # blur
    neigh_diameter = bilateral_filter_neigh_diameter  # pixel neighborhood diameter
    sigma_color = bilateral_filter_sigma_color        # sigma in color space
    sigma_space = bilateral_filter_sigma_space        # sigma in coord space
    filtered_sharp_clahe_img = cv.bilateralFilter(sharp_clahe_img, 
                                                  neigh_diameter, 
                                                  sigma_color, 
                                                  sigma_space)

    return filtered_sharp_clahe_img


def image_to_edge(img: NDArray[onp.uint8], 
                  apert_size=3, 
                  th1=30, th2 = 80) -> NDArray[onp.uint8]:
    # edge detection
    aperture_size = apert_size   # size of Sobel operator
    thresh1 = th1                # below this, sure-not-an-edge
    thresh2 = th2                # above this, sure-is-an-edge
    ...                          # between thresh1 and thresh2, is-edge-if-connected-to-other-edges
    l2_gradient = True  # whether to use L2 norm for gradient calculation
    edge_img = cv.Canny(img, 
                        thresh1, 
                        thresh2, 
                        None, 
                        aperture_size, 
                        l2_gradient)

    return edge_img

def smoothen_edges(edge_img: NDArray,
                   k_size=1,
                   sigma=1) -> NDArray[onp.float64]:
    # blunt edges
    edge_img = edge_img.astype(onp.float64)
    kernel_size = k_size # Gaussian kernel size
    sigma_x = sigma      # standard deviation in X direction
    sigma_y = 0          # standard deviation in Y direction (equal to sigmaX if 0)
    smooth_edge_img = cv.GaussianBlur(edge_img, None, kernel_size, sigma_x, sigma_y)

    return smooth_edge_img


def rtef_inv_exp_dist_transform(edge_img, dist_surf_saturation_distance, alpha_iedt, formulation):
    rtef_iedt = RTEF_IEDT(dist_surf_saturation_distance, alpha_iedt, formulation)
    egde_iedt = rtef_iedt.compute_edge_iedt(edge_img)
    return egde_iedt


def eincm_inv_exp_dist_transform(edge_img, alpha=6):
    euc_dist_transform = ndimage.distance_transform_edt(~(edge_img.astype('bool')))
    exp_dist_transform = 1 - onp.exp(-euc_dist_transform / alpha)
    inverse_exp_dist_transform = 1 - normalize_to_unit_range(exp_dist_transform)
    return inverse_exp_dist_transform 


class RTEF_IEDT:
    """
    
    Python re-implementation of the (originally C++) implementaion of distance surface computation
    (https://github.com/heudiasyc/rt_of_low_high_res_event_cameras/blob/master/src/distance_surface_cpu.cpp)
    from the paper "Real-Time Optical Flow for Vehicular Perception with Low- and High-Resolution Event Cameras," (RTEF).

    Computes the distance surface from an edge image, on CPU, using the exact method described in 
    https://pageperso.lif.univ-mrs.fr/~edouard.thiel/print/2007-geodis-thiel-coeurjolly.pdf 
    (the algorithm is given in the part 5.4.2), which was itself inspired by the method described in the following paper: 
    http://fab.cba.mit.edu/classes/S62.12/docs/Meijster_distance.pdf.

    In our paper EINCM, serves the purpose of smoothing edges through inverse exponential distance transforms (IEDT).
    
    """
    def __init__(self, distance_surface_saturation_distance=None, alpha=None, formulation='exponential'):

        if distance_surface_saturation_distance is not None and alpha is not None:
            print(f'Both {distance_surface_saturation_distance=} and {alpha=} provided. Ignoring distance_surface_saturation_distance.')

        self.d_sat = distance_surface_saturation_distance if distance_surface_saturation_distance is not None else 6.0
        self.alpha = alpha if alpha is not None else self.d_sat/5.541
        self.formulation = formulation

        self.BIG_INT = onp.iinfo(onp.int32).max


    def initialize(self, edge_img):
        self.edge_img = edge_img.astype(onp.bool_)
        self.n_rows, self.n_cols = self.edge_img.shape
        self.map_x = onp.zeros((self.n_rows, self.n_cols), dtype=onp.int32)
        self.euclidean_distance_surface = onp.zeros((self.n_rows, self.n_cols), dtype=onp.int32)



    def parabola_ordinate(self, cur_col, row_origin, row_query):
        """Computes the parabola ordinate (the F^i_y(j) function described in 
        http://fab.cba.mit.edu/classes/S62.12/docs/Meijster_distance.pdf)

        Args:
            cur_col (int): Current column
            row_origin (int): Row of origin of the parabola
            row_query (int): Row for which we want to compute the ordinate

        Returns:
            onp.int32: The parabola ordinate at the given row coordinates
        """
        if self.map_x[row_origin, cur_col] == self.BIG_INT:
            return self.BIG_INT
        
        return onp.int32(
            self.map_x[row_origin, cur_col]**2 + (row_query - row_origin)**2
        )


    def parabolas_intersection_abscissa(self, cur_col, row_1, row_2): 
        """Computes the abscissa of the intersection of two consecutive parabolas (the Sep^i(u, v) function described in 
        http://fab.cba.mit.edu/classes/S62.12/docs/Meijster_distance.pdf)

        Args:
            cur_col (int): Current column
            row_1 (int): The row (abscissa) of the first parabolaption
            row_2 (int): The row (abscissa) of the second parabola

        Returns:
            onp.int32: The abscissa of intersection of the two parabolas
        """
        if self.map_x[row_1, cur_col] == self.BIG_INT or self.map_x[row_2, cur_col] == self.BIG_INT:
            return self.BIG_INT
        
        return onp.int32(
            (
                row_2**2 - row_1**2 
                + self.map_x[row_2, cur_col]**2 - self.map_x[row_1, cur_col]**2
            ) // (2*(row_2 - row_1))
        ) 
    

    def construct_map_x(self):
        """Computation of the distance transform: 
            Step 1/2: compute values for all row lines 
        """
        for row in range(self.n_rows):
            # setting the value for the first column
            self.map_x[row, 0] = 0 if self.edge_img[row, 0] else self.BIG_INT

            # forward pass (going to the right)
            for col in range(1, self.n_cols):
                if self.edge_img[row, col]:
                    self.map_x[row, col] = 0
                else:
                    self.map_x[row, col] = self.map_x[row, col - 1] + 1 if not self.map_x[row, col - 1] == self.BIG_INT else self.BIG_INT

            # backward pass (going to the left)
            for col in reversed(range(self.n_cols - 1)):
                if self.map_x[row, col] > self.map_x[row, col + 1]:
                    self.map_x[row, col] = self.map_x[row, col + 1] + 1


    def compute_euclidean_distance_transform(self):
        """Computation of the distance transform: 
            Step 2/2: compute the final map using the one computed in step 1 
        """
        # compute final map
        for col in range(self.n_cols):
            # initialize variables (refer http://fab.cba.mit.edu/classes/S62.12/docs/Meijster_distance.pdf)
            q = 0
            s = onp.zeros((self.n_rows), dtype=onp.int32)
            t = onp.zeros((self.n_rows), dtype=onp.int32)

            # For each row of the current column (col), the best segment of parabola is searched and is added to
            # the stack (or it replaces the whole stack if it is better than all the other segments)
            for row in range(1, self.n_rows):
                while q >=0 and self.parabola_ordinate(col, s[q], t[q]) > self.parabola_ordinate(col, row, t[q]):
                    q -=1 

                if q < 0:
                    q = 0
                    s[0] = row
                else:
                    parab_inter = self.parabolas_intersection_abscissa(col, s[q], row)
                    if not parab_inter == self.BIG_INT:
                        w = 1 + parab_inter
                        if w >= 0 and w < self.n_rows:
                            q += 1
                            s[q] = row
                            t[q] = w

            # Once all the segments for the column were determined, the values are computed and attributed
            # to the cells, and the segments of hyperbolas are removed from the stack once the next best
            # segment is reached                    
            for row in reversed(range(self.n_rows)):
                self.euclidean_distance_surface[row, col] = self.parabola_ordinate(col, s[q], row) 
                if row == t[q]:
                    q -= 1

        # since the value of each cell is the squared distance, we compute the sqrt to obtain the euclidean distance
        self.euclidean_distance_surface = onp.sqrt(onp.abs(self.euclidean_distance_surface.astype(onp.float64)))

        # If a formulation other than "linear" is used, a final step has to be computed to apply the correct formulation
        if not self.formulation == 'linear':
            if self.formulation == 'linear-bound':
                self.euclidean_distance_surface = onp.minimum(self.euclidean_distance_surface, self.d_sat)
            elif self.formulation == 'logarithmic':
                self.euclidean_distance_surface = onp.log(self.euclidean_distance_surface + 1.0)
            elif self.formulation == 'exponential':
                self.euclidean_distance_surface = 1 - onp.exp(-self.euclidean_distance_surface / self.alpha)
            else:
                print(f'Invalid option: {self.formulation=}')
                raise NotImplementedError

        # finally normalize
        self.euclidean_distance_surface = (
            (self.euclidean_distance_surface - self.euclidean_distance_surface.min()) 
            / (self.euclidean_distance_surface.max() - self.euclidean_distance_surface.min() + sys.float_info.epsilon)
        )

        return self.euclidean_distance_surface
    

    def compute_edge_iedt(self, edge_img):
        assert (len(edge_img.shape) == 2 
                and len(set(edge_img.flatten())) == 2
                and 0 in set(edge_img.astype('int').flatten())), 'Need 2D binary edge image'
        
        self.initialize(edge_img)

        # compute phase 1 and phase 2
        self.construct_map_x()
        self.compute_euclidean_distance_transform()

        # construct iedt
        edge_iedt = 1 - self.euclidean_distance_surface

        return edge_iedt



@jax.jit
def sobel_scharr_optimized_image_grads(image: JaxArray) -> JaxArray:

    scharr_gx = jnp.array([[3.0, 0.0, -3.0], [10.0, 0.0, -10.0], [3.0, 0.0, -3.0]])
    scharr_gy = jnp.array([[3.0, 10.0, 3.0], [0.0, 0.0, 0.0], [-3.0, -10.0, -3.0]])
    
    I_x = jax.scipy.signal.convolve(image, scharr_gx, mode='same')
    I_y = jax.scipy.signal.convolve(image, scharr_gy, mode='same')
    
    stacked_grads = jnp.stack([I_x, I_y], axis=-1) # (H, W, 2)

    return stacked_grads


@jax.jit
def gaussian_blur(image: JaxArray) -> JaxArray:
    kern = jnp.array([[1/16, 1/8, 1/16], [1/8, 1/4, 1/8], [1/16, 1/8, 1/16]])
    blur_img = jax.scipy.signal.convolve(image, kern, mode='same')
    return blur_img


@jax.jit
def gradient_magnitude(image: JaxArray) -> JaxArray:

    img_grads = sobel_scharr_optimized_image_grads(image) # (H, W, 2)
    I_x = img_grads[..., 0]
    I_y = img_grads[..., 1]

    # # only positive edges
    # I_x = jnp.where(I_x > 0.0, I_x, jnp.zeros_like(I_x))
    # I_y = jnp.where(I_y > 0.0, I_y, jnp.zeros_like(I_y))

    mag = (I_x**2 + I_y**2) **0.5
    mag = (mag - mag.min()) / (mag.max() - mag.min() + EPSN)

    return mag
