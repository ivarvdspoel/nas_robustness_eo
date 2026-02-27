# File with common perturbation functions found in the literature
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve


class OpticalSystem:
    def __init__(self, 
                 sensor, 
                 M_b_n_TGT, 
                 SNR,
                 wavelength=0.0008, 
                 l_ref=100.0):
        
        # sensor: must be either 'sentinel' or 'venus'
        self.sensor = sensor
        M_b_n_SRC = self.stored_params(sensor)['MTF_at_nyquist']
        self.M_b_SRC = M_b_n_SRC
        
        self.wavelength = wavelength  # Wavelength in mm (800 nm)
        pixel_pitch = self.stored_params(sensor)['pixel_pitch']
        f_number = self.stored_params(sensor)['f_number']
        self.l_ref = l_ref  # Reference luminance for noise addition
        
        assert M_b_n_TGT <= M_b_n_SRC, 'Target MTF must be less than or equal to the source MTF'
        self.M_b_n_TGT = M_b_n_TGT # 
        assert SNR > 0, 'SNR must be greater than 0'
        assert M_b_n_SRC > 0, 'MTF at Nyquist frequency must be greater than 0'
        # Kernel Config:
        self.SNR = SNR
        kernel_size = 7  # Size of the kernel
        self.mtf_kernel = self.compute_mtf_kernel(pixel_pitch, f_number, M_b_n_SRC, M_b_n_TGT, wavelength, kernel_size)
        self.psf_kernel = self.compute_psf_kernel(self.mtf_kernel)
        

    def compute_mtf_kernel(self, pixel_pitch, f_number, M_b_n_SRC, M_b_n_TGT, wavelength, kernel_size):
        """
        Computes a 2D MTF kernel based on instrument parameters, including F-number.
        
        Args:
        - pixel_pitch (float): Pixel pitch (sampling frequency inverse).
        - f_number (float): F-number of the optical system.
        - M_b_n_SRC (float): MTF amplitude at the Nyquist frequency for the specific band (SRC original).
        - M_b_n_TGT (float): MTF amplitude at the Nyquist frequency for the specific band (Target MTF).
        - wavelength (float): Wavelength of the light (in meters) for which MTF is computed.
        - kernel_size (int): Size of the output MTF kernel (n x n). Should be odd.
        
        Returns:
        - mtf_kernel (numpy.ndarray): 2D MTF kernel of size (n x n).
        """
        # pixel_pitch, f_number, M_b_n_SRC, M_b_n_TGT, wavelength, kernel_size = self.pixel_pitch, self.f_number, self.M_b_n, self.M_b_n, self.wavelength, self.kernel_size
        
        # Compute sampling frequency (fs) and Nyquist frequency (fN)
        sampling_frequency = 1.0 / pixel_pitch  # Sampling frequency (fs)
        nyquist_frequency = sampling_frequency / 2  # Nyquist frequency (fN)
        
        # Compute cutoff frequency (fc) based on F-number and wavelength
        cutoff_frequency = 1 / (wavelength * f_number)  # Cutoff frequency (fc)
        normalized_cutoff_frequency = cutoff_frequency / nyquist_frequency  # Normalize with respect to Nyquist frequency
        
        # Create a frequency grid (u, v) in normalized coordinates (-1 to 1)
        u = np.linspace(-1, 1, kernel_size)
        v = np.linspace(-1, 1, kernel_size)
        U, V = np.meshgrid(u, v)
        
        # Compute the radial frequency k(u, v)
        K = np.sqrt(U**2 + V**2)  # Radial spatial frequency in normalized units
        
        # Apply the MTF formula to compute the MTF values on the grid
        MTF = np.exp(np.log(M_b_n_TGT / M_b_n_SRC) * K**2)  # Gaussian hypotheis (OrbitalAI paper)
        
        # Set MTF to zero beyond the normalized cutoff frequency
        MTF[K > normalized_cutoff_frequency] = 0
        
        # Normalize the MTF kernel
        mtf_kernel = MTF / np.max(MTF)
        
        return mtf_kernel


    @staticmethod
    def compute_psf_kernel(mtf_kernel):
        """
        Computes a 2D PSF kernel by taking the inverse Fourier transform of the MTF kernel.
        
        Args:
        - mtf_kernel (numpy.ndarray): 2D MTF kernel.
        
        Returns:
        - psf_kernel (numpy.ndarray): 2D PSF kernel.
        """
        # Compute the inverse 2D Fourier transform of the MTF kernel
        psf_kernel = np.fft.ifft2(mtf_kernel)
        
        # Shift the zero-frequency component to the center
        psf_kernel = np.fft.fftshift(psf_kernel)
        
        # Take the magnitude (real part) since the PSF is real-valued
        psf_kernel = np.abs(psf_kernel)
        
        # Normalize the PSF kernel to make its sum equal to 1
        psf_kernel /= np.sum(psf_kernel)
        
        return psf_kernel


    def stored_params(self, sensor):
        # Parameters
        # VENUS:
        # https://www.eoportal.org/satellite-missions/venus#VENuS.html.6
        # https://en.wikipedia.org/wiki/VENμS
        # Pixel pitch and F-number for VENµS Super Spectral Camera (VSSC)
        # f_number = 7.  # F-number for the VENµS VSSC
        # pixel_pitch = 0.0169  # Pixel pitch in mm 5200 pixels in 69 mm
        # MTF_at_nyquist = 0.2 # https://www.spiedigitallibrary.org/conference-proceedings-of-spie/11180/111804F/Measuring-modeling-and-removing-optical-straylight-from-venμs-super-spectral/10.1117/12.2536078.full
        VENUS = {'f_number': 7., 'pixel_pitch': 69/5200, 'MTF_at_nyquist': 0.15}
        
        
        
        
        
        
        ########################################################
        # Sentinel-2:
        aperture =  150 # mm 
        focal_length = 600 # mm
        # MTF_at_nyquist = 0.3 # From OrbitalAI paper.
        f_number = focal_length / aperture
        # pixel_pitch = 7.5 micro meters # Available in the paper: https://www.sciencedirect.com/topics/materials-science/sentinel-2
        SENTINEL = {'f_number': f_number, 'pixel_pitch': 0.0075, 'MTF_at_nyquist': 0.3, 'l_ref': 103., 'SNR': 174} # https://sentinels.copernicus.eu/en/web/sentinel/technical-guides/sentinel-2-msi/mission-performance
        
        if sensor == 'sentinel':
            return SENTINEL
        elif sensor == 'venus':
            return VENUS

    
    def plot_kernels(self):
        # Plot the PSF and MTF kernels side by side
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        # Plot the PSF kernel
        axes[0].imshow(self.psf, cmap='viridis', interpolation='nearest')
        axes[0].set_title("2D PSF Kernel")
        axes[0].set_xlabel('X-axis')
        axes[0].set_ylabel('Y-axis')
        # BEGIN: Add colorbar
        cbar = fig.colorbar(axes[0].images[0], ax=axes[0], orientation='vertical')
        # Plot the MTF kernel
        axes[1].imshow(self.mtf_kernel, cmap='viridis', interpolation='nearest')
        axes[1].set_title("2D MTF Kernel")
        axes[1].set_xlabel('X-axis')
        axes[1].set_ylabel('Y-axis')
        # BEGIN: Add colorbar
        cbar = fig.colorbar(axes[1].images[0], ax=axes[1], orientation='vertical')
        # END: Add colorbar

        plt.tight_layout()
        plt.show()


    def apply(self, img, normalize=True):
        """
        Adds noise to the image based on the provided SNR and applies a PSF kernel convolution.
        
        Parameters:
        img (ndarray): Input image.
        snr (float): Signal-to-noise ratio for noise addition. If 0 or less, no noise is added.
        psf_percentage (float): Percentage to control the range of the PSF kernel values.
        normalize (bool): Whether to normalize the output to match the original image's mean and std.
        ks (int): Kernel size for PSF convolution.
        l_ref (float): Scaling factor for noise addition.
        
        Returns:
        ndarray: The image after noise addition and PSF convolution.
        """
        l_ref = self.l_ref
        # Step 2: Generate random noise with zero mean and unit variance
        random_noise = np.random.normal(loc=0, scale=1, size=img.shape)

        # Step 3: Add noise to the image based on SNR
        if self.SNR > 0:
            noise = l_ref * random_noise / self.SNR
            noisy_data = img + noise  # Scale noise by l_ref and snr
        else:
            noisy_data = img  # No noise added if SNR <= 0

        # Step 4: Convolve the noisy data with the normalized PSF kernel
            
        psf_convolved_data = convolve(noisy_data, self.psf_kernel, mode='mirror')
        
        # Step 5: Normalize output to match original data distribution if requested
        if normalize:
            mean_orig, std_orig = img.mean(), img.std()
            mean_conv, std_conv = psf_convolved_data.mean(), psf_convolved_data.std()
            # Avoid division by zero in normalization
            if std_conv != 0:
                adjusted_convolved_data = (psf_convolved_data - mean_conv) * (std_orig / std_conv) + mean_orig
            else:
                adjusted_convolved_data = psf_convolved_data  # If std_conv is 0, keep as is
            return adjusted_convolved_data
        
        else:
            return psf_convolved_data


# CONSTANTS
CROP_SIZE = (72, 72)
S2_RESOLUTION = 10
PHISAT2_RESOLUTION = 4.75
BBOX_SIZE = 20140  # in metres
BBOX_SIZE_CROPPED = 19456  # in metres
S2_BANDS = ["B02", "B03", "B04", "B08", "B05", "B06", "B07"]
S2_PAN_BANDS = ["B02", "B03", "B04", "PAN", "B08", "B05", "B06", "B07"]
L1A_ABSOLUTE_SHIFTS = [6.329, 5.889, 5.339, 4.622, 3.785, 2.842, 1.796, 0]
L1A_RELATIVE_SHIFTS = [
    0,
    1.105,
    1.046,
    0.943,
    0.837,
    0.717,
    0.55,
    0.44,
]  # these values are in reverse order wrt the absolute shifts
L1A_RAND_STD = 0.4
L1A_RAND_MEAN = 0.0
PAN_WEIGHTS = [0.21594369, 0.28731533, 0.25719303, 0.0, 0.12275664, 0.11679131, 0.0]


# class ProcessingLevels(Enum):
#     L1A = "L1A"
#     L1B = "L1B"
#     L1C = "L1C"


def get_shifts_l1a(num_bands):
    mis_amplitude = np.random.normal(L1A_RAND_MEAN, L1A_RAND_STD, size=(num_bands,)) \
                    + np.array(L1A_RELATIVE_SHIFTS[:num_bands])

    mis_angle = np.random.uniform(0, 2 * np.pi, size=(num_bands,))

    shifts = np.stack([
        mis_amplitude * np.cos(mis_angle),
        mis_amplitude * np.sin(mis_angle)
    ], axis=1)

    shifts[0] = [0.0, 0.0]  # reference band
    shifts = np.flip(np.cumsum(shifts, axis=0), axis=0)

    return shifts  # shape: (C, 2)

def get_shifts_l1b(num_bands, rand_std):
    mis_amplitude = np.random.normal(0, rand_std, size=(num_bands,))
    mis_angle = np.random.uniform(0, 2 * np.pi, size=(num_bands,))

    shifts = np.stack([
        mis_amplitude * np.cos(mis_angle),
        mis_amplitude * np.sin(mis_angle)
    ], axis=1)

    shifts[2] = [0.0, 0.0]  # reference band (like original)

    return shifts  # shape: (C, 2)

import cv2
import numpy as np

def apply_band_misalignment(
    img,
    processing_level="L1A",
    rand_std=1,
    interpolation=cv2.INTER_LINEAR,
    border_value=0,
    channels_first=True
):
    """
    Apply band misalignment to a numpy image.

    Args:
        img: np.ndarray
            Shape (C, H, W) or (H, W, C)
        processing_level: "L1A" or "L1B"
        rand_std: std for L1B shifts
        interpolation: cv2 interpolation
        border_value: fill value
        channels_first: whether input is (C, H, W)

    Returns:
        shifted_img: same shape as input
        shifts: (C, 2) array of (dx, dy)
    """

    # ---- Ensure (C, H, W) ----
    if not channels_first:
        img = np.moveaxis(img, -1, 0)

    C, H, W = img.shape

    # ---- Generate shifts ----
    if processing_level == "L1A":
        shifts = get_shifts_l1a(C)
    else:
        shifts = get_shifts_l1b(C, rand_std)

    # ---- Apply shifts ----
    shifted = np.zeros_like(img)

    for b in range(C):
        dx, dy = shifts[b]

        M = np.array([
            [1, 0, dx],
            [0, 1, dy]
        ], dtype=np.float32)

        shifted[b] = cv2.warpAffine(
            img[b],
            M,
            dsize=(W, H),
            flags=interpolation,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border_value
        )

    # ---- Restore original layout ----
    if not channels_first:
        shifted = np.moveaxis(shifted, 0, -1)

    return shifted, shifts

if __name__ == '__main__':
    # Example usage
    venus_optics = OpticalSystem('venus', 0.2, SNR=5)
    sen2_optics = OpticalSystem('sentinel', 0.3, SNR=5)
    
    # Load an example image
    img = np.random.rand(512, 512)
    img_w_noise = venus_optics.apply(img)
    img_w_noise_sen2 = sen2_optics.apply(img)
    # Plot the noisy image
    plt.imshow(img_w_noise, cmap='gray')
    
    print('Done')