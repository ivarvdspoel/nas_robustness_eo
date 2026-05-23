import numpy as np
from scipy import ndimage
from scipy.signal import fftconvolve


def perturb_brightness(x: np.ndarray, alpha: float) -> np.ndarray:
    """
    Apply a global multiplicative brightness perturbation.

    Parameters
    ----------
    x : np.ndarray
        Input PhiSat-2 patch with shape (8, H, W).
    alpha : float
        Multiplicative brightness factor.
        alpha > 1 increases brightness.
        alpha < 1 decreases brightness.

    Returns
    -------
    np.ndarray
        Brightness-perturbed patch with the same shape as x.
    """
    return (x.astype(np.float32) * alpha).astype(np.float32)


def perturb_snr(
    x: np.ndarray,
    snr_b: np.ndarray | None = None,
    snr_factor: float = 1.0,
    pan_index: int = 3,
    pan_snr: float = 256.0,
    seed: int | None = None,
) -> np.ndarray:
    """
    Add band-specific Gaussian noise using degraded effective SNR.

    snr_factor:
        1.0  = nominal SNR
        0.75 = mild degradation
        0.50 = medium degradation
        0.25 = severe degradation
    """
    if x.ndim != 3 or x.shape[0] != 8:
        raise ValueError(f"Expected x with shape (8, H, W), got {x.shape}")

    if snr_factor <= 0:
        raise ValueError("snr_factor must be > 0.")

    if not 0 <= pan_index < 8:
        raise ValueError(f"pan_index must be in [0, 7], got {pan_index}")

    x = x.astype(np.float32, copy=False)
    rng = np.random.default_rng(seed)

    if snr_b is None:
        snr_nominal_b = np.full(8, (54.0 + 192.0) / 2.0, dtype=np.float32)
        snr_nominal_b[pan_index] = pan_snr
    else:
        snr_nominal_b = np.asarray(snr_b, dtype=np.float32)
        if snr_nominal_b.shape != (8,):
            raise ValueError(f"snr_b must have shape (8,), got {snr_nominal_b.shape}")

    snr_effective_b = snr_nominal_b * snr_factor

    L_ref_b = x.mean(axis=(1, 2)).astype(np.float32)
    sigma_b = L_ref_b / snr_effective_b

    noise = rng.normal(
        loc=0.0,
        scale=sigma_b[:, None, None],
        size=x.shape,
    ).astype(np.float32)

    x_noisy = x + noise

    return x_noisy.astype(np.float32)



def precompute_k_grid(height: int = 256, width: int = 256) -> np.ndarray:
    """
    Precompute normalized radial frequency grid k.

    k = 0 is the DC / smooth component.
    k = 1 is the Nyquist frequency.
    """
    fy = np.fft.fftfreq(height)
    fx = np.fft.fftfreq(width)

    FX, FY = np.meshgrid(fx, fy)

    # Nyquist frequency is 0.5 cycles / pixel
    k = np.sqrt(FX**2 + FY**2) / 0.5
    k = np.clip(k, 0.0, 1.0)

    return k.astype(np.float32)



def mtf_to_psf(mtf_filter: np.ndarray) -> np.ndarray:
    """
    Convert frequency-domain MTF filter to spatial-domain PSF kernel.
    """
    psf = np.real(np.fft.ifft2(mtf_filter))
    psf = np.fft.fftshift(psf)

    # Numerical cleanup and normalization
    psf = np.maximum(psf, 0)
    psf = psf / (psf.sum() + 1e-8)

    return psf.astype(np.float32)


def perturb_mtf(
    x: np.ndarray,
    mtf_nyquist: float,
    k_grid: np.ndarray | None = None,
) -> np.ndarray:
    """
    Apply MTF-based optical blur by explicitly converting MTF to PSF.

    MTF(k) = exp(log(M) * k^2)

    Lower mtf_nyquist means stronger blur.
    """
    if x.ndim != 3 or x.shape[0] != 8:
        raise ValueError(f"Expected x with shape (8, H, W), got {x.shape}")

    if not (0.0 < mtf_nyquist <= 1.0):
        raise ValueError("mtf_nyquist must be in (0, 1].")

    x = x.astype(np.float32, copy=False)
    _, h, w = x.shape

    if k_grid is None:
        k_grid = precompute_k_grid(h, w)
    else:
        if k_grid.shape != (h, w):
            raise ValueError(f"k_grid must have shape {(h, w)}, got {k_grid.shape}")

    mtf_filter = np.exp(np.log(mtf_nyquist) * (k_grid ** 2)).astype(np.float32)
    psf = mtf_to_psf(mtf_filter)

    x_blur = np.empty_like(x, dtype=np.float32)

    for b in range(x.shape[0]):
        x_blur[b] = fftconvolve(x[b], psf, mode="same").astype(np.float32)

    return x_blur


def perturb_band_misalignment(
    x: np.ndarray,
    max_shift_px: float,
    reference_band: int = 3,
    seed: int | None = None,
    order: int = 1,
    mode: str = "nearest",
):
    """
    Apply random band-to-band misalignment.

    Expected input:
        x: PhiSat-2 patch with shape (8, H, W)

    Model:
        x'_b(i, j) = x_b(i + dy_b, j + dx_b)

    The reference band is kept fixed. All other bands are shifted randomly.

    Parameters
    ----------
    max_shift_px:
        Maximum shift magnitude in pixels.
        Example: 1 = realistic/typical, 3 = bad, 10 = extreme observed case.

    reference_band:
        Band kept fixed. Use 3 if PAN is your master/reference band.

    order:
        Interpolation order.
        0 = nearest, 1 = bilinear, 3 = bicubic.

    mode:
        Boundary handling for shifted pixels.
    """
    if x.ndim != 3 or x.shape[0] != 8:
        raise ValueError(f"Expected x with shape (8, H, W), got {x.shape}")

    x = x.astype(np.float32, copy=False)
    rng = np.random.default_rng(seed)

    x_shifted = np.empty_like(x, dtype=np.float32)
    shifts = np.zeros((8, 2), dtype=np.float32)  # dy, dx

    for b in range(8):
        if b == reference_band:
            x_shifted[b] = x[b]
            continue

        r = rng.uniform(0.0, max_shift_px)
        theta = rng.uniform(0.0, 2.0 * np.pi)

        dy = r * np.sin(theta)
        dx = r * np.cos(theta)

        shifts[b] = [dy, dx]

        x_shifted[b] = ndimage.shift(
            x[b],
            shift=(dy, dx),
            order=order,
            mode=mode,
            prefilter=(order > 1),
        ).astype(np.float32)

    return x_shifted

def perturb_haze(
    x: np.ndarray,
    t: float,
    atmospheric_light: str | np.ndarray = "p95",
) -> np.ndarray:
    """
    Add haze using the atmospheric scattering model.

    Expected input:
        x: PhiSat-2 patch with shape (8, H, W)

    Model:
        I_b = J_b * t + A_b * (1 - t)

    where:
        J_b = clean band
        I_b = hazy band
        t   = transmission, shared across bands
        A_b = atmospheric light for band b

    Parameters
    ----------
    t : float
        Transmission.
        t close to 1.0 = little haze.
        t close to 0.0 = strong haze.

    atmospheric_light:
        "p95", "p99", "max", or explicit np.ndarray with shape (8,).
    """
    if x.ndim != 3 or x.shape[0] != 8:
        raise ValueError(f"Expected x with shape (8, H, W), got {x.shape}")

    if not (0.0 < t <= 1.0):
        raise ValueError("t must be in (0, 1].")

    x = x.astype(np.float32, copy=False)

    # Atmospheric light A_b per band
    if isinstance(atmospheric_light, str):
        if atmospheric_light == "p95":
            A_b = np.percentile(x, 95, axis=(1, 2)).astype(np.float32)
        elif atmospheric_light == "p99":
            A_b = np.percentile(x, 99, axis=(1, 2)).astype(np.float32)
        elif atmospheric_light == "max":
            A_b = np.max(x, axis=(1, 2)).astype(np.float32)
        else:
            raise ValueError("atmospheric_light must be 'p95', 'p99', 'max', or array.")
    else:
        A_b = np.asarray(atmospheric_light, dtype=np.float32)
        if A_b.shape != (8,):
            raise ValueError(f"atmospheric_light must have shape (8,), got {A_b.shape}")

    x_hazy = x * t + A_b[:, None, None] * (1.0 - t)

    return x_hazy.astype(np.float32)
