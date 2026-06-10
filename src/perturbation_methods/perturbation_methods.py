"""
Sensor-aware perturbations for Sentinel-2 7-channel patches and PhiSat-2 8-channel patches.

Expected channel orders used by this project:
    Sentinel-2 7ch: [B02, B03, B04, B05, B06, B07, B08]
    PhiSat-2 8ch:  [MS1, MS2, MS3, PAN, MS4, MS5, MS6, MS7]

Source references used for the constants:
    - Sentinel-2 SNR and Lref values:
      Copernicus SentiWiki, Sentinel-2 mission page, Table 3: spectral information and
      associated SNR per band for S2A/S2B. The table gives "Radiance sensibility range
      Lmin < Lref < Lmax" in W m^-2 sr^-1 um^-1 and "SNR @ Lref".
      URL: https://sentiwiki.copernicus.eu/web/s2-mission

    - Sentinel-2 MTF@Nyquist:
      Copernicus SentiWiki, Sentinel-2 mission page, radiometric performance section.
      The page states the MTF requirement as 0.15 to 0.3 for 10 m bands and <0.45 for
      20 m and 60 m bands. This file uses 0.225, the midpoint of 0.15--0.30, as a
      conservative single default for B02--B08. If exact per-band PSFs/MTFs are needed,
      replace SENTINEL2_MTF_NYQUIST_7 with band-specific measured values.

    - PhiSat-2 channel mapping and simulator equations:
      ESA PhiSat-2 Mission Overview for the #ORBITALAI challenge. Table 7 maps
      Sentinel-2 B02--B08 to PhiSat-2 MS1--MS7 and gives the PhiSat-2 PAN band. The
      document states that the PAN band is simulated from Sentinel-2 B02--B06 using
      spectral-response-function overlap weights. Section 3.1.3 defines the sensor-noise
      model as noise(bn) = Lref(bn) / SNR(bn) * N(0, 1), added to Top-of-Atmosphere
      radiance in W m^-2 sr^-1 um^-1.
      URL: https://challenges.philab.esa.int/wp-content/uploads/2023/02/Phisat-2_Mission_Overview_Web.pdf

Important unit note:
    The published Lref values are radiance values. Therefore, perturb_snr(...,
    use_official_lref=True) is physically meaningful only when x is still Top-of-Atmosphere
    radiance in W m^-2 sr^-1 um^-1, before reflectance conversion, sqrt transforms,
    clipping, or mean/std normalization. If x is already reflectance, normalized, or in an
    arbitrary data scale, set use_official_lref=False to use the patch mean as an empirical
    Lref in the same units as x.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.signal import fftconvolve


# -----------------------------------------------------------------------------
# Channel conventions
# -----------------------------------------------------------------------------

SENTINEL2_CHANNELS_7 = ("B02", "B03", "B04", "B05", "B06", "B07", "B08")
PHISAT2_CHANNELS_8 = ("MS1", "MS2", "MS3", "PAN", "MS4", "MS5", "MS6", "MS7")
PHISAT2_PAN_INDEX = 3


# -----------------------------------------------------------------------------
# Sentinel-2 radiometric constants for channels [B02, B03, B04, B05, B06, B07, B08]
# -----------------------------------------------------------------------------
# Copernicus SentiWiki Table 3 gives, per Sentinel-2 band, the reference radiance
# Lref and SNR measured at that Lref. Units for Lref are W m^-2 sr^-1 um^-1.
# We use S2A/S2B values because they are the values most commonly associated with
# the public Sentinel-2 MSI performance table and with the PhiSat-2 simulator input.

SENTINEL2_LREF_7 = np.array(
    [128.00, 128.00, 108.00, 74.60, 68.23, 66.70, 103.00],
    dtype=np.float32,
)

SENTINEL2_SNR_7 = np.array(
    [154.0, 168.0, 142.0, 117.0, 89.0, 105.0, 174.0],
    dtype=np.float32,
)

# MTF@Nyquist default. The SentiWiki radiometric-performance section gives a
# requirement of 0.15--0.30 for 10 m bands and <0.45 for 20/60 m bands. Since this
# project uses B02--B08 together and the augmentation is intended as a simple blur
# model, use the midpoint of the stricter 10 m range for all seven bands.
SENTINEL2_MTF_NYQUIST_7 = np.full(7, 0.225, dtype=np.float32)
SENTINEL2_MTF_NYQUIST_RANGE_10M = (0.15, 0.30)


# -----------------------------------------------------------------------------
# PhiSat-2 radiometric constants for channels [MS1, MS2, MS3, PAN, MS4, MS5, MS6, MS7]
# -----------------------------------------------------------------------------
# The PhiSat-2 public overview maps MS1..MS7 closely to Sentinel-2 B02..B08:
#   MS1~B02, MS2~B03, MS3~B04, MS4~B05, MS5~B06, MS6~B07, MS7~B08.
# It defines the PAN band as a linear combination of B02..B06 using overlap weights
# alpha_i / alpha_P. The public PDF gives the formula but not the actual alpha_i
# values. Therefore, PHISAT2_LREF_8 is an approximation derived from the Sentinel-2
# Lref values: MS bands inherit their corresponding Sentinel-2 Lref, and PAN uses an
# unweighted mean of B02..B06 Lref. If the alpha_i weights become available, replace
# the PAN entry with np.average(SENTINEL2_LREF_7[:5], weights=alpha_weights).

PHISAT2_LREF_PAN_APPROX = float(np.mean(SENTINEL2_LREF_7[:5]))  # B02..B06

PHISAT2_LREF_8 = np.array(
    [
        SENTINEL2_LREF_7[0],      # MS1 ~ S2 B02
        SENTINEL2_LREF_7[1],      # MS2 ~ S2 B03
        SENTINEL2_LREF_7[2],      # MS3 ~ S2 B04
        PHISAT2_LREF_PAN_APPROX,  # PAN ~ unweighted B02..B06 approximation
        SENTINEL2_LREF_7[3],      # MS4 ~ S2 B05
        SENTINEL2_LREF_7[4],      # MS5 ~ S2 B06
        SENTINEL2_LREF_7[5],      # MS6 ~ S2 B07
        SENTINEL2_LREF_7[6],      # MS7 ~ S2 B08
    ],
    dtype=np.float32,
)

# The PhiSat-2 PDF gives a public multispectral SNR range rather than per-MS-band
# values, and gives PAN SNR separately. Use the midpoint of 54--129 for MS bands and
# 256 for PAN. Keep the ranges here so paper text and ablations can state exactly
# what was assumed.
PHISAT2_MS_SNR_RANGE = (54.0, 129.0)
PHISAT2_MS_SNR_NOMINAL = float(np.mean(PHISAT2_MS_SNR_RANGE))  # 91.5
PHISAT2_PAN_SNR = 256.0

PHISAT2_SNR_8 = np.array(
    [
        PHISAT2_MS_SNR_NOMINAL,
        PHISAT2_MS_SNR_NOMINAL,
        PHISAT2_MS_SNR_NOMINAL,
        PHISAT2_PAN_SNR,
        PHISAT2_MS_SNR_NOMINAL,
        PHISAT2_MS_SNR_NOMINAL,
        PHISAT2_MS_SNR_NOMINAL,
        PHISAT2_MS_SNR_NOMINAL,
    ],
    dtype=np.float32,
)

# PhiSat-2 MTF@Nyquist range from the public mission overview. Use the midpoint as
# the nominal default. This is intentionally simple because the public document does
# not expose the full confidential MTF curve.
PHISAT2_MTF_NYQUIST_RANGE = (0.039, 0.072)
PHISAT2_MTF_NYQUIST_NOMINAL = float(np.mean(PHISAT2_MTF_NYQUIST_RANGE))  # 0.0555
PHISAT2_MTF_NYQUIST_8 = np.full(8, PHISAT2_MTF_NYQUIST_NOMINAL, dtype=np.float32)


# -----------------------------------------------------------------------------
# Sensor defaults
# -----------------------------------------------------------------------------

def sensor_defaults(num_channels: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return default (snr, lref, mtf_nyquist) arrays for the supported channel counts.

    Args:
        num_channels: 7 for Sentinel-2 or 8 for PhiSat-2.

    Returns:
        snr_b: per-band SNR values.
        lref_b: per-band reference radiance values in W m^-2 sr^-1 um^-1.
        mtf_b: per-band MTF at Nyquist frequency.
    """
    if num_channels == 7:
        return SENTINEL2_SNR_7.copy(), SENTINEL2_LREF_7.copy(), SENTINEL2_MTF_NYQUIST_7.copy()
    if num_channels == 8:
        return PHISAT2_SNR_8.copy(), PHISAT2_LREF_8.copy(), PHISAT2_MTF_NYQUIST_8.copy()
    raise ValueError(
        f"Unsupported number of channels: {num_channels}. "
        "Expected 7 for Sentinel-2 or 8 for PhiSat-2."
    )


def perturb_brightness(x: np.ndarray, alpha: float) -> np.ndarray:
    """
    Apply a global multiplicative brightness perturbation.

    Expected input:
        x: patch with shape (C, H, W), e.g. (8, 256, 256) or (7, 128, 128).
    """
    return (x.astype(np.float32) * alpha).astype(np.float32)


# -----------------------------------------------------------------------------
# SNR perturbation
# -----------------------------------------------------------------------------

def perturb_snr(
    x: np.ndarray,
    snr_b: np.ndarray | None = None,
    lref_b: np.ndarray | None = None,
    snr_factor: float = 1.0,
    seed: int | None = None,
    use_official_lref: bool = True,
) -> np.ndarray:
    """
    Add band-specific Gaussian sensor noise using SNR and Lref.

    The PhiSat-2 simulator document defines the noise model as:
        noise(bn, i, j) = Lref(bn) / SNR(bn) * N(0, 1)
        Lout(i, j) = Lin(i, j) + noise(bn, i, j)

    where Lref is the reference Top-of-Atmosphere radiance used to specify the SNR.
    Therefore, when use_official_lref=True, x should be TOA radiance in the same units
    as Lref: W m^-2 sr^-1 um^-1.

    If x is already reflectance, normalized data, square-root transformed data, or an
    arbitrary project-specific scale, use_official_lref=False. In that case, the function
    estimates Lref as the per-band mean of x, so the noise standard deviation remains in
    the same units as the input array. This is less physically exact, but scale-consistent.

    Args:
        x: input image patch with shape (C, H, W).
        snr_b: optional per-band SNR array. If None, inferred from C.
        lref_b: optional per-band Lref array. If None and use_official_lref=True,
            inferred from C. If use_official_lref=False, this argument is ignored.
        snr_factor: multiplies nominal SNR. Lower values add more noise:
            1.0 nominal, 0.75 mild degradation, 0.5 medium, 0.25 severe.
        seed: optional random seed.
        use_official_lref: True uses published/approximated radiance Lref; False uses
            x.mean(axis=(1, 2)) as a data-scale empirical Lref.
    """
    if x.ndim != 3:
        raise ValueError(f"x must have shape (C, H, W), got {x.shape}.")

    c, _, _ = x.shape
    x = x.astype(np.float32, copy=False)
    rng = np.random.default_rng(seed)

    default_snr_b, default_lref_b, _ = sensor_defaults(c)

    if snr_b is None:
        snr_nominal_b = default_snr_b
    else:
        snr_nominal_b = np.asarray(snr_b, dtype=np.float32)

    if snr_nominal_b.shape != (c,):
        raise ValueError(f"snr_b must have shape ({c},), got {snr_nominal_b.shape}.")

    snr_effective_b = snr_nominal_b * float(snr_factor)

    if use_official_lref:
        if lref_b is None:
            lref = default_lref_b
        else:
            lref = np.asarray(lref_b, dtype=np.float32)
            if lref.shape != (c,):
                raise ValueError(f"lref_b must have shape ({c},), got {lref.shape}.")
    else:
        # Empirical fallback for non-radiance or normalized input units.
        lref = x.mean(axis=(1, 2)).astype(np.float32)

    sigma_b = lref / (snr_effective_b + 1e-12)

    noise = rng.normal(
        loc=0.0,
        scale=sigma_b[:, None, None],
        size=x.shape,
    ).astype(np.float32)

    return (x + noise).astype(np.float32)


def perturb_snr_sentinel2(
    x: np.ndarray,
    snr_factor: float = 1.0,
    seed: int | None = None,
    use_official_lref: bool = True,
) -> np.ndarray:
    """
    SNR perturbation for Sentinel-2 [B02, B03, B04, B05, B06, B07, B08].

    Uses Sentinel-2 SentiWiki SNR@Lref and Lref values by default.
    """
    return perturb_snr(
        x,
        snr_b=SENTINEL2_SNR_7,
        lref_b=SENTINEL2_LREF_7,
        snr_factor=snr_factor,
        seed=seed,
        use_official_lref=use_official_lref,
    )


def perturb_snr_phisat2(
    x: np.ndarray,
    snr_factor: float = 1.0,
    seed: int | None = None,
    use_official_lref: bool = True,
) -> np.ndarray:
    """
    SNR perturbation for PhiSat-2 [MS1, MS2, MS3, PAN, MS4, MS5, MS6, MS7].

    Uses PhiSat-2 public SNR assumptions and Sentinel-2-derived Lref approximations.
    The PAN Lref is approximated as the unweighted mean of Sentinel-2 B02--B06 Lref.
    """
    return perturb_snr(
        x,
        snr_b=PHISAT2_SNR_8,
        lref_b=PHISAT2_LREF_8,
        snr_factor=snr_factor,
        seed=seed,
        use_official_lref=use_official_lref,
    )


# -----------------------------------------------------------------------------
# MTF / PSF perturbation
# -----------------------------------------------------------------------------

def precompute_k_grid(height: int, width: int) -> np.ndarray:
    """
    Precompute normalized radial frequency grid k.

    k = 0 is the DC/smooth component.
    k = 1 is the Nyquist frequency.
    """
    fy = np.fft.fftfreq(height)
    fx = np.fft.fftfreq(width)
    FX, FY = np.meshgrid(fx, fy)
    k = np.sqrt(FX**2 + FY**2) / 0.5
    k = np.clip(k, 0.0, 1.0)
    return k.astype(np.float32)


def mtf_to_psf(mtf_filter: np.ndarray) -> np.ndarray:
    """Convert a frequency-domain MTF filter to a normalized spatial PSF kernel."""
    psf = np.real(np.fft.ifft2(mtf_filter))
    psf = np.fft.fftshift(psf)
    psf = np.maximum(psf, 0.0)
    psf = psf / (psf.sum() + 1e-8)
    return psf.astype(np.float32)


def perturb_mtf(
    x: np.ndarray,
    mtf_nyquist: float | np.ndarray | None = None,
    k_grid: np.ndarray | None = None,
) -> np.ndarray:
    """
    Apply MTF-based optical blur.

    If mtf_nyquist is None, defaults are inferred from x.shape[0]:
        7 channels -> Sentinel-2 nominal MTF@Nyquist = 0.225 for B02--B08
        8 channels -> PhiSat-2 nominal MTF@Nyquist = 0.0555 for all bands

    mtf_nyquist can be either:
        - scalar float: same MTF@Nyquist for all bands
        - array with shape (C,): band-specific values

    The radial frequency response is approximated as:
        MTF(k) = exp(log(M) * k^2)
    where M is the MTF at Nyquist and k is normalized radial frequency.
    """
    if x.ndim != 3:
        raise ValueError(f"x must have shape (C, H, W), got {x.shape}.")

    c, h, w = x.shape
    x = x.astype(np.float32, copy=False)

    if k_grid is None:
        k_grid = precompute_k_grid(h, w)

    if mtf_nyquist is None:
        _, _, mtf_b = sensor_defaults(c)
    else:
        mtf_arr = np.asarray(mtf_nyquist, dtype=np.float32)
        if mtf_arr.ndim == 0:
            mtf_b = np.full(c, float(mtf_arr), dtype=np.float32)
        else:
            mtf_b = mtf_arr.astype(np.float32)

    if mtf_b.shape != (c,):
        raise ValueError(f"mtf_nyquist must be scalar or shape ({c},), got {mtf_b.shape}.")

    x_blur = np.empty_like(x, dtype=np.float32)

    for b in range(c):
        if not (0.0 < float(mtf_b[b]) <= 1.0):
            raise ValueError("MTF@Nyquist values must be in the interval (0, 1].")
        mtf_filter = np.exp(np.log(float(mtf_b[b])) * (k_grid ** 2)).astype(np.float32)
        psf = mtf_to_psf(mtf_filter)
        x_blur[b] = fftconvolve(x[b], psf, mode="same").astype(np.float32)

    return x_blur


def perturb_mtf_sentinel2(x: np.ndarray, k_grid: np.ndarray | None = None) -> np.ndarray:
    """MTF perturbation for Sentinel-2 [B02, B03, B04, B05, B06, B07, B08]."""
    return perturb_mtf(x, mtf_nyquist=SENTINEL2_MTF_NYQUIST_7, k_grid=k_grid)


def perturb_mtf_phisat2(x: np.ndarray, k_grid: np.ndarray | None = None) -> np.ndarray:
    """MTF perturbation for PhiSat-2 [MS1, MS2, MS3, PAN, MS4, MS5, MS6, MS7]."""
    return perturb_mtf(x, mtf_nyquist=PHISAT2_MTF_NYQUIST_8, k_grid=k_grid)


# -----------------------------------------------------------------------------
# Other perturbations
# -----------------------------------------------------------------------------

def perturb_band_misalignment(
    x: np.ndarray,
    max_shift_px: float,
    reference_band: int | None = "red",
    seed: int | None = None,
    order: int = 1,
    mode: str = "nearest",
) -> np.ndarray:
    """
    Apply random band-to-band misalignment.

    By default, the red band is kept fixed as the co-registration master band:
        PhiSat-2 8ch:    red = band 3
        Sentinel-2B 7ch: red = band 2

    If reference_band=None, all bands are shifted independently.
    If reference_band is an int, that band is kept fixed.
    """
    if x.ndim != 3:
        raise ValueError(f"x must have shape (C, H, W), got {x.shape}.")

    c, _, _ = x.shape

    if reference_band == "red":
        if c == 8:
            reference_band = 3
        elif c == 7:
            reference_band = 2
        else:
            raise ValueError(
                f"Cannot infer red reference band for {c} channels. "
                "Use reference_band as an int or None."
            )

    x = x.astype(np.float32, copy=False)
    rng = np.random.default_rng(seed)
    x_shifted = np.empty_like(x, dtype=np.float32)

    for b in range(c):
        if reference_band is not None and b == reference_band:
            x_shifted[b] = x[b]
            continue

        r = rng.uniform(0.0, max_shift_px)
        theta = rng.uniform(0.0, 2.0 * np.pi)

        dy = r * np.sin(theta)
        dx = r * np.cos(theta)

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

    Supports:
        (8, H, W)
        (7, H, W)

    Model:
        I_b = J_b * t + A_b * (1 - t)

    atmospheric_light:
        "p95", "p99", "max", or explicit np.ndarray with shape (C,).
    """
    if x.ndim != 3:
        raise ValueError(f"x must have shape (C, H, W), got {x.shape}.")

    c, _, _ = x.shape
    x = x.astype(np.float32, copy=False)

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
        if A_b.shape != (c,):
            raise ValueError(f"atmospheric_light array must have shape ({c},), got {A_b.shape}.")

    x_hazy = x * float(t) + A_b[:, None, None] * (1.0 - float(t))
    return x_hazy.astype(np.float32)
