import datetime
import os

import matplotlib.pyplot as plt
import cv2
import numpy as np
import rasterio
#import geopandas as gpd
from eolearn.core import (
    EOTask, 
    EOPatch,
    EOWorkflow,
    FeatureType,
    MapFeatureTask,
    RemoveFeatureTask,
    linearly_connect_tasks,
    EOExecutor,
)
from eolearn.features import SimpleFilterTask
from eolearn.io import SentinelHubInputTask
from eolearn.features.utils import spatially_resize_image as resize_images
from sentinelhub import (
    BBox,
    DataCollection,
    SHConfig,
    get_utm_crs,
    wgs84_to_utm,
)
from sentinelhub.exceptions import SHDeprecationWarning
from tqdm.auto import tqdm

from phisat2_constants import (
    S2_BANDS,
    S2_RESOLUTION,
    BBOX_SIZE,
    PHISAT2_RESOLUTION,
    ProcessingLevels,
)
from phisat2_utils import (
    AddPANBandTask,
    AddMetadataTask,
    CalculateRadianceTask,
    CalculateReflectanceTask,
    SCLCloudTask,
    BandMisalignmentTask,
    PhisatCalculationTask,
    AlternativePhisatCalculationTask,
    CropTask,
    GriddingTask,
    ExportGridToTiff,
    get_extent,
)

# filter out some SHDeprecationWarnings
import warnings

warnings.filterwarnings("ignore", category=SHDeprecationWarning)

ProcessingLevels._member_names_
PROCESSING_LEVEL = ProcessingLevels.L1C
PROCESSING_LEVEL


def get_utm_bbox(lat_centre: float, lon_centre: float):
    """Returns a bounding box of size corresponding to the swath width of Φ-sat-2 given the centroid of the area-of-interest in WGS84"""

    east, north = wgs84_to_utm(lon_centre, lat_centre)

    east, north = 10 * int(east / 10), 10 * int(north / 10)
    crs = get_utm_crs(lon_centre, lat_centre)

    return BBox(
        bbox=(
            (east - BBOX_SIZE // 2, north - BBOX_SIZE // 2),
            (east + BBOX_SIZE // 2, north + BBOX_SIZE // 2),
        ),
        crs=crs,
    )

#lat_centre, lon_centre = 42.348, 13.397  # l'Aquila
#bbox = get_utm_bbox(lat_centre, lon_centre)


import os
import numpy as np
import rasterio
from eolearn.core import EOPatch, FeatureType, EOWorkflow, linearly_connect_tasks, EOExecutor
from sentinelhub import BBox, CRS

# Import specific tasks from your uploaded notebook's utilities
from phisat2_utils import (
    AddMetadataTask, CalculateRadianceTask, CalculateReflectanceTask, 
    SCLCloudTask, BandMisalignmentTask, PhisatCalculationTask, 
    GriddingTask, ExportGridToTiff
)
from phisat2_constants import PHISAT2_RESOLUTION

def load_npy_to_eopatch(image_path, mask_path):
    """
    Loads .npy files and formats them for the simulation workflow.
    Sentinel-2 EOPatch structure: (Time, Height, Width, Channels)
    """
    # Load Image (Sentinel-2 L1C Bands)
    # The simulation expects 7 bands: B02, B03, B04, B08, B05, B06, B07
    image = np.load(image_path) 
    if image.shape[0] == 7: # If (C, H, W), transpose to (H, W, C)
        image = image.transpose(1, 2, 0)
    
    # Load Mask (SCL)
    mask = np.load(mask_path)
    if mask.ndim == 3 and mask.shape[0] == 4: # If multi-channel mask, pick SCL channel
        mask = mask[0, :, :] # Adjust index based on dataset docs
        
    eop = EOPatch()
    
    # 1. Add Bands (Step 1)
    # Add dummy time dimension: (1, H, W, 7)
    eop[FeatureType.DATA, 'BANDS'] = image[np.newaxis, ...].astype(np.float32)
    
    # 2. Add SCL Mask
    eop[FeatureType.MASK, 'SCL'] = mask[np.newaxis, ..., np.newaxis].astype(np.uint8)
    
    # 3. Simulation tasks require 'sunZenithAngles' 
    # If not in dataset, we use a standard default (approx 30 degrees)
    eop[FeatureType.DATA, 'sunZenithAngles'] = np.full_like(image[..., :1], 30.0)[np.newaxis, ...]
    
    # 4. Define a BBox (Required for spatial tasks even if arbitrary)
    eop.bbox = BBox(((0, 0), (10000, 10000)), crs=CRS.WGS84)
    eop.timestamp = [datetime.datetime(2000, 1, 1)]
    return eop

# --- Simulation Execution ---

# Path to your local files
root_dir = '/local/s3167445/data'
img_file = os.path.join(root_dir, 'TrainVal/numpy_images/0000479.npy')
msk_file = os.path.join(root_dir, 'TrainVal/numpy_masks/0000479.npy')

# Initialize EOPatch
eopatch = load_npy_to_eopatch(img_file, msk_file)

# Build the workflow (Steps 2 through 9)
workflow_nodes = linearly_connect_tasks(
    AddMetadataTask(),                 # Required for radiance
    #CalculateRadianceTask(FeatureType.DATA, "BANDS"), (FeatureType.DATA, "BANDS-RAD"),           # Step 2
    # PhisatCalculationTask(             # Step 3, 5, 6
    #     resampling_res=PHISAT2_RESOLUTION, 
    #     snr_degradation=True, 
    #     mtf_degradation=True
    # ),
    # BandMisalignmentTask(),            # Step 4
    # CalculateReflectanceTask(),        # Step 7
    # GriddingTask(grid_shape=(4, 4)),   # Step 8 (16 patches)
    # ExportGridToTiff(folder='output_phisat2') # Step 9
)

workflow = EOWorkflow(workflow_nodes)

# Run the simulation on the loaded patch
result = workflow.execute({workflow_nodes[0]: {'eopatch': eopatch}})

print("Phisat-2 simulation finished. Patches saved as Tiff in 'output_phisat2'.")