from .baseline_cnn import *
import pytorch_lightning as pl
from dataset_box.data_loader import DataLoader, SegmentationDataModule
from .losses import *
from .lightning_module import SegmentationTask
    

resnet = ResNet18UNet(in_channels=7, num_classes=4)
task = SegmentationTask(resnet, use_focal=True)

# Setup DataModule
dm = SegmentationDataModule(root_dir='/tmp/ivanderspoel/burn_dataset', batch_size=16, num_workers=8)
dm.setup(stage='test')  # load test dataset

trainer = pl.Trainer(
    accelerator="gpu", 
    devices=[0], # Use GPU 0 as identified in your nvidia-smi
    max_epochs=10,
    precision=32#"16-mixed"

)

trainer.fit(task, datamodule=dm)
torch.save(resnet.state_dict(), "model_box/saved_models/resnet.pt")