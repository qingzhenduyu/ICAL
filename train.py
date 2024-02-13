from pytorch_lightning.plugins.training_type.ddp import DDPPlugin
from pytorch_lightning.utilities.cli import LightningCLI

from ical.datamodule import HMEDatamodule
from ical.lit_ical import LitICAL

cli = LightningCLI(
    LitICAL,
    HMEDatamodule,
    save_config_overwrite=True,
    trainer_defaults={"plugins": DDPPlugin(find_unused_parameters=False)},
)
