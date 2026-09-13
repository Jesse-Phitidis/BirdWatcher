from pytorch_lightning.cli import LightningCLI
import pytorch_lightning as pl  

if __name__ == "__main__":
    cli = LightningCLI(
        pl.LightningModule,
        pl.LightningDataModule,
        subclass_mode_model=True,
        subclass_mode_data=True,
        parser_kwargs={"parser_mode": "omegaconf"},
    )