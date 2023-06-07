
import model
import config
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, Callback

class MyCallbackPrint(Callback):
    def __init__(self) -> None:
        super().__init__()


    def on_train_start(self, trainer, pl_module):
        print("Starting to train!")

    def on_train_end(self, trainer, pl_module):
        print("Training is done.")