from model import *
import config
from dataload import MnistDataModule
from callbacks import MyCallbackPrint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

torch.set_float32_matmul_precision("high")

def main():

    logger = TensorBoardLogger("tb_logs", name="MNIST_Model")

    # Set device cuda for GPU if it's available otherwise run on the CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NN(input_size=config.input_size, num_classes=config.num_classes)
    dm = MnistDataModule(data_dir=config.data_dir, batch_size=config.batch_size, num_workers=config.num_workers)
    trainer = pl.Trainer(
        profiler="simple",
        logger=logger,
        accelerator=config.Accelerator, 
        devices=config.DEVICES, 
        min_epochs=config.MIN_EPOCH, 
        max_epochs=config.MAX_EPOCH)
    callbacks=[MyCallbackPrint(), EarlyStopping(monitor="val_accuracy")], #monitor val loss
    trainer.fit(model, dm)
    trainer.validate(model, dm)
    trainer.test(model, dm)


if __name__ =="__main__":
    main()