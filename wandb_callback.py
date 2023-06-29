import tensorflow as tf
import wandb

class WandbCallback(tf.keras.callbacks.Callback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.wandb = wandb.init()
        self.wandb = wandb

    def on_epoch_end(self, epoch, logs=None):
        wandb.log(logs)

    def on_train_batch_end(self, batch, logs=None):
        wandb.log(logs)

