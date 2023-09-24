from keras.callbacks import *
import keras
import os

class RedirectModel(keras.callbacks.Callback):
    """
    Callback which wraps another callback, but executed on a different model.

    ```python
    model = keras.models.load_model('model.h5')
    model_checkpoint = ModelCheckpoint(filepath='snapshot.h5')
    parallel_model = multi_gpu_model(model, gpus=2)
    parallel_model.fit(X_train, Y_train, callbacks=[RedirectModel(model_checkpoint, model)])
    ```

    Args
        callback : callback to wrap.
        model : model to use when executing callbacks.
    """

    def __init__(self,
                 callback,
                 model):
        super(RedirectModel, self).__init__()

        self.callback = callback
        self.redirect_model = model

    def on_epoch_begin(self, epoch, logs=None):
        self.callback.on_epoch_begin(epoch, logs=logs)

    def on_epoch_end(self, epoch, logs=None):
        self.callback.on_epoch_end(epoch, logs=logs)

    def on_batch_begin(self, batch, logs=None):
        self.callback.on_batch_begin(batch, logs=logs)

    def on_batch_end(self, batch, logs=None):
        self.callback.on_batch_end(batch, logs=logs)

    def on_train_begin(self, logs=None):
        # overwrite the model with our custom model
        self.callback.set_model(self.redirect_model)

        self.callback.on_train_begin(logs=logs)

    def on_train_end(self, logs=None):
        self.callback.on_train_end(logs=logs)
        

def scheduler(epoch, lr):
    if epoch < 40:
        return 1e-5
    else:
        return 1e-6


def create_callbacks(training_model, prediction_model, validation_generator, snapshot_path, backbone,fpn,n_stage,
                     evaluation=True, dataset_type='voc', snapshots=True):
    """
    Creates the callbacks to use during training.

    Args
        training_model: The model that is used for training.
        prediction_model: The model that should be used for validation.
        validation_generator: The generator for creating validation data.
        args: parseargs args object.

    Returns:
        A list of callbacks used for training.
    """
    callbacks = []


    if evaluation and validation_generator:
        if dataset_type == 'coco':
            from eval.coco import CocoEval
            # use prediction model for evaluation
            evaluation = CocoEval(validation_generator, prediction_model)
        else:
            from eval.pascal import Evaluate
            evaluation = Evaluate(validation_generator, prediction_model)
        callbacks.append(evaluation)

    # save the model
    if snapshots:
        # ensure directory created first; otherwise h5py will error after epoch.
        if not os.path.exists(snapshot_path):
            os.makedirs(snapshot_path)
        checkpoint = ModelCheckpoint(
            os.path.join(
                snapshot_path,
                '{dataset_type}_{backbone}_{fpn}_{n_stage}_{{epoch:02d}}.h5'.format(dataset_type=dataset_type,
                                                                                             fpn=fpn,
                                                                                             n_stage=n_stage,
                                                                                             backbone=backbone)
            ),
            verbose=1,
            save_best_only=True,
            monitor="mAP",
            mode='max'
        )
        checkpoint = RedirectModel(checkpoint, training_model)
        callbacks.append(checkpoint)
      
    # To do !
    early_stopping = EarlyStopping(
                               monitor='mAP',                                                     
                               min_delta=0.0,
                               mode = 'max',
                               patience=5,
                               verbose=1,
                               restore_best_weights = True)
    
    callbacks.append(early_stopping)
    return callbacks
