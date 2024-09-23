from abc import ABCMeta, abstractmethod  # abstractclassmethod
import warnings

from torch.utils.tensorboard import SummaryWriter
import mlflow

class TrainingLogger(metaclass=ABCMeta):

    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def log_losses(self, train_loss, valid_loss, epoch):
        pass

    @abstractmethod
    def close(self):
        pass


class MlflowTrainingLogger(TrainingLogger):
    def __init__(self, log_dir, experiment_name, run_name):
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        self.run_name = run_name

    def setup(self):
        self.log_dir.mkdir(parents=True,exist_ok =True)
        mlflow.set_tracking_uri(self.log_dir)
        mlflow.set_experiment(self.experiment_name)
        mlflow.start_run(run_name=self.run_name)

    def log_losses(self, train_loss, valid_loss, epoch):
        mlflow.log_metric("train loss", train_loss, step=epoch)
        mlflow.log_metric("valid loss", valid_loss, step=epoch)

    def close(self):
        mlflow.end_run()


class TensorboardTrainingLogger(TrainingLogger):
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.writer_training = None

    def setup(self):
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer_training = SummaryWriter(self.log_dir)

    def log_losses(self, train_loss, valid_loss, epoch):
        self.writer_training.add_scalars('Losses', {'train_loss': train_loss, 'valid_loss': valid_loss}, epoch)

    def close(self):
        if self.writer_training is None:
            warnings.warn(
                "`writer_training` has not been initialized. Make sure `setup()` was called before closing.",
                stacklevel=2)
            return

        self.writer_training.close()
