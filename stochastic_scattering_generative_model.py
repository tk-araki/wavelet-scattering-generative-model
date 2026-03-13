import torch

from scattering_generative_model import ScatteringGenerativeModel

class StochasticScatteringGenerativeModel(ScatteringGenerativeModel):
    def __init__(self,
                 model_def_params=None,
                 model_file=None,
                 comp_std_file = None):

        super().__init__(model_def_params,model_file)

        if model_def_params is not None:
            if comp_std_file is None:
                raise ValueError("When 'model_def_params' is specified, 'comp_std_file' must also be specified.")

            self.comp_stds = torch.load(comp_std_file)

        if model_file is not None:
            checkpoint = torch.load(model_file)
            self.comp_stds = checkpoint["comp_stds"]

    def _update_image_generator(self, dataloader_train, loss):
        if not self.image_generator.training:
            self.image_generator.train()

        for image_batch, scat_batch in dataloader_train:
            image_batch = image_batch.to(self.cur_device)
            scat_batch = scat_batch.to(self.cur_device)
            rand_batch = torch.normal(mean=0, std=1, size=scat_batch.shape) * self.comp_stds
            rand_batch = rand_batch.to(self.cur_device)

            scat_batch = scat_batch + rand_batch

            self.optimizer.zero_grad()
            recons_batch = self.image_generator(scat_batch)
            loss_recons = loss(image_batch, recons_batch)
            loss_recons.backward()

            self.optimizer.step()


    def save_checkpoint(self,file_path,epoch,batch_size,train_loss,valid_loss,save_optimizer = True): # .pth
        state = {"epoch": epoch,
                 "model_def_params":dict(self.model_def_params),
                 "batch_size":batch_size,
                 "comp_stds":self.comp_stds,
                 "train_loss":train_loss,
                 "valid_loss":valid_loss,
                 "model_state_dict": self.image_generator.state_dict()}# batch数追加
        if save_optimizer:
            state['optimizer_state_dict'] = self.optimizer.state_dict()

        torch.save(state,file_path)