import time
import json
import warnings
from types import MappingProxyType

from tqdm import trange
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter #  tensorboard --logdir=runs --bind_all

from datasets import ImageScatDataset
from generator import ImageGenerator

class ScatteringGenerativeModel:# ScatteringGenerativeModel

    def __init__(self,
                 model_def_params=None,
                 model_file=None): # configs  # load_optimizer = None, # cpu_only = False
        '''
        : dictionary
            key:scat_dim image_size num_final_channels input_size(arbitrary)
        '''

        if model_def_params is None and model_file is None:
            raise ValueError("No argument. Specify one argument, 'model_def_params' or 'model_file'.")
        if model_def_params is not None and model_file is not None:
            raise ValueError("Two arguments. Specify only one argument, 'model_def_params' or 'model_file'.")
        '''
                self.study_name = study_name
                if direction is None and directions is None:
                    raise ("Specify one of `direction` and `directions`.")
                elif directions is not None:
                    self._directions = list(directions)
                elif direction is not None:
                    self._directions = [direction]
                else:
                    raise ValueError("Specify only one of `direction` and `directions`.")
        '''
        self.cur_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.optimizer = None
        if model_def_params is not None:
            self.image_generator = ImageGenerator(**model_def_params)
            self.model_def_params = MappingProxyType(model_def_params.copy())
            # self.test = model_def_params
            self.epoch = 0

        if model_file is not None:
            # with open(model_file.parent / 'model_def_params.json') as f:
            #     model_def_params = json.load(f)
            checkpoint = torch.load(model_file)# torch.load(model_file,map_location=torch.device('cpu'))
            self.epoch = checkpoint['epoch']
            self.model_def_params = MappingProxyType(checkpoint['model_def_params'])
            self.image_generator = ImageGenerator(**self.model_def_params)

            self.image_generator.load_state_dict(checkpoint['model_state_dict'])
            # optimizerあったらロード
            if 'optimizer_state_dict' in checkpoint.keys():
                self.optimizer = optim.Adam(self.image_generator.parameters())
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(self.cur_device)

        self.image_generator.to(self.cur_device)

    # def load_optimizer(self,optim_path = None):
    # def create_optimizer(self,optim_path = None):
    #     if optim_path is None:
    #         optim.Adam(self.image_generator.parameters())
    # def configure_optimizers(self):

    def __update_image_generator(self,dataloader_train,loss):# __update_generator_params
        if not self.image_generator.training:
            self.image_generator.train()

        # train_loss
        for image_batch, scat_batch in dataloader_train:
            image_batch = image_batch.to(self.cur_device)
            scat_batch = scat_batch.to(self.cur_device)

            self.optimizer.zero_grad()
            recons_batch = self.image_generator(scat_batch)
            loss_recons = loss(image_batch, recons_batch)
            loss_recons.backward()# list(self.image_generator.parameters())[0].grad; list(self.image_generator.parameters())[0][0,0]

            self.optimizer.step()

        # return train_loss

    def loss_recons_for_eval(self,dataloader_eval,loss,num_data):
        if self.image_generator.training:
            self.image_generator.eval()
        mean_loss = 0.0
        with torch.inference_mode():
            for image_batch, scat_batch in dataloader_eval:
                image_batch = image_batch.to(self.cur_device)
                scat_batch = scat_batch.to(self.cur_device)

                recons_batch = self.image_generator(scat_batch)
                loss_recons = loss(image_batch, recons_batch)
                mean_loss += loss_recons.item() * scat_batch.shape[0]

        return mean_loss / num_data

    # def get_model_def_params(self):
    #     return {
    #         'scat_dim': self.image_generator.generator_input.in_features,
    #         'image_size': self.image_generator.input_tensor_size[1] * 2 ** self.image_generator.level_uppsampling,
    #         'num_final_channels': self.image_generator.final_layer[0].in_channels,
    #         'input_size': self.image_generator.input_tensor_size[1],
    #     }

    def save_checkpoint(self,file_path,epoch,batch_size,train_loss,valid_loss,save_optimizer = True): # .pth
        state = {"epoch": epoch,
                 "model_def_params":dict(self.model_def_params),
                 "batch_size":batch_size,
                 "train_loss":train_loss,
                 "valid_loss":valid_loss,
                 "model_state_dict": self.image_generator.state_dict()}# batch数追加
        if save_optimizer:
            state['optimizer_state_dict'] = self.optimizer.state_dict()

        torch.save(state,file_path)

    def train(self, num_epochs, batch_size, train_scat_dir, train_image_dir,
              valid_scat_dir,valid_image_dir,
              save_dir=Path('model_checkpoints'),  # model_save_dir = ^^^
              log_dir=Path('logs_training'),
              epochs_model_save=None, epochs_valid_log = 1,
              pin_memory=True, num_workers=2,lr_adam = None):
        '''
        run_all.py
        batch_size=train_batch_size; train_scat_dir = path_set['train_scatpcs_dir']; save_dir=path_set['trained_model_dir']
        '''

        if not save_dir.exists():
            save_dir.mkdir(parents=True)

        if not log_dir.exists():
            log_dir.mkdir(parents=True)

        if epochs_model_save is None:
            epochs_model_save = self.epoch + num_epochs

        if self.optimizer is not None and lr_adam is not None:
            print("'lr_adam' is not used since self.optimizer already exists")# Argument warnings.warn(

        # 学習用dataloader作成
        train_imgscat_dataset = ImageScatDataset(train_image_dir,train_scat_dir,scat_suffix='_scat_pcs.npy') # scat_suffix モジュールファイルのグローバル変数にする？
        train_imgscat_dataloader = DataLoader(train_imgscat_dataset, batch_size, pin_memory=pin_memory, num_workers=num_workers)

        # validation用dataloader作成
        valid_imgscat_dataset = ImageScatDataset(valid_image_dir,valid_scat_dir,scat_suffix='_scat_pcs.npy') # scat_suffix モジュールファイルのグローバル変数にする？
        valid_imgscat_dataloader = DataLoader(valid_imgscat_dataset, batch_size, pin_memory=pin_memory, num_workers=num_workers)

        l1_loss = torch.nn.L1Loss() # l1_loss

        if self.optimizer is None: # , optim_params = {} ↓ ( , optim_params)
            if lr_adam is None: # lr_adam = 1e-3 if lr_adam is None else lr_adam
                lr_adam = 1e-3
            self.optimizer = optim.Adam(self.image_generator.parameters(),lr=lr_adam)

        writer_training = SummaryWriter(log_dir) # フォルダがないときは作成される

        for epoch in trange(self.epoch,self.epoch + num_epochs+1): # 後で変更予定
            # torch.cuda.synchronize()

            # train
            if epoch != self.epoch:
                start_time_train = time.perf_counter()
                self.__update_image_generator(train_imgscat_dataloader,l1_loss)
                end_time_train = time.perf_counter()
                print(f"train_1_epoch {end_time_train-start_time_train:.3f}sec")

            if not epoch % epochs_valid_log:
                # training error
                start_time_eval = time.perf_counter()
                train_loss=self.loss_recons_for_eval(train_imgscat_dataloader, l1_loss, len(train_imgscat_dataset))
                end_time_eval = time.perf_counter()
                print(f"training error calc {end_time_eval - start_time_eval:.3f}sec")
                # validation error　
                start_time_eval = time.perf_counter()
                valid_loss=self.loss_recons_for_eval(valid_imgscat_dataloader, l1_loss, len(valid_imgscat_dataset))
                end_time_eval = time.perf_counter()
                print(f"validation error calc {end_time_eval - start_time_eval:.3f}sec")

                # end_time = time.perf_counter()
                # elapsed_time = end_time - start_time #　＜ー　いらなそう デバッグ用 Tensorboardで見れる,CSVに書き出せる
                # print(f"time.time = {time.time()} just before writer_training.add_scalars")  # 日付に変換^^^
                # log　追記
                writer_training.add_scalars('Loss', {'train_loss': train_loss, 'valid_loss': valid_loss},epoch)

            # モデル等を保存
            if epoch != self.epoch and not epoch % epochs_model_save:
                self.save_checkpoint(save_dir / f'model_at_{epoch}epoch.pth', epoch,batch_size,
                                     train_loss,valid_loss, save_optimizer=True) # historyなし^^^
                # ベストValidation Score modelを保存    #-- epochs_model_save == 1, if min_valid_loss > valid_loss, self.save_checkpoin t()

        writer_training.close()
        self.epoch += num_epochs


    # レビュー対象　return 部分   名前generate_image_grid  grid of images
    def generate_images(self,scat_batch,num_cols=8,save_file=None # save_fileの方が良さそう ^^^^^
                        ):  # train_scat_dir=None,scat_batch=None ;; scat_coefs
        if self.image_generator.training:
            self.image_generator.eval()

        with torch.inference_mode():
            scat_batch = scat_batch.to(self.cur_device)
            generated_batch = self.image_generator(scat_batch)

        images_grid = torchvision.utils.make_grid(generated_batch,nrow=num_cols)
        images_grid = (images_grid + 1)/2
        pil_images_grid = transforms.functional.to_pil_image(images_grid) # 入力tensorのスケールは[0,1]でないといけん

        if save_file:
            pil_images_grid.save(save_file)# / f'generated_images_grid_{num_cols}rows.png')
        else:
            return pil_images_grid
