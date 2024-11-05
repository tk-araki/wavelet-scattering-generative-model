from importlib import import_module
import sys

from pathlib import Path
import torch
import torchvision.utils as utils
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from datasets import ImageScatDataset
import mlflow

from utils import load_config

models = {
    "ssgm": {
        "generative_model": {
            "module": "stochastic_scattering_generative_model",
            "class": "StochasticScatteringGenerativeModel"
        },
        "encode": {
            "module": "image_scattering_encode",
            "func": "encode_images_with_varscales"
        }
    },
    "sgm": {
        "generative_model": {
            "module": "scattering_generative_model",
            "class": "ScatteringGenerativeModel"
        },
        "encode": {
            "module": "image_scattering_encode",
            "func": "encode_images"
        }
    }
}

### Reconstruct images

def to_grid_image(image_batch, num_cols, save_file):
    images_grid = utils.make_grid(image_batch, nrow=num_cols)
    images_grid = (images_grid + 1) / 2
    pil_images_grid = transforms.functional.to_pil_image(images_grid)
    pil_images_grid.save(save_file)


def reconstruct_image(image_dir, scatpcsdir, sgmodel, nrows_grid=2, ncols_grid=8,  # reconstruct_image
                      save_original=False, save_recns_dir=Path('./'),pin_memory=True, num_workers=2):

    save_recns_dir.mkdir(parents=True, exist_ok=True)

    image_scat_dataset = ImageScatDataset(image_dir, scatpcsdir, scat_suffix='_scat_pcs.npy')  # モジュールファイルのグローバル変数にする？
    image_scat_dataloader = DataLoader(image_scat_dataset, nrows_grid * ncols_grid, pin_memory=pin_memory, num_workers=num_workers)
    for batch_num, (image_batch, scat_batch) in enumerate(image_scat_dataloader):
        # recons_images_file = save_recns_dir / f'reconstructed_images_batch_{batch_num}.png'
        sgmodel.generate_images(scat_batch, num_cols=ncols_grid,
                                save_file=save_recns_dir / f'reconstructed_images_batch_{batch_num}.png')
        # mlflow.log_artifact(recons_images_file)

        if save_original:
            # original_images_file = save_recns_dir / f"original_images_batch_{batch_num}.png"
            to_grid_image(image_batch, num_cols=ncols_grid,
                          save_file=save_recns_dir / f"original_images_batch_{batch_num}.png")
            # mlflow.log_artifact(original_images_file)

def loss_test_recons(encode,model,config,pin_memory=True):

    encode(config['image']['size'], config['image']['test_dir'],
           config['scat']['J'], config['scat']['batch_size'],
           config['scat']['test_dir'],
           config['pca']['algorithm'],
           config['pca']['test_pcs_dir'],
           obj_load_file=config['pca']['obj_file'],
           ipca_batch_size=config['pca']['batch_size'])

    # test用dataloader作成
    test_imgscat_dataset = ImageScatDataset(config['image']['test_dir'], config['pca']['test_pcs_dir'],
                                             scat_suffix='_scat_pcs.npy')  # scat_suffix モジュールファイルのグローバル変数にする？
    test_imgscat_dataloader = DataLoader(test_imgscat_dataset, config['generator']['batch_size'], pin_memory=pin_memory,
                                          num_workers=config['generator']['num_workers'])

    l1_loss = torch.nn.L1Loss()  # l1_loss

    test_mean_l1_loss = model.recons_loss(test_imgscat_dataloader, l1_loss, len(test_imgscat_dataset))

    return test_mean_l1_loss

def main(args):
    specified_model = args[1]
    config_file = Path(args[2])
    model_epoch = args[3]

    model_module = import_module(models[specified_model]["generative_model"]["module"])
    GenerativeModel = getattr(model_module, models[specified_model]["generative_model"]["class"])

    encode_module = import_module(models[specified_model]["encode"]["module"])
    encode = getattr(encode_module, models[specified_model]["encode"]['func'])

    config = load_config(config_file)

    mlflow.set_tracking_uri(config['logger']['log_dir'])
    mlflow.set_experiment(config['logger']['experiment_name'])

    with mlflow.start_run(run_name=config['logger']['common_run_name']+'_infer'): # f'{model_epoch}epoch__infer
        mlflow.log_param("model_epoch", model_epoch)
        mlflow.log_param("model", specified_model)

        if specified_model == "ssgm":
            mlflow.log_param("comp_std_1st", config['pca'].get('comp_std_1st'))

        # load model
        trained_model_file = config['generator']['checkpoint_dir'] / f'model_at_{model_epoch}epoch.pth'
        trained_sgmodel = GenerativeModel(model_file=trained_model_file)

        # generate images from Gaussian noises
        config['infer']['generated_image_dir'].mkdir(parents=True, exist_ok=True)

        torch.manual_seed(seed=0)
        n_images = 128
        generated_image_file = config['infer']['generated_image_dir'] / \
                               f"{model_epoch}epoch_{config_file.stem.replace('config_', '')}_n_images{n_images}.png" # [len("config_"):]
        rand_scat_batch = torch.randn(n_images,trained_sgmodel.model_def_params['scat_dim'])
        trained_sgmodel.generate_images(rand_scat_batch, num_cols=16,
                                        save_file=generated_image_file)

        print(f"Generated image was saved in\n{generated_image_file.parent.resolve()}")

        # mlflow log
        mlflow.log_artifact(generated_image_file)

        # reconstruct images
        input_image_dir = config['infer']['recons_input_image_dir']
        recont_scat_dir = config['infer']['recons_output_dir'] / "scats"
        recons_scatpcs_dir = config['infer']['recons_output_dir'] / "scatpcs"
        recons_image_dir = config['infer']['recons_output_dir'] / f"reconstructed_images_{model_epoch}epoch"

        encode(config['image']['size'], input_image_dir,
                config['scat']['J'],config['scat']['batch_size'],
                recont_scat_dir,
                config['pca']['algorithm'],
                recons_scatpcs_dir,
                    obj_load_file=config['pca']['obj_file'],
                    ipca_batch_size = config['pca']['batch_size'])


        reconstruct_image(input_image_dir,recons_scatpcs_dir,trained_sgmodel,#nrows_grid =2,ncols_grid = 8,
                         save_original = True,save_recns_dir=recons_image_dir)

        print(f"Reconstructed images were saved in\n{recons_image_dir.resolve()}")

        # mlflow log
        mlflow.log_artifact(recons_image_dir)

        # test loss
        test_mean_l1_loss = loss_test_recons(encode,trained_sgmodel,config)
        mlflow.log_metric("test loss", test_mean_l1_loss)


if __name__ == "__main__":
    main(sys.argv)