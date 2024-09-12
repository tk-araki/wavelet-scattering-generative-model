import argparse

from utils import load_config, fix_torch_seed
from image_scattering_encode import encode_images
from scattering_generative_model import ScatteringGenerativeModel

##
def main(args):

    config = load_config(args.config_file)

    # encode
    if args.encode_enabled:
        # 学習データ
        encode_images(config['image']['size'],config['image']['train_dir'],
               config['scat']['J'],config['scat']['batch_size'],
               config['scat']['train_dir'],
               config['pca']['algorithm'],
               config['pca']['train_pcs_dir'],
               obj_save_file=config['pca']['obj_file'],
               pcs_dim=config['pca']['pcs_dim'],
               ipca_batch_size = config['pca']['batch_size'])

        # Validationデータ
        encode_images(config['image']['size'],config['image']['valid_dir'],
               config['scat']['J'],config['scat']['batch_size'],
               config['scat']['valid_dir'],
               config['pca']['algorithm'],
               config['pca']['valid_pcs_dir'],
                   obj_load_file=config['pca']['obj_file'],
                    ipca_batch_size = config['pca']['batch_size'])

    ## Train Generator

    fix_torch_seed(0)

    sgmodel = ScatteringGenerativeModel(model_def_params={'scat_dim': config['pca']['pcs_dim'],
                                                          'image_size': config['image']['size'],
                                                          'num_image_channels': config['image']['num_channels'],
                                                          'num_final_channels': config['generator']['num_final_channels'],
                                                          'input_size': config['generator']['input_size']})

    # train the model and save
    sgmodel.train(config['generator']['num_epochs'],
                  config['generator']['batch_size'],
                  config['pca']['train_pcs_dir'],config['image']['train_dir'],
                  config['pca']['valid_pcs_dir'],config['image']['valid_dir'], #path_set['colab_valid_image_dir'],
                  config['generator']['checkpoint_dir'],
                  config['generator']['log_dir'],
                  config['generator']['log_interval'],
                   lr_adam=config['generator']['lr'],num_workers=config['generator']['num_workers']) # pin_memory=True


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='train the scattering generative model')
    parser.add_argument('config_file',
                        metavar='FILE',
                        help="Path of the configuration file")

    parser.add_argument('--disable_encode',dest="encode_enabled", action='store_false')

    args = parser.parse_args()
    main(args)