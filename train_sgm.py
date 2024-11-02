import argparse

from utils import load_config, fix_torch_seed
from loggers import MlflowSGMTrainLogger, TensorboardTrainingLogger
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
    if args.fix_seed:
        fix_torch_seed(args.seed)

    train_logger = MlflowSGMTrainLogger(config['logger']['log_dir'],
                                        config['logger']['experiment_name'],
                                        config['logger']['common_run_name'] + '_train') if args.logger == 'mlflow' else \
        TensorboardTrainingLogger(config['logger']['log_dir'])

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
                  train_logger,

                  config['logger']['train_log_interval'],config['generator']['shuffle'],
                   lr_adam=config['generator']['lr'],num_workers=config['generator']['num_workers']) # pin_memory=True


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='train the scattering generative model')
    parser.add_argument('config_file',
                        metavar='FILE',
                        help="Path to the configuration file")

    parser.add_argument('--logger',choices=['mlflow','tensorboard'], default='mlflow')
    parser.add_argument('--disable_encode',dest="encode_enabled", action='store_false')
    parser.add_argument('--fix_seed', action='store_true')
    parser.add_argument('--seed',type=int, default=314)

    args = parser.parse_args()
    main(args)