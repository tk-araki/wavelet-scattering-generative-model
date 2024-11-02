from pathlib import Path
import joblib
import tempfile

from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from kymatio.torch import Scattering2D
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.utils import gen_batches

from datasets import ImageDataset

# os.environ["KYMATIO_BACKEND_2D"] = "skcuda"

def scattering_imageset(J,image_size, batch_size, image_dir, save_dir): # ToDo:scattering_imagesに名称変更

    if not save_dir.exists():
        save_dir.mkdir(parents=True)

    scattering = Scattering2D(J, (image_size, image_size))

    cur_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    scattering.to(cur_device)

    img_dataset = ImageDataset(image_dir)
    img_dataloader = DataLoader(img_dataset, batch_size, pin_memory=True, num_workers=2)

    for img_batch, base_batch in tqdm(img_dataloader):

        img_batch = img_batch.to(cur_device)
        img_scat_batch = scattering(img_batch).cpu().numpy()

        # save
        for i,base in enumerate(base_batch):
            save_file = save_dir / f'{base}_scat.npy'
            np.save(save_file, img_scat_batch[i])

def load_scat_set(scat_dir):
    '''
    load scattering coefficients
    '''

    scat_files = list(scat_dir.glob('*_scat.npy'))
    scats = [np.load(f).ravel() for f in scat_files]
    return np.stack(scats),scat_files

def pca_scattering(scat_dir,pcs_save_dir,pca_save_file=None,pca_load_file=None,num_components=None):

    if pca_save_file is None and pca_load_file is None:
        raise ValueError("No argument. Specify one argument, 'pca_save_file' or 'pca_load_file'.")
    if pca_save_file is not None and pca_load_file is not None:
        raise ValueError("Two arguments. Specify only one argument, 'pca_save_file' or 'pca_load_file'.")
    if pca_save_file is not None and num_components is None:
        raise ValueError("When 'pca_save_file' is set, 'num_components' must be specified.")

    if not pcs_save_dir.exists():
        pcs_save_dir.mkdir(parents=True)

    scat,scat_files = load_scat_set(scat_dir)

    if pca_save_file is not None:
        pca = PCA(n_components=num_components,whiten=True,svd_solver='arpack')
        pca.fit(scat)
        joblib.dump(pca, pca_save_file, compress=True)

    elif pca_load_file is not None:
        pca = joblib.load(pca_load_file)

    scat_pcs = pca.transform(scat)

    # save principal component scores
    for i,f in enumerate(scat_files):
        save_file = pcs_save_dir / f'{f.stem}_pcs.npy'
        np.save(save_file, scat_pcs[i])

#--- for large data
def load_scats_to_memmap(scat_dir,J,image_len):
    '''
    load scattering coefficients into memoery-map
    '''

    scat_files = list(scat_dir.glob('*_scat.npy'))

    Q=8
    scat_dim = int(image_len*(1+Q*J+Q*Q*J*(J-1)/2)/(2**(2*J)))
    fp = tempfile.NamedTemporaryFile()
    scats_mmap = np.memmap(fp, dtype='float32', mode='w+', shape=(len(scat_files), scat_dim))

    for n, f in enumerate(scat_files):
        scats_mmap[n, :] = np.load(f).ravel() # scats = [np.load(f).ravel() for f in scat_files]

    return scats_mmap, scat_files

def ipca_large_scattering(scat_dir,J,image_len,ipcs_save_dir,ipca_save_file=None,ipca_load_file=None,
                          num_components=None,batch_size=None):

    if ipca_save_file is None and ipca_load_file is None:
        raise ValueError("One of the arguments 'ipca_save_file' or 'ipca_load_file' must be specified.")
    if ipca_save_file is not None and ipca_load_file is not None:
        raise ValueError("Only one of the arguments 'ipca_save_file' or 'ipca_load_file' can be specified.")
    ipca_params_specified = num_components is not None and batch_size is not None
    if ipca_save_file is not None and not ipca_params_specified:
        raise ValueError("When 'ipca_save_file' is specified, 'num_components' and 'batch_size' must also be specified.")

    ipcs_save_dir.mkdir(parents=True,exist_ok=True)

    scats_memmap,scat_files = load_scats_to_memmap(scat_dir,J,image_len)

    if ipca_save_file is not None:
        ipca = IncrementalPCA(n_components=num_components,whiten=True,batch_size=batch_size)
        ipca.fit(scats_memmap)
        joblib.dump(ipca, ipca_save_file, compress=True)

    elif ipca_load_file is not None:
        ipca = joblib.load(ipca_load_file)
        if batch_size is None:
            batch_size = len(scat_files)

    for batch in gen_batches(len(scat_files),batch_size):
        scat_pcs = ipca.transform(scats_memmap[batch]).astype(np.float32)

        # 主成分スコア　保存
        for i,f in enumerate(scat_files[batch]):
            save_file = ipcs_save_dir / f'{f.stem}_pcs.npy'
            np.save(save_file, scat_pcs[i])


def encode_images(image_size,image_dir,J,scat_batch_size,scat_dir, pca_algorithm, scatpcs_dir, #
           obj_save_file=None,obj_load_file=None,pcs_dim=None,ipca_batch_size = None):

    print("Image Scattering") #"scattering images"
    scattering_imageset(J, image_size, scat_batch_size, image_dir, scat_dir)# scattering_images

    if pca_algorithm == 'pca':
        print("PCA of the scattering coefficients")
        pca_scattering(scat_dir, scatpcs_dir,
                       pca_save_file=obj_save_file,
                       pca_load_file=obj_load_file,
                       num_components=pcs_dim)  # pca_save_file.parent.exists()
    elif pca_algorithm == 'ipca':
        print("Incremental PCA of the scattering coefficients")
        ipca_large_scattering(scat_dir, J,
                              image_len=3 * image_size**2,
                              ipcs_save_dir=scatpcs_dir,
                              ipca_save_file=obj_save_file,
                              ipca_load_file=obj_load_file,
                              num_components=pcs_dim, batch_size=ipca_batch_size)  # int(5e3), int(1e4)
    else:
        raise ValueError("Undefined pca algorithm at 'pca: algorithm:' in config file.")

    print("Done")


def _pcscoef_compstd(comp_std_1st,pcs_vars):
    # 主成分分散変換係数ベクトル
    pcs_var_1pc = 1 - comp_std_1st ** 2  # 第一主成分の分散(変換後)

    pcs1_var_ratio = (pcs_var_1pc / pcs_vars[0])  # 第一主成分の分散(変換後)/第一主成分の分散(変換前) a
    base_pcs_vars = pcs_vars * pcs1_var_ratio  # 定数＊主成分分散ベクトル　( 第一主成分の分散は変換後の値 ) a * ov
    base_pos_vars = base_pcs_vars + comp_std_1st ** 2  # 定数＊主成分分散ベクトル + コンポネント分散　（第一成分は分散１）bv = a * ov + c^2

    pcs_adj_coef = np.sqrt(pcs1_var_ratio) / np.sqrt(base_pos_vars)  # cv = sqrt(a) / sqrt(bv)

    # コンポネント標準偏差保存
    comp_stds = comp_std_1st / np.sqrt(base_pos_vars)
    comp_stds = torch.from_numpy(comp_stds).float()

    return pcs_adj_coef, comp_stds


def pca_scattering_with_varscales(scat_dir, pcs_save_dir, pca_save_file=None,
                                  compstd_save_file=None, pca_load_file=None,
                                  num_components=None, comp_std_1st=0.1):

    if pca_save_file is None and pca_load_file is None:
        raise ValueError("No argument. Specify one argument, 'pca_save_file' or 'pca_load_file'.")
    if pca_save_file is not None and pca_load_file is not None:
        raise ValueError("Two arguments. Specify only one argument, 'pca_save_file' or 'pca_load_file'.")
    if pca_save_file is not None and compstd_save_file is None:
        raise ValueError("If 'pca_save_file' is specified, 'compstd_save_file' must be specified.")
    if pca_save_file is not None and num_components is None:
        raise ValueError("If 'pca_save_file' is specified, 'num_components' must be specified.")

    pcs_save_dir.mkdir(parents=True,exist_ok=True)

    scat,scat_files = load_scat_set(scat_dir)

    if pca_save_file is not None:
        pca = PCA(n_components=num_components,whiten=False,svd_solver='arpack')
        pca.fit(scat)
        #---
        pcs_adj_coef, comp_stds = _pcscoef_compstd(comp_std_1st,pca.explained_variance_)
        #---
        joblib.dump((pca,pcs_adj_coef), pca_save_file, compress=True)
        torch.save(comp_stds, compstd_save_file)

    elif pca_load_file is not None:
        pca,pcs_adj_coef = joblib.load(pca_load_file)

    scat_pcs = pca.transform(scat)
    # Scat pcsの分散を変更
    scat_pcs = scat_pcs * pcs_adj_coef

    # 主成分スコア　保存
    for i,f in enumerate(scat_files):
        save_file = pcs_save_dir / f'{f.stem}_pcs.npy'
        np.save(save_file, scat_pcs[i])


def ipca_large_scattering_with_varscales(scat_dir,J,image_len,ipcs_save_dir,ipca_save_file=None,compstd_save_file=None,
                                         ipca_load_file=None, num_components=None,comp_std_1st=0.1,batch_size=None):

    if ipca_save_file is None and ipca_load_file is None:
        raise ValueError("One of the arguments 'ipca_save_file' or 'ipca_load_file' must be specified.")
    if ipca_save_file is not None and ipca_load_file is not None:
        raise ValueError("Only one of the arguments 'ipca_save_file' or 'ipca_load_file' can be specified.")
    if ipca_save_file is not None and compstd_save_file is None:
        raise ValueError("If 'ipca_save_file' is specified, 'compstd_save_file' must be specified.")
    ipca_params_specified = num_components is not None and batch_size is not None
    if ipca_save_file is not None and not ipca_params_specified:
        raise ValueError("If 'ipca_save_file' is specified, 'num_components' and 'batch_size' must be specified.")

    ipcs_save_dir.mkdir(parents=True,exist_ok=True)

    scats_memmap,scat_files = load_scats_to_memmap(scat_dir,J,image_len)

    if ipca_save_file is not None:
        ipca = IncrementalPCA(n_components=num_components,whiten=False,batch_size=batch_size)
        ipca.fit(scats_memmap)

        pcs_adj_coef, comp_stds = _pcscoef_compstd(comp_std_1st,ipca.explained_variance_)

        joblib.dump((ipca,pcs_adj_coef), ipca_save_file, compress=True)
        torch.save(comp_stds,  compstd_save_file)

    elif ipca_load_file is not None:
        ipca,pcs_adj_coef = joblib.load(ipca_load_file)
        if batch_size is None:
            batch_size = len(scat_files)

    for batch in gen_batches(len(scat_files),batch_size):
        scat_pcs = ipca.transform(scats_memmap[batch])

        scat_pcs = scat_pcs * pcs_adj_coef
        scat_pcs = scat_pcs.astype(np.float32)

        # 主成分スコア　保存
        for i,f in enumerate(scat_files[batch]):
            save_file = ipcs_save_dir / f'{f.stem}_pcs.npy'
            np.save(save_file, scat_pcs[i])


def encode_images_with_varscales(image_size,image_dir,J,scat_batch_size,scat_dir, pca_algorithm, scatpcs_dir, # encode_for_ssgm
           obj_save_file=None,compstd_save_file=None,obj_load_file=None,pcs_dim = None,comp_std_1st = 0.1,ipca_batch_size = None):  # encode_image_with_varscales()

    print("Image scattering")
    scattering_imageset(J, image_size, scat_batch_size, image_dir, scat_dir)

    if pca_algorithm == 'pca':
        print("PCA of the scattering coefficients")
        pca_scattering_with_varscales(scat_dir, scatpcs_dir,
                       pca_save_file=obj_save_file,
                       compstd_save_file=compstd_save_file,
                       pca_load_file=obj_load_file,
                       num_components=pcs_dim,
                       comp_std_1st = comp_std_1st)
    elif pca_algorithm == 'ipca':
        print("Incremental PCA of the scattering coefficients")
        ipca_large_scattering_with_varscales(scat_dir, J,
                              image_len=3 * image_size**2,
                              ipcs_save_dir=scatpcs_dir,
                              ipca_save_file=obj_save_file,
                              compstd_save_file=compstd_save_file,
                              ipca_load_file=obj_load_file,
                              num_components=pcs_dim, comp_std_1st = comp_std_1st,
                              batch_size=ipca_batch_size)
    else:
        raise ValueError("Undefined pca algorithm at 'pca: algorithm:' in config file.")
    print("Done")
