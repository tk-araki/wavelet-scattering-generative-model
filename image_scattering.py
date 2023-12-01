from pathlib import Path
import joblib
import tempfile

from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from kymatio.torch import Scattering2D
from sklearn.decomposition import PCA,IncrementalPCA
from sklearn.utils import gen_batches


from datasets import ImageDataset

# os.environ["KYMATIO_BACKEND_2D"] = "skcuda"

def scattering_imageset(J,image_size, batch_size, image_dir, save_dir): # ToDo:scattering_imageに名称変更

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
            np.save(save_file, img_scat_batch[i]) # (3, 417, 4, 4) when J=4

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
        # scat_pcs = pca.fit_transform(scat)
        pca.fit(scat)
        joblib.dump(pca, pca_save_file, compress=True) # 　PCAパラメータ（主成分ベクトル等）保存

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
    scat_dim = int(image_len*(1+Q*J+Q*Q*J*(J-1)/2)/(2**(2*J)))# 20016 if J=4
    fp = tempfile.NamedTemporaryFile()
    scats_mmap = np.memmap(fp, dtype='float32', mode='w+', shape=(len(scat_files), scat_dim))

    for n, f in enumerate(scat_files):
        scats_mmap[n, :] = np.load(f).ravel() # scats = [np.load(f).ravel() for f in scat_files]

    # print(f"fp.name={fp.name}")
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
        ipca = joblib.load(ipca_load_file) #読み出し
        if batch_size is None:
            batch_size = len(scat_files)

    for batch in gen_batches(len(scat_files),batch_size):
        scat_pcs = ipca.transform(scats_memmap[batch]).astype(np.float32)

        # 主成分スコア　保存
        for i,f in enumerate(scat_files[batch]):
            save_file = ipcs_save_dir / f'{f.stem}_pcs.npy'
            np.save(save_file, scat_pcs[i])
