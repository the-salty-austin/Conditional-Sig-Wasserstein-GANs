import os
import pickle
import datetime
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

def transfer_percentage_seq(x):
    # base value (t=0)
    start = x[0,:]
    start = start.expand(x.shape)
    # remove zero start
    idx_ = torch.nonzero(start == 0, as_tuple=False).tolist()
    if idx_:
        idx_ = idx_[0]
    idx_ = list(set(list(range(x.shape[0]))) - set(idx_))

    new_x = x[idx_, ...]
    new_start = start[idx_, ...]

    new_x = (new_x - new_start) / new_start
    return new_x

# def rolling_window(x, x_lag, add_batch_dim=True):
#     if add_batch_dim:
#         x = x[None, ...]
#     return torch.cat([x[:, t:t + x_lag] for t in range(x.shape[1] - x_lag)], dim=0)

def zero_based_rolling_window(x, p_lag, q_lag, add_batch_dim=True):
    window = p_lag + q_lag
    # print(window, x.shape, x.shape[0]-window)

    combined_segments = []
    for t in range(x.shape[0]-window):
        # p_segment = x[t      :t+p_lag]
        # p_segment = transfer_percentage_seq(p_segment)
        # q_segment = x[t+p_lag:t+window]
        # q_segment = transfer_percentage_seq(q_segment)

        # combined_segment = torch.cat([p_segment, q_segment], dim=0)
        # combined_segments.append(combined_segment)
        segment = x[ t : t+p_lag+q_lag ]
        combined_segment = transfer_percentage_seq(segment)
        combined_segments.append(combined_segment)

    if add_batch_dim:
        return torch.stack(combined_segments)[None, ...]
    
    return torch.stack(combined_segments)

class Pipeline:
    def __init__(self, steps):
        """ Pre- and postprocessing pipeline. """
        self.steps = steps

    def transform(self, x, until=None):
        x = x.clone()
        for n, step in self.steps:
            if n == until:
                break
            x = step.transform(x)
        return x

    def inverse_transform(self, x, until=None):
        for n, step in self.steps[::-1]:
            if n == until:
                break
            x = step.inverse_transform(x)
        return x


class StandardScalerTS():
    """ Standard scales a given (indexed) input vector along the specified axis. """

    def __init__(self, axis=(1)):
        self.mean = None
        self.std = None
        self.axis = axis

    def transform(self, x):
        if self.mean is None:
            self.mean = torch.mean(x, dim=self.axis)
            self.std = torch.std(x, dim=self.axis)
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def inverse_transform(self, x):
        return x * self.std.to(x.device) + self.mean.to(x.device)


def get_equities_dataset(assets=('SPX', 'DJI'), with_vol=True):
    """
    Get different returns series.
    """
    oxford = pd.read_csv('./data/oxfordmanrealizedvolatilityindices.csv')

    start = '2005-01-01 00:00:00+01:00'
    end = '2020-01-01 00:00:00+01:00'

    # if assets == ('SPX',):
    #     df_asset = oxford[oxford['Symbol'] == '.SPX'].set_index(['Unnamed: 0'])  # [start:end]
    #     price = np.log(df_asset[['close_price']].values)
    #     rtn = (price[1:] - price[:-1]).reshape(1, -1, 1)
    #     vol = np.log(df_asset[['medrv']].values[-rtn.shape[1]:]).reshape(1, -1, 1)
        
    #     if with_vol:
    #         data_raw = np.concatenate([rtn, vol], axis=-1)
    #     else:
    #         data_raw = np.concatenate([rtn], axis=-1)
    if assets == ('SPX',):
        df_asset = oxford[oxford['Symbol'] == '.SPX'].set_index(['Unnamed: 0'])  # [start:end]
        price = df_asset[['close_price']].values
        data_raw = np.concatenate([price], axis=-1)[None, ...]
    # elif assets == ('SPX', 'DJI'):
    #     df_spx = oxford[oxford['Symbol'] == '.SPX'].set_index(['Unnamed: 0'])[start:end]
    #     df_dji = oxford[oxford['Symbol'] == '.DJI'].set_index(['Unnamed: 0'])[start:end]
    #     index = df_dji.index.intersection(df_spx.index)
    #     df_dji = df_dji.loc[index]
    #     df_spx = df_spx.loc[index]
    #     price_spx = np.log(df_spx[['close_price']].values)
    #     rtn_spx = (price_spx[1:] - price_spx[:-1]).reshape(1, -1, 1)
    #     vol_spx = np.log(df_spx[['medrv']].values).reshape(1, -1, 1)
    #     price_dji = np.log(df_dji[['close_price']].values)
    #     rtn_dji = (price_dji[1:] - price_dji[:-1]).reshape(1, -1, 1)
    #     vol_dji = np.log(df_dji[['medrv']].values).reshape(1, -1, 1)
    elif assets == ('SPX', 'DJI'):
        df_spx = oxford[oxford['Symbol'] == '.SPX'].set_index(['Unnamed: 0'])[start:end]
        df_dji = oxford[oxford['Symbol'] == '.DJI'].set_index(['Unnamed: 0'])[start:end]
        index = df_dji.index.intersection(df_spx.index)
        df_dji = df_dji.loc[index]
        df_spx = df_spx.loc[index]
        price_spx = np.log(df_spx[['close_price']].values)
        price_dji = np.log(df_dji[['close_price']].values)
        data_raw = np.concatenate([price_spx, price_dji], axis=-1)[None, ...]
    else:
        raise NotImplementedError()

    data_raw = torch.from_numpy(data_raw).float()
    # pipeline = Pipeline(steps=[('standard_scale', StandardScalerTS())])
    # data_preprocessed = pipeline.transform(data_raw)
    # return pipeline, data_raw, data_preprocessed
    return data_raw


def get_binance_dataset(assets=('BTC', 'ETH')):
    datasets = []
    start = datetime.datetime(1970,1,1)
    end = datetime.datetime(2023,12,31)
    
    for asset in assets:
        data_raw = pd.read_csv(f'.\\data\\binance\\{asset}USDT_1h.csv').set_index('Unnamed: 0')
        data_raw['timestamp'] = pd.to_datetime(data_raw['timestamp'], format='%Y-%m-%d %H:%M:%S') 
        data_raw = data_raw.set_index(['timestamp'])

        # if data_raw.index[0] > start:
        #     start = data.index[0]
        # if data_raw.index[-1] < end:
        #     end = data.index[-1]

        data_raw = data_raw[start:end][["close"]].to_numpy(dtype='float')
        data_raw = torch.FloatTensor(data_raw)

        data = torch.FloatTensor(data_raw)
        # data = rolling_window(data, 5, add_batch_dim=True)
        # data = transfer_percentage_seq(data)

        # print(data)

        datasets.append(data)
        print(f'\tRolled data for training, shape {list(data.shape)}')

    data_processed = torch.cat(datasets, dim=-1)[None,:,:]

    return data_processed


def get_var_dataset(window_size, batch_size=5000, dim=3, phi=0.8, sigma=0.5):
    def multi_AR(window_size, dim=3, phi=0.8, sigma=0.5, burn_in=200):
        window_size = window_size + burn_in
        xt = np.zeros((window_size, dim))
        one = np.ones(dim)
        ide = np.identity(dim)
        MU = np.zeros(dim)
        COV = sigma * one + (1 - sigma) * ide
        W = np.random.multivariate_normal(MU, COV, window_size)
        for i in range(dim):
            xt[0, i] = 0
        for t in range(window_size - 1):
            xt[t + 1] = phi * xt[t] + W[t]
        return xt[burn_in:]

    var_samples = []
    for i in range(batch_size):
        tmp = multi_AR(window_size, dim, phi=phi, sigma=sigma)
        var_samples.append(tmp)
    data_raw = torch.from_numpy(np.array(var_samples)).float()

    def get_pipeline():
        transforms = list()
        transforms.append(('standard_scale', StandardScalerTS(axis=(0, 1))))  # standard scale
        pipeline = Pipeline(steps=transforms)
        return pipeline

    pipeline = get_pipeline()
    data_preprocessed = pipeline.transform(data_raw)
    return pipeline, data_raw, data_preprocessed


def get_arch_dataset(window_size, lag=4, bt=0.055, N=5000, dim=1):
    """
    Creates the dataset: loads data.

    :param data_path: :param t_lag: :param device: :return:
    """

    def get_raw_data(N=5000, lag=4, T=2000, omega=0.00001, bt=0.055, burn_in=2000):
        beta = bt * np.ones(lag)
        eps = np.random.randn(N, T + burn_in)
        logrtn = np.zeros((N, T + burn_in))

        initial_arch = omega / (1 - beta[0])

        arch = initial_arch + np.zeros((N, T + burn_in))

        logrtn[:, :lag] = np.sqrt(arch[:, :lag]) * eps[:, :lag]

        for t in range(lag - 1, T + burn_in - 1):
            arch[:, t + 1] = omega + np.matmul(beta.reshape(1, -1), np.square(
                logrtn[:, t - lag + 1:t + 1]).transpose())  # * (logrtn[:, t] < 0.)
            logrtn[:, t + 1] = np.sqrt(arch[:, t + 1]) * eps[:, t + 1]
        return arch[:, burn_in:], logrtn[:, burn_in:]

    pipeline = Pipeline(steps=[('standard_scale', StandardScalerTS(axis=(0, 1)))])
    _, logrtn = get_raw_data(T=window_size, N=N, bt=bt)
    data_raw = torch.from_numpy(logrtn[..., None]).float()
    data_pre = pipeline.transform(data_raw)
    return pipeline, data_raw, data_pre


def get_mit_arrythmia_dataset(filenames):
    DATA_DIR = './data/mit-bih-arrhythmia-database-1.0.0/'
    import wfdb
    records = list()
    for fn in filenames:
        records.append(wfdb.rdsamp(os.path.join(DATA_DIR, fn), sampto=3000)[0][None, ...])
    records = np.concatenate(records, axis=0)
    records = np.log(5 * (records - records.min() + 1))
    data_raw = torch.from_numpy(records).float()
    pipeline = Pipeline(steps=[('standard_scale', StandardScalerTS(axis=(0, 1)))])
    data_pre = pipeline.transform(data_raw)
    return pipeline, data_raw, data_pre


def get_data(data_type, p, q, **data_params):
    
    print(f"get_data() - p={p} q={q} p+q={p+q}")

    if data_type == 'VAR':
        pipeline, x_real_raw, x_real = get_var_dataset(
            40000, batch_size=1, **data_params
        )
    elif data_type == 'STOCKS':
        # pipeline, x_real_raw, x_real = get_equities_dataset(**data_params)
        x_real = get_equities_dataset(**data_params)
    elif data_type == 'BINANCE':
        x_real = get_binance_dataset(**data_params)
    elif data_type == 'ARCH':
        pipeline, x_real_raw, x_real = get_arch_dataset(
            40000, N=1, **data_params
        )
    elif data_type == 'ECG':
        pipeline, x_real_raw, x_real = get_mit_arrythmia_dataset(**data_params)
    else:
        raise NotImplementedError('Dataset %s not valid' % data_type)
    
    print("get_data() before rolling", x_real.shape)
    # print(x_real)
    assert x_real.shape[0] == 1
    x_real = zero_based_rolling_window(x_real[0], p, q, add_batch_dim=False)
    print("get_data() after rolling",  x_real.shape)
    return x_real


def download_man_ahl_dataset():
    import requests
    from zipfile import ZipFile
    url = 'https://realized.oxford-man.ox.ac.uk/images/oxfordmanrealizedvolatilityindices.zip'
    r = requests.get(url)
    with open('./oxford.zip', 'wb') as f:
        pbar = tqdm(unit="B", total=int(r.headers['Content-Length']))
        for chunk in r.iter_content(chunk_size=100 * 1024):
            if chunk:
                pbar.update(len(chunk))
                f.write(r.content)
    zf = ZipFile('./oxford.zip')
    zf.extractall(path='./data')
    zf.close()
    os.remove('./oxford.zip')


def download_mit_ecg_dataset():
    import requests
    from zipfile import ZipFile
    url = 'https://storage.googleapis.com/mitdb-1.0.0.physionet.org/mit-bih-arrhythmia-database-1.0.0.zip'
    r = requests.get(url)
    with open('./mit_db.zip', 'wb') as f:
        pbar = tqdm(unit="B", total=int(r.headers['Content-Length']))
        for chunk in r.iter_content(chunk_size=100 * 1024):
            if chunk:
                pbar.update(len(chunk))
                f.write(r.content)
    zf = ZipFile('./mit_db.zip')
    zf.extractall(path='./data')
    zf.close()
    os.remove('./mit_db.zip')


if __name__ == "__main__":

    # print("--------Below 2 are equivalent-------")
    # pipeline, data_raw, data_preprocessed = get_arch_dataset(window_size=5)
    # print(pipeline)
    # print(data_preprocessed[0])
    # print((data_raw[0] - torch.mean(data_raw, dim=(0,1))) / torch.std(data_raw, dim=(0,1)))
    # print("--------Above 2 are equivalent-------")

    res = get_data("BINANCE", 3, 10)
    res = get_data("STOCKS", 13, 10)
    # print(res)