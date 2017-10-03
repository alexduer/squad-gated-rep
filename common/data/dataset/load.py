from common.data.dataset.preprocess import preprocess
from common.data.dataset.proc_dataset import ProcDataset
from common.data.dataset.raw_dataset import load_raw_dataset
from common.util.cache import cache


def load_dataset(dataset_path: str, data_dir: str, cache_dir: str) -> ProcDataset:
    raw_dataset = load_raw_dataset(dataset_path)
    cache_key = 'load_dataset' + dataset_path
    return cache(lambda: preprocess(raw_dataset, data_dir), cache_dir, cache_key)
