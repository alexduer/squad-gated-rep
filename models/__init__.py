from typing import Dict, Any, List

from common.data.dataset.proc_dataset import ProcDataset
from models.common.base import BaseModel


def create_model(name: str, config: Dict[str, Any], datasets: List[ProcDataset]) -> BaseModel:
    module = __import__('models.{}'.format(name.lower()), fromlist=[name])
    class_ = getattr(module, name)
    instance = class_(config, datasets)
    return instance
