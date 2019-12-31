"""
This module provides data loaders and transformers for ADE20K datasets.
"""
from .ade import ADE20KSegmentation

datasets = {
    'ade20k': ADE20KSegmentation,
}

def get_segmentation_dataset(name, **kwargs):
    """Segmentation Datasets"""
    return datasets[name.lower()](**kwargs)
