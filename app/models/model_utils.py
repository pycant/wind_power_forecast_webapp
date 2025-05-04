# model_utils.py
import json
from pathlib import Path
from typing import Dict, Any, Optional
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)



class MyModels:
    def __init__(self, config_path: str="models/configure.json", data_path: str = "data/processed/",output_path: str="output/" ):
        self.config_path = config_path
        self.data_path = data_path
        self.output_path = output_path
        