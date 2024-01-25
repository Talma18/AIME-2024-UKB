import json
import os
from enum import IntEnum

from dotenv import load_dotenv

from .base_handler import BaseHandler

load_dotenv()

DATA_PATH = os.getenv("DATA_PATH")


class ICDLevel(IntEnum):
    Chapter = 1
    Block = 2
    Code = 3


class ICDHandler(BaseHandler):
    def __init__(self, default_level: ICDLevel | int = ICDLevel.Code, vocab: dict[str, int] = None):
        if isinstance(default_level, int):
            default_level = ICDLevel(default_level)

        with open(os.path.join(DATA_PATH, "processed/helper_files", "icd10_hierarchy.json")) as f:
            icd_hierarchy = json.load(f)

        # [from][to](from_code) -> to_code
        self.converters = {
            ICDLevel.Chapter: {ICDLevel.Chapter: lambda x: x},
            ICDLevel.Block: {
                ICDLevel.Chapter: lambda x: icd_hierarchy["block_to_chapter"][x],
                ICDLevel.Block: lambda x: x,
            },
            ICDLevel.Code: {
                ICDLevel.Block: lambda x: icd_hierarchy["code_to_block"][x],
                ICDLevel.Chapter: lambda x: icd_hierarchy["code_to_chapter"][x],
                ICDLevel.Code: lambda x: x,
            },
        }

        # [level][code] -> name
        self.decoders = {
            ICDLevel.Chapter: icd_hierarchy["chapter_to_name"],
            ICDLevel.Block: icd_hierarchy["block_to_name"],
            ICDLevel.Code: icd_hierarchy["code_to_name"],
        }

        super().__init__(default_level, "D", vocab)

    def _convert_level_token(self, token: str, from_level: ICDLevel, to_level: ICDLevel) -> str:
        return self.converters[from_level][to_level](token)

    def _get_decoder(self, level: ICDLevel) -> dict[str, str]:
        return self.decoders[level]
