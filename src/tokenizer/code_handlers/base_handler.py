from abc import ABC, abstractmethod
from enum import IntEnum
from typing import Optional


class BaseHandler(ABC):
    def __init__(self, default_level: IntEnum, leading_character: str, vocab: dict[str, int] = None):
        self.default_level = default_level
        self.level_enum_cls = type(default_level)
        self.finest_level = max(self.level_enum_cls)

        self.leading_character = leading_character

        # construct level 'vocab' dicts if a pretrained tokenizer loads the handler
        if vocab is not None:
            # conversions from coarser to finer levels is impossible, this is a many-to-one mapping
            # depending on the default level, not all conversion dicts are needed

            # filter to only handled tokens
            handled_tokens = [token for token in vocab if self._is_handled_token(token)]

            # map all handled tokens to level tokens on all possible levels
            self.tokens_to_level_tokens = {
                level: {token: self._token_to_level_token(token, level) for token in handled_tokens}
                if self.valid_conversion(default_level, level)
                else None
                for level in self.level_enum_cls
            }

            # sort all unique level tokens for each possible level and map them to level ids
            self.level_token_to_level_id = {
                level: {
                    level_token: i
                    for i, level_token in enumerate(sorted(list(set(self.tokens_to_level_tokens[level].values()))))
                }
                if self.valid_conversion(default_level, level)
                else None
                for level in self.level_enum_cls
            }

            # map all level ids to their corresponding level tokens on all possible levels
            self.level_id_to_level_token = {
                level: {v: k for k, v in self.level_token_to_level_id[level].items()}
                if self.valid_conversion(default_level, level)
                else None
                for level in self.level_enum_cls
            }

            self.token_id_to_level_id = {
                level: {
                    vocab[token]: self.level_token_to_level_id[level][level_token]
                    for token, level_token in self.tokens_to_level_tokens[level].items()
                }
                if self.valid_conversion(default_level, level)
                else None
                for level in self.level_enum_cls
            }

    @abstractmethod
    def _convert_level_token(self, token: str, from_level: IntEnum, to_level: IntEnum) -> str:
        pass

    @abstractmethod
    def _get_decoder(self, level: IntEnum) -> dict[str, str]:
        pass

    @staticmethod
    def valid_conversion(from_level: IntEnum, to_level: IntEnum) -> bool:
        return from_level >= to_level

    def _is_handled_token(self, token: str) -> bool:
        return token.startswith(f"{self.leading_character}_")

    def _token_to_level_token(self, token: str, level: IntEnum) -> str:
        default_level_token = token[2:]
        return self._convert_level_token(default_level_token, self.default_level, level)

    def finest_level_to_token(self, code: str) -> str:
        return self.leading_character + "_" + self._convert_level_token(code, self.finest_level, self.default_level)

    def finest_level_to_level_token(self, code: str | list[str], level: IntEnum) -> str | list[str]:
        if isinstance(code, str):
            return self._convert_level_token(code, self.finest_level, level)

        return [self._convert_level_token(c, self.finest_level, level) for c in code]

    def _token_ids_to_level_ids(self, token_ids: list[int], level: IntEnum) -> list[int]:
        # convert token_ids to icd_level_ids
        return [self.token_id_to_level_id[level].get(token_id, -1) for token_id in token_ids]

    def token_ids_to_level_ids(
        self, token_ids: list[int], level: Optional[IntEnum] = None
    ) -> list[int] | dict[IntEnum, list[int]]:
        # if no level is given convert to all possible levels
        if level is not None:
            return self._token_ids_to_level_ids(token_ids, level)

        return {
            level: self._token_ids_to_level_ids(token_ids, level)
            if self.valid_conversion(self.default_level, level)
            else None
            for level in self.level_enum_cls
        }

    def _decode_level_ids(self, level_ids: list[int], level: IntEnum) -> list[str]:
        # decode level_ids to level_tokens
        return [self.level_id_to_level_token[level].get(level_id, "<unk>") for level_id in level_ids]

    def decode_level_ids(
        self, level_ids: list[int] | dict[IntEnum, list[int]], level: Optional[IntEnum] = None
    ) -> list[str] | dict[IntEnum, list[str]]:
        if level is not None:
            return self._decode_level_ids(level_ids, level)

        return {
            level: self._decode_level_ids(level_ids[level], level)
            if self.valid_conversion(self.default_level, level)
            else None
            for level in self.level_enum_cls
        }

    def _decode_level_tokens(self, level_tokens: str | list[str], level: IntEnum) -> str | list[str]:
        # decode level_tokens to level_names
        if isinstance(level_tokens, str):
            return self._get_decoder(level).get(level_tokens, f"Unknown: '{level_tokens}'")
        return [self._get_decoder(level).get(token, f"Unknown: '{token}'") for token in level_tokens]

    def decode_level_tokens(
        self, level_tokens: str | list[str] | dict[IntEnum, list[str]], level: Optional[IntEnum] = None
    ) -> str | list[str] | dict[IntEnum, list[str]]:
        if level is not None:
            return self._decode_level_tokens(level_tokens, level)

        return {
            level: self._decode_level_tokens(level_tokens[level], level)
            if self.valid_conversion(self.default_level, level)
            else None
            for level in self.level_enum_cls
        }

    def decode_token(self, token: str) -> str:
        # decode tokens to default level names
        return self._get_decoder(self.default_level).get(token[2:], f"Unknown: '{token}'")

    def vocab_size(self, level: Optional[IntEnum]) -> int:
        if level is None:
            level = self.default_level
        d = self.level_token_to_level_id[level]
        return None if d is None else len(d)
