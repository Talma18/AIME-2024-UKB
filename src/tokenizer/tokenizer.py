import json
import os
from typing import Dict, List, Optional, Union, Mapping, Sized

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from transformers.tokenization_utils_base import (
    PreTrainedTokenizerBase,
    AddedToken,
    BatchEncoding,
    PaddingStrategy,
    EncodedInput,
)
from transformers.utils import TensorType, is_tf_tensor, is_torch_tensor, to_py_obj
from tqdm.auto import tqdm

from .code_handlers import ICDHandler, ICDLevel

load_dotenv()

DATA_PATH = os.getenv("DATA_PATH")


class Tokenizer(PreTrainedTokenizerBase):
    model_input_names = []

    def __init__(
        self,
        name_or_path=None,
        errors="replace",
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
        sep_token="<sep>",
        unk_token="<unk>",
        mask_token="<mask>",
        add_prefix_space=False,
        use_personal=True,
        use_lab=True,
        **kwargs,
    ):
        
        # model input names
        model_input_types = [
            "input_ids",
            "attention_mask",
            "token_type_ids",
            "measurement_mask",
            "year_ids",
            "month_ids",
            "day_ids",
        ]

        self.modalities = []

        def add_modality(modality):
            self.modalities.append(modality)
            self.model_input_names += list(map(lambda name: modality+'_'+name, model_input_types))
        
        add_modality('disease')
        add_modality('drug')
        
        if use_personal:
            add_modality('personal')

        if use_lab:
            add_modality('lab')

        # Add special tokens
        bos_token = AddedToken(bos_token, lstrip=False, rstrip=False) if isinstance(bos_token, str) else bos_token
        eos_token = AddedToken(eos_token, lstrip=False, rstrip=False) if isinstance(eos_token, str) else eos_token
        unk_token = AddedToken(unk_token, lstrip=False, rstrip=False) if isinstance(unk_token, str) else unk_token
        pad_token = AddedToken(pad_token, lstrip=False, rstrip=False) if isinstance(pad_token, str) else pad_token
        mask_token = AddedToken(mask_token, lstrip=True, rstrip=False) if isinstance(mask_token, str) else mask_token
        sep_token = AddedToken(sep_token, lstrip=False, rstrip=False) if isinstance(sep_token, str) else sep_token

        self.icd_handler = ICDHandler(default_level=ICDLevel.Code)

        path = os.path.join(DATA_PATH, 'raw/helper_files/Data_Dictionary_Showcase.tsv')
        self.full_field_info_df = pd.read_csv(path, sep='\t').set_index('FieldID')

        def key_to_int(data):
            return {int(key): value for key, value in data.items()}

        path = os.path.join(DATA_PATH, 'processed/helper_files/field_to_modality.json')
        with open(path) as f:
            self.field_to_modality: dict[int, str] = key_to_int(json.load(f))
        self.modality_to_ids = {
            modality: [
                field_id for field_id, mod in self.field_to_modality.items() if mod == modality
            ] for modality in self.modalities
        }

        path = os.path.join(DATA_PATH, 'processed/helper_files/field_timevariant.json')
        with open(path) as f:
            self.field_timevariant: dict[int, bool] = key_to_int(json.load(f))

        path = os.path.join(DATA_PATH, 'processed/helper_files/special_field_values.json')
        with open(path) as f:
            self.special_field_values: dict[int, dict[int, str]] = {int(k): key_to_int(v) for k,v in json.load(f).items()}

        # Load pretrained
        if name_or_path is not None:
            self._load_pretrained(name_or_path)
        
        # Set token types
        self.special_token_type = "<SPECIAL>"
        self.uid_token_type = "<UID>"
        self.numerical_measurement_token_type = "<NUMERICAL_MEASUREMENT>"
        self.categorical_value_token_type = "<CATEGORICAL_MEASUREMENT>"

        self.token_types = [
            self.special_token_type,
            self.uid_token_type,
            self.numerical_measurement_token_type,
            self.categorical_value_token_type,
        ]

        self.special_token_type_id = self.token_types.index(self.special_token_type)
        self.uid_token_type_id = self.token_types.index(self.uid_token_type)
        self.numerical_measurement_token_type_id = self.token_types.index(self.numerical_measurement_token_type)
        self.categorical_value_token_type_id = self.token_types.index(self.categorical_value_token_type)

        # Init base tokenizer
        super().__init__(
            errors=errors,
            pad_token=pad_token,
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            unk_token=unk_token,
            mask_token=mask_token,
            add_prefix_space=add_prefix_space,
            name_or_path=name_or_path,
            **kwargs,
        )
        
        # Add extra arguments to init_kwargs to be saved
        self.init_kwargs["use_personal"] = use_personal
        self.init_kwargs["use_lab"] = use_lab

    def _load_pretrained(self, name_or_path):
        
        def load_tokens(filename):
            with open(os.path.join(name_or_path, filename)) as token_file:
                return {token.strip(): i for i, token in enumerate(token_file.readlines())}

        self.encoder = load_tokens(f"vocab.txt")
        self.decoder = {v: k for k, v in self.encoder.items()}

        self.type_encoder = load_tokens(f"types.txt")
        self.type_decoder = {v: k for k, v in self.type_encoder.items()}

        with open(os.path.join(name_or_path, f"token_types.txt")) as f:
            self.token_to_type = {
                token: token_type for token, token_type in list(map(lambda x: x.split(), f.readlines()))
            }

            self.token_id_to_type_id = {
                self.encoder[token]: self.type_encoder[token_type] for token, token_type in self.token_to_type.items()
            }

        self.year_encoder = load_tokens("years.txt")
        self.year_decoder = {v: k for k, v in self.year_encoder.items()}

        self.month_encoder = load_tokens("months.txt")
        self.month_decoder = {v: k for k, v in self.month_encoder.items()}

        self.day_encoder = load_tokens("days.txt")
        self.day_decoder = {v: k for k, v in self.day_encoder.items()}

        self.field_info_df = pd.read_csv(os.path.join(name_or_path, 'field_info_df.csv'), index_col=0)

    @staticmethod
    def format_name(name):
        return name.replace(' ', '_')

    def make_pretrained(
        self,
        path,
        data,
    ):
        if os.makedirs(path, exist_ok=True):
            os.mkdir(path)

        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump({k: v.content if isinstance(v, AddedToken) else v for k, v in self.init_kwargs.items()}, f)

        special_tokens = [
            self.pad_token,
            self.bos_token,
            self.eos_token,
            self.sep_token,
            self.mask_token,
            self.unk_token,
        ]

        # drug and disease tokens
        chembl_id_tokens = sorted(data['drug_codes'].explode().dropna().unique())

        # replace rare icd10 codes with a 'RARE_DISEASE' token
        icd10_code_rate = data['disease_codes'].explode().dropna().value_counts() / len(data)
        icd10_code_tokens = ['RARE_DISEASE'] + sorted(icd10_code_rate[icd10_code_rate >= 1 / 50000].index)

        # field id tokens
        field_ids = sorted(list(set([int(c.split('-')[0]) for c in data.columns if '-' in c])))
        field_to_token_series = self.full_field_info_df.loc[field_ids, 'Field'].apply(self.format_name)
        field_tokens = field_to_token_series.values.tolist()

        # special value tokens
        special_value_tokens = []
        for field in self.special_field_values:
            field_token = field_to_token_series.loc[field]
            for v in self.special_field_values[field].values():
                special_value_tokens.append(field_token + '_' + self.format_name(v))

        # categorical measurement tokens
        categorical_value_tokens = []
        processed = set()
        for c in data:
            if not pd.api.types.is_categorical_dtype(data[c]):
                continue

            field_id = int(c.split('-')[0])

            if field_id in processed:
                continue

            processed.add(field_id)
            field_token = field_to_token_series.loc[field_id]

            for v in data[c].dtype.categories:
                categorical_value_tokens.append(field_token + '_' + self.format_name(v))

        vocab = (
                special_tokens
                + icd10_code_tokens
                + chembl_id_tokens
                + field_tokens
                + special_value_tokens
                + categorical_value_tokens
        )

        token_types = (
                [self.special_token_type] * len(special_tokens)
                + [self.uid_token_type] * (len(icd10_code_tokens) + len(chembl_id_tokens) + len(field_tokens))
                + [self.categorical_value_token_type] * (len(categorical_value_tokens) + len(special_value_tokens))
        )

        # min and max value of years
        drug_dates = data['drug_dates'].explode().dropna().unique()
        disease_dates = data['disease_dates'].explode().dropna().unique()

        date_columns = ['33-0.0', '53-0.0', '53-1.0', '53-2.0', '53-3.0']
        event_dates = data[date_columns].to_numpy()
        mask = data[date_columns].isna().to_numpy()
        event_dates = event_dates[~mask]

        min_year = int(min([disease_dates.min(), drug_dates.min(), event_dates.min()]).split('-')[0])
        max_year = int(max([disease_dates.max(), drug_dates.max(), event_dates.max()]).split('-')[0])

        # standardization parameters
        standardization_params = {}

        for c in data:
            if not pd.api.types.is_numeric_dtype(data[c]):
                continue

            field_id = int(c.split('-')[0])

            if field_id in standardization_params:
                continue

            standardization_params[field_id] = {
                'mean': data[c].mean(),
                'std': data[c].std(),
            }

        field_info_df = pd.DataFrame.from_dict(standardization_params, orient='index')

        field_info_df.to_csv(os.path.join(path, 'field_info_df.csv'))

        with open(os.path.join(path, "vocab.txt"), "wt") as f:
            f.write("\n".join(vocab))

        with open(os.path.join(path, "types.txt"), "wt") as f:
            f.write("\n".join(self.token_types))

        with open(os.path.join(path, "token_types.txt"), "wt") as f:
            f.write("\n".join([token + " " + token_types for token, token_types in zip(vocab, token_types)]))

        # Save date encodings
        with open(os.path.join(path, "years.txt"), "wt") as f:
            f.write("\n".join(["<no_date>"] + list(map(lambda x: f"{x:04d}", range(min_year, max_year + 1)))))

        with open(os.path.join(path, "months.txt"), "wt") as f:
            f.write("\n".join(["<no_date>"] + list(map(lambda x: f"{x:02d}", range(1, 13)))))

        with open(os.path.join(path, "days.txt"), "wt") as f:
            f.write("\n".join(["<no_date>"] + list(map(lambda x: f"{x:02d}", range(1, 32)))))

        print("Saved pretrained tokenizer to", path)

        self._load_pretrained(path)

    @staticmethod
    def from_pretrained(pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs):
        with open(os.path.join(pretrained_model_name_or_path, "config.json")) as f:
            tokenizer_config = json.load(f)

        tokenizer_config["name_or_path"] = pretrained_model_name_or_path
        tokenizer_config.update(kwargs)

        return Tokenizer(**tokenizer_config)

    def convert_tokens_to_ids(self, tokens: list[str]) -> list[int]:
        if isinstance(tokens, str):
            return self.encoder.get(tokens, self.encoder[self.unk_token])
        else:
            return [self.encoder.get(token, self.encoder[self.unk_token]) for token in tokens]

    def token_ids_to_icd_level_ids(self, icd_ids: list[int]) -> dict[ICDLevel, list[int]]:
        return self.icd_handler.token_ids_to_level_ids(icd_ids)

    def get_token_type_ids(self, tokens_ids: list[int]) -> list[int]:
        return [
            self.token_id_to_type_id[token_id]
            if isinstance(token_id, int)
            else self.numerical_measurement_token_type_id
            for token_id in tokens_ids
        ]

    @staticmethod
    def get_dates(str_date):
        if pd.isna(str_date):
            return ["<no_date>"] * 3

        str_date = str_date.split("-")
        # pad to 3 elements with no_date token
        str_date = str_date + ["<no_date>"] * (3 - len(str_date))
        return str_date

    def tokenize_list_columns(
        self, patient: dict, **kwargs
    ):
        tokens_by_modality = {'disease': [], 'drug': []}
        dates_by_modality = {'disease': [], 'drug': []}

        tokens_by_modality['disease'] = list(map(lambda code: code if code in self.encoder else 'RARE_DISEASE', patient["disease_codes"]))
        dates_by_modality['disease'] = list(map(self.get_dates, patient["disease_dates"]))

        # filter to first N occurences of each drug, to handle the problem of long sequences
        def filter_to_first_n(code_list, date_list, n=5, truncate=256):
            codes = pd.Series(code_list)
            counts = codes.groupby(codes).cumcount()
            mask = counts < n
            return code_list[mask][:truncate], date_list[mask][:truncate]

        drug_codes = np.array(patient["drug_codes"])
        drug_dates = np.array(patient["drug_dates"])

        drug_codes, drug_dates = filter_to_first_n(drug_codes, drug_dates)

        tokens_by_modality['drug'] = list(drug_codes)
        dates_by_modality['drug'] = list(map(self.get_dates, drug_dates))

        return tokens_by_modality, dates_by_modality

    def tokenize_value_columns(self, patient: dict, **kwargs):
        no_date = self.get_dates(None)
        tokens_by_modality = {'personal': [], 'lab': []}
        dates_by_modality = {'personal': [], 'lab': []}

        for c in patient:
            if '-' not in c:
                continue

            field_id, instance_id, array_id = map(int, c.replace('.', '-').split('-'))
            if field_id not in self.field_to_modality or instance_id != 0:
                # Only keep instance 0 because few ppl have multiple instances,
                # and long sequences cause memory issues.
                # There also no need to sort the these sequences in this case
                # because all events have the same date.
                continue

            field_value = patient[c]
            if pd.isna(field_value):
                continue

            modality = self.field_to_modality[field_id]
            if modality not in self.modalities:
                continue

            field_token = self.format_name(self.full_field_info_df.loc[field_id, 'Field'])

            if field_id == 33:  # Handle date of birth separately
                date = self.get_dates(field_value)

                tokens_by_modality[modality].append(field_token)
                dates_by_modality[modality].append(date)

                tokens_by_modality[modality].append(field_token)
                dates_by_modality[modality].append(date)

                continue

            if self.field_timevariant[field_id]:
                date = self.get_dates(patient[f'53-{instance_id}.0'])  # Estimated assessment date
            else:
                date = no_date

            if pd.api.types.is_numeric_dtype(type(field_value)):
                is_special_value = False
                if field_id in self.special_field_values:
                    for special_value in self.special_field_values[field_id]:
                        if np.isclose(special_value, field_value):
                            is_special_value = True
                            field_value = self.special_field_values[field_id][special_value]
                            field_value = field_token + '_' + self.format_name(field_value)
                            break
                if not is_special_value:
                    mean = self.field_info_df.loc[field_id, 'mean']
                    std = self.field_info_df.loc[field_id, 'std']
                    field_value = (field_value - mean) / std
            else:
                field_value = field_token + '_' + self.format_name(field_value)

            tokens_by_modality[modality].append(field_token)
            dates_by_modality[modality].append(date)

            tokens_by_modality[modality].append(field_value)
            dates_by_modality[modality].append(date)

        return tokens_by_modality, dates_by_modality

    def tokenize(self, patient: dict, add_special_tokens: bool = True, **kwargs) -> (list[str], list[str], list[str]):
        tokens_by_modality, dates_by_modality = self.tokenize_list_columns(patient, **kwargs)

        tmp_tokens, tmp_dates = self.tokenize_value_columns(patient, **kwargs)

        tokens_by_modality.update(tmp_tokens)
        dates_by_modality.update(tmp_dates)

        if add_special_tokens:
            no_date = self.get_dates(None)
            for modality in self.modalities:
                tokens_by_modality[modality] = [self.bos_token] + tokens_by_modality[modality] + [self.eos_token]
                dates_by_modality[modality] = [no_date] + dates_by_modality[modality] + [no_date]

        return tokens_by_modality, dates_by_modality

    def encode(self, patient: dict, add_special_tokens: bool = True, **kwargs) -> Dict[str, List[int]]:
        # tokenize and convert to ids
        tokens_by_modality, dates_by_modality = self.tokenize(patient,add_special_tokens = True, **kwargs)

        inputs = {}

        def get_token_ids(tokens):
            return [self.encoder.get(token, token) for token in tokens]

        for modality in self.modalities:
            inputs[f'{modality}_input_ids'] = get_token_ids(tokens_by_modality[modality])
            inputs[f'{modality}_token_type_ids'] = self.get_token_type_ids(inputs[f'{modality}_input_ids'])
            inputs[f'{modality}_measurement_mask'] = (np.array(inputs[f'{modality}_token_type_ids']) == self.numerical_measurement_token_type_id).tolist()
            inputs[f'{modality}_year_ids'] = [self.year_encoder[date[0]] for date in dates_by_modality[modality]]
            inputs[f'{modality}_month_ids'] = [self.month_encoder[date[1]] for date in dates_by_modality[modality]]
            inputs[f'{modality}_day_ids'] = [self.day_encoder[date[2]] for date in dates_by_modality[modality]]

        return inputs

    def batch_encode_plus(
        self,
        patient_list: list[dict] = None,
        add_special_tokens: bool = True,
        padding: PaddingStrategy = PaddingStrategy.LONGEST,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_attention_mask: Optional[bool] = None,
        **kwargs,
    ) -> BatchEncoding:
        # encode and pad a list of patients

        samples = [self.encode(patient, add_special_tokens=add_special_tokens) for patient in patient_list]

        return self.pad(
            samples,
            padding=padding,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors=return_tensors,
            return_attention_mask=return_attention_mask,
        )

    def __call__(
        self,
        patients: list[dict] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_attention_mask: Optional[bool] = None,
        **kwargs,
    ) -> BatchEncoding:
        # if not isinstance(patients, List):
        #     raise NotImplementedError(
        #         """Derived class __call__ might produces different
        #                                behaviour for single patient then expected, use .encode explicitly"""
        #     )

        return self.batch_encode_plus(
            patients,
            add_special_tokens=add_special_tokens,
            padding=padding,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors=return_tensors,
            return_attention_mask=return_attention_mask,
        )

    def pad(
        self,
        encoded_inputs: Union[
            BatchEncoding,
            List[BatchEncoding],
            Dict[str, EncodedInput],
            Dict[str, List[EncodedInput]],
            List[Dict[str, EncodedInput]],
        ],
        padding: Union[bool, str, PaddingStrategy] = True,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = True,
        return_tensors: Optional[Union[str, TensorType]] = None,
        verbose: bool = True,
    ) -> BatchEncoding:
        # If we have a list of dicts, let's convert it in a dict of lists
        # We do this to allow using this method as a collate_fn function in PyTorch Dataloader
        if isinstance(encoded_inputs, (list, tuple)) and isinstance(encoded_inputs[0], Mapping):
            encoded_inputs = {key: [example[key] for example in encoded_inputs] for key in encoded_inputs[0].keys()}

        # The models's main input name, usually `input_ids`, has be passed for padding
        if self.model_input_names[0] not in encoded_inputs:
            raise ValueError(
                "You should supply an encoding or a list of encodings to this method "
                f"that includes {self.model_input_names[0]}, but you provided {list(encoded_inputs.keys())}"
            )

        required_input = encoded_inputs[self.model_input_names[0]]

        if required_input is None or (isinstance(required_input, Sized) and len(required_input) == 0):
            if return_attention_mask:
                encoded_inputs["attention_mask"] = []
            return encoded_inputs

        # If we have PyTorch/TF/NumPy tensors/arrays as inputs, we cast them as python objects
        # and rebuild them afterward if no return_tensors is specified
        # Note that we lose the specific device the tensor may be on for PyTorch

        first_element = required_input[0]
        if isinstance(first_element, (list, tuple)):
            # first_element might be an empty list/tuple in some edge cases, so we grab the first non-empty element.
            for item in required_input:
                if len(item) != 0:
                    first_element = item[0]
                    break
        # At this state, if `first_element` is still a list/tuple, it's an empty one so there is nothing to do.
        if not isinstance(first_element, (int, list, tuple)):
            if is_tf_tensor(first_element):
                return_tensors = "tf" if return_tensors is None else return_tensors
            elif is_torch_tensor(first_element):
                return_tensors = "pt" if return_tensors is None else return_tensors
            elif isinstance(first_element, np.ndarray):
                return_tensors = "np" if return_tensors is None else return_tensors
            else:
                raise ValueError(
                    f"type of {first_element} unknown: {type(first_element)}. "
                    "Should be one of a python, numpy, pytorch or tensorflow object."
                )

            for key, value in encoded_inputs.items():
                encoded_inputs[key] = to_py_obj(value)

        # Convert padding_strategy in PaddingStrategy
        # padding_strategy, _, max_length, _ = self._get_padding_truncation_strategies(
        #     padding=padding, max_length=max_length, verbose=verbose
        # )
        padding_strategy = padding

        # required_input = encoded_inputs[self.model_input_names[0]]
        # if required_input and not isinstance(required_input[0], (list, tuple)):
        #     encoded_inputs = self._pad(
        #         encoded_inputs,
        #         max_length=max_length,
        #         padding_strategy=padding_strategy,
        #         pad_to_multiple_of=pad_to_multiple_of,
        #         return_attention_mask=return_attention_mask,
        #     )
        #     return BatchEncoding(encoded_inputs, tensor_type=return_tensors)

        batch_size = len(required_input)
        assert all(
            len(v) == batch_size for v in encoded_inputs.values()
        ), "Some items in the output dictionary have a different batch size than others."

        # keys should be padded if they are list of lists
        keys_to_pad = [k for k, v in encoded_inputs.items() if isinstance(v[0], (list, tuple))]

        # max_length = {k: max(len(inputs) for inputs in v) for k, v in encoded_inputs.items()}

        max_length = {k: max(len(inputs) for inputs in encoded_inputs[k]) for k in keys_to_pad}

        batch_outputs = {}
        for i in range(batch_size):
            inputs = {k: v[i] for k, v in encoded_inputs.items()}
            outputs = self._pad(
                inputs,
                max_length=max_length,
                padding_strategy=padding_strategy,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=return_attention_mask,
            )

            for key, value in outputs.items():
                if key not in batch_outputs:
                    batch_outputs[key] = []
                batch_outputs[key].append(value)

        return BatchEncoding(batch_outputs, tensor_type=return_tensors)

    def _pad(
        self,
        encoded_inputs: Union[Dict[str, EncodedInput], BatchEncoding],
        max_length: dict[str, int] = None,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = True,
    ) -> dict:

        # required_input = encoded_inputs[self.model_input_names[0]]

        def pad_to_multiple(length: int) -> int:
            if pad_to_multiple_of is not None and (length % pad_to_multiple_of != 0):
                return ((length // pad_to_multiple_of) + 1) * pad_to_multiple_of
            else:
                return length

        max_length = {k: pad_to_multiple(v) for k, v in max_length.items()}

        needs_to_be_padded = {
            k: (padding_strategy != PaddingStrategy.DO_NOT_PAD and len(encoded_inputs[k]) != max_length[k])
            for k in max_length
        }
        # Initialize attention mask if not present.
        if return_attention_mask:
            for modality in self.modalities:
                encoded_inputs[f'{modality}_attention_mask'] = [1] * len(encoded_inputs[f'{modality}_input_ids'])
                max_length[f'{modality}_attention_mask'] = max_length[f'{modality}_input_ids']
                needs_to_be_padded[f'{modality}_attention_mask'] = True

        for k in max_length:
            if needs_to_be_padded[k]:
                difference = max_length[k] - len(encoded_inputs[k])

                if self.padding_side == "right":
                    encoded_inputs[k] = encoded_inputs[k] + [0] * difference

                elif self.padding_side == "left":
                    encoded_inputs[k] = [0] * difference + encoded_inputs[k]

                else:
                    raise ValueError("Invalid padding strategy:" + str(self.padding_side))

        return encoded_inputs

    def batch_decode(
        self,
        sequences: dict[str, list[list[int]]] | dict[str, list[int]] | list[list[int]] | list[int],
        skip_special_tokens: bool = False,
        **kwargs,
    ) -> list[dict[str, list[str]]] | dict[str, list[str]] | list[list[str]] | list[str]:
        # dict[str, list[list[int]]] # decode input batch
        # dict[str, list[int]] # decode single input
        # list[list[int]] # decode output batch
        # list[int] # decode single output

        sequences = to_py_obj(sequences)

        if isinstance(sequences, dict):
            # decode input
            if isinstance(sequences["disease_input_ids"][0], list):
                # decode input batch
                batch_size = len(sequences["disease_input_ids"])
                return [
                    self._decode(
                        {key: v[i] for key, v in sequences.items()}, skip_special_tokens=skip_special_tokens, **kwargs
                    )
                    for i in range(batch_size)
                ]
            else:
                # decode single input
                return self._decode(sequences, skip_special_tokens=skip_special_tokens, **kwargs)
        elif isinstance(sequences, list):
            # decode output
            if isinstance(sequences[0], list):
                # decode output batch
                return [self._decode(seq, skip_special_tokens=skip_special_tokens, **kwargs) for seq in sequences]
            else:
                # decode single output
                return self._decode(sequences, skip_special_tokens=skip_special_tokens, **kwargs)
        else:
            raise ValueError("Invalid input type for batch_decode")

    def _decode(
        self, token_ids: list[int] | dict[str, list[int]], skip_special_tokens: bool = False, **kwargs
    ) -> list[str] | dict[str, list[str]]:
        # dict[str, list[int]] # decode single input
        # list[int] # decode single output

        if isinstance(token_ids, dict):
            # decode input
            # keys:
            #   input_ids: with seq decoder
            #   token_type_ids: with seq type decoder
            #   year_ids, month_ids, day_ids: with respective decoders and '-'.join to 'date'
            #   context_input_ids: with context decoder
            #   context_token_type_ids: with context type decoder

            decoded_input = {
                "disease_input_ids": [],
                "disease_token_type_ids": [],
                "disease_date": [],
                "drug_input_ids": [],
                "drug_token_type_ids": [],
                "drug_date": [],
            }

            if self.use_context:
                decoded_input["context_input_ids"] = []
                decoded_input["context_token_type_ids"] = []
                decoded_input["context_visit_ids"] = []

            if self.use_lab:
                decoded_input["lab_input_ids"] = []
                decoded_input["lab_token_type_ids"] = []

            if self.use_geno:
                decoded_input["geno_input_ids"] = []
                decoded_input["geno_token_type_ids"] = []

            for token_id, token_type_id, year, month, day in zip(
                token_ids["disease_input_ids"],
                token_ids["disease_token_type_ids"],
                token_ids["disease_year_ids"],
                token_ids["disease_month_ids"],
                token_ids["disease_day_ids"],
            ):
                if not skip_special_tokens or token_type_id != self.special_token_type_id:
                    decoded_input["disease_input_ids"].append(self.decoder[token_id])
                    decoded_input["disease_token_type_ids"].append(self.type_decoder[token_type_id])
                    full_date = f"{self.year_decoder[year]}-{self.month_decoder[month]}-{self.day_decoder[day]}"
                    decoded_input["disease_date"].append(full_date)

            for token_id, token_type_id, year, month, day in zip(
                token_ids["drug_input_ids"],
                token_ids["drug_token_type_ids"],
                token_ids["drug_year_ids"],
                token_ids["drug_month_ids"],
                token_ids["drug_day_ids"],
            ):
                if not skip_special_tokens or token_type_id != self.special_token_type_id:
                    decoded_input["drug_input_ids"].append(self.decoder[token_id])
                    decoded_input["drug_token_type_ids"].append(self.type_decoder[token_type_id])
                    full_date = f"{self.year_decoder[year]}-{self.month_decoder[month]}-{self.day_decoder[day]}"
                    decoded_input["drug_date"].append(full_date)

            if self.use_context:
                for token_id, token_type_id in zip(token_ids["context_input_ids"], token_ids["context_token_type_ids"]):
                    if not skip_special_tokens or token_type_id != self.special_token_type_id:
                        if token_type_id == self.numerical_measurement_token_type_id:
                            decoded_input["context_input_ids"].append(str(token_id))
                        else:
                            decoded_input["context_input_ids"].append(self.decoder[token_id])
                        decoded_input["context_token_type_ids"].append(self.type_decoder[token_type_id])
                decoded_input["context_visit_ids"] = token_ids["context_visit_ids"]

            if self.use_lab:
                for token_id, token_type_id in zip(token_ids["lab_input_ids"], token_ids["lab_token_type_ids"]):
                    if not skip_special_tokens or token_type_id != self.special_token_type_id:
                        if token_type_id == self.numerical_measurement_token_type_id:
                            decoded_input["lab_input_ids"].append(str(token_id))
                        else:
                            decoded_input["lab_input_ids"].append(self.decoder[token_id])
                        decoded_input["lab_token_type_ids"].append(self.type_decoder[token_type_id])

            if self.use_geno:
                for token_id, token_type_id in zip(token_ids["geno_input_ids"], token_ids["geno_token_type_ids"]):
                    if not skip_special_tokens or token_type_id != self.special_token_type_id:
                        decoded_input["geno_input_ids"].append(self.decoder[token_id])
                        decoded_input["geno_token_type_ids"].append(self.type_decoder[token_type_id])

            return decoded_input

        elif isinstance(token_ids, list):
            # decode output
            token_type_ids = self.get_token_type_ids(token_ids)
            return [
                self.decoder[token_id]
                for token_id, token_type_id in zip(token_ids, token_type_ids)
                if not skip_special_tokens or token_type_id != self.special_token_type_id
            ]
        else:
            raise ValueError("Invalid input type for _decode")

    def to_human_readable(
        self, sequence: dict[str, list[str]] | list[dict[str, list[str]]] | list[str] | list[list[str]]
    ) -> dict[str, list[str]] | list[dict[str, list[str]]] | list[str] | list[list[str]]:
        # dict[str, list[str]] # decode single input
        # list[dict[str, list[str]]] # decode input batch
        # list[str] # decode single output
        # list[list[str]] # decode output batch

        def decode_seq_to_human_readable(seq_tokens: list[str], seq_types: Optional[list[str]] = None) -> list[str]:
            def decode_seq_token_to_human_readable(token: str, token_type: str) -> str:
                if token_type == self.disease_token_type:
                    return self.icd_handler.decode_token(token)
                elif token_type == self.treatment_token_type:
                    return self.atc_handler.decode_token(token)
                else:
                    return token

            if seq_types is None:
                seq_types = [self.seq_token_to_type[token] for token in seq_tokens]

            return [
                decode_seq_token_to_human_readable(token_id, token_type)
                for token_id, token_type in zip(seq_tokens, seq_types)
            ]

        def decode_context_to_human_readable(context_tokens: list[str], context_types: list[str]) -> list[str]:
            def decode_context_token_to_human_readable(token: str, token_type: str) -> str:
                if token_type == self.uid_token_type:
                    return self.context_uid_names[token]
                elif token_type == self.categorical_measurement_token_type:
                    return self.context_categorical_names.loc[tuple(token.split("_"))].Meaning
                else:
                    return token

            return [
                decode_context_token_to_human_readable(token_id, token_type)
                for token_id, token_type in zip(context_tokens, context_types)
            ]

        def decode_lab_to_human_readable(lab_tokens: list[str], lab_types: list[str]) -> list[str]:
            def decode_lab_token_to_human_readable(token: str, token_type: str) -> str:
                if token_type == self.uid_token_type:
                    return self.lab_uid_names[token]
                else:
                    return token

            return [
                decode_lab_token_to_human_readable(token_id, token_type)
                for token_id, token_type in zip(lab_tokens, lab_types)
            ]

        def decode_input_to_human_readable(input_dict: dict[str, list[str]]) -> dict[str, list[str]]:
            decoded_input = {
                "disease_input_ids": decode_seq_to_human_readable(
                    input_dict["disease_input_ids"], input_dict["disease_token_type_ids"]
                ),
                "disease_token_type_ids": input_dict["disease_token_type_ids"],
                "disease_date": input_dict["disease_date"],
                "drug_input_ids": decode_seq_to_human_readable(
                    input_dict["drug_input_ids"], input_dict["drug_token_type_ids"]
                ),
                "drug_token_type_ids": input_dict["drug_token_type_ids"],
                "drug_date": input_dict["drug_date"],
            }

            if self.use_context:
                decoded_input["context_input_ids"] = decode_context_to_human_readable(
                    input_dict["context_input_ids"], input_dict["context_token_type_ids"]
                )
                decoded_input["context_token_type_ids"] = input_dict["context_token_type_ids"]
                decoded_input["context_visit_ids"] = input_dict["context_visit_ids"]

            if self.use_lab:
                decoded_input["lab_input_ids"] = decode_lab_to_human_readable(
                    input_dict["lab_input_ids"], input_dict["lab_token_type_ids"]
                )
                decoded_input["lab_token_type_ids"] = input_dict["lab_token_type_ids"]

            if self.use_geno:
                decoded_input["geno_input_ids"] = input_dict["geno_input_ids"]
                decoded_input["geno_token_type_ids"] = input_dict["geno_token_type_ids"]

            return decoded_input

        if isinstance(sequence, dict):
            if isinstance(sequence["disease_input_ids"][0], list):
                # decode input batch
                return [decode_input_to_human_readable(input_dict) for input_dict in sequence]
            else:
                # decode single input
                return decode_input_to_human_readable(sequence)

        elif isinstance(sequence, list):
            if isinstance(sequence[0], list):
                # decode output batch
                return [decode_seq_to_human_readable(seq) for seq in sequence]
            else:
                # decode single output
                return decode_seq_to_human_readable(sequence)
        else:
            raise ValueError("Invalid input type for to_human_readable")

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(name_or_path='{self.name_or_path}',"
            f" vocab_size={self.vocab_size}, num_disease_tokens={self.icd_handler.vocab_size()},"
            f" num_drug_tokens={self.atc_handler.vocab_size()}, special_tokens={self.special_tokens_map_extended}"
            f" use_context={self.use_context}, use_lab={self.use_lab})"
            f" num_context_tokens={len(self.context_encoder) if self.use_context else 0},"
            f" num_lab_tokens={len(self.lab_encoder) if self.use_lab else 0})"
            f" num_geno_tokens={len(self.geno_encoder) if self.use_geno else 0})"
        )

    def __len__(self) -> int:
        return len(self.encoder)

    def vocab_size(self) -> int:
        return len(self.encoder)
