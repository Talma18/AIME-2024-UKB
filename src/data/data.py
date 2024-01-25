import datasets
from datasets import DatasetDict, Dataset, config


def load_dataset(path, tokenizer, mini=False, num_proc=32) -> DatasetDict:
    dataset = datasets.load_from_disk(path)

    if mini:
        dataset = dataset.select(range(10000))

    dataset = dataset.map(
        lambda examples: tokenizer.encode(examples, padding=False), remove_columns=list(dataset.features.keys())[:-1],
        num_proc=num_proc,
    )

    # # Filter out too long sequences because of memory issues
    # dataset = dataset.filter(lambda x: (len(x["drug_input_ids"]) < 256) and (len(x["lab_input_ids"]) < 660), num_proc=num_proc)

    return DatasetDict({'train': dataset, 'test': dataset})
