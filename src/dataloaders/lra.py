"""Long Range Arena datasets"""
import io
import logging
import os
import pickle
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
import torchtext
import torchvision
import numpy as np
from einops.layers.torch import Rearrange, Reduce
from PIL import Image  # Only used for Pathfinder
from datasets import DatasetDict, Value, load_dataset

from functools import partial
from einops import rearrange

from src.dataloaders.base import default_data_path, SequenceDataset, ImageResolutionSequenceDataset

# import masking utils
from src.dataloaders.base import MaskedSequenceDatasetWrapper
from src.models.functional import masking


AAN_CACHE_DIR = '/home/gamir/idoamos/projects/state-spaces/data/.cache/huggingface/datasets/'


class IMDB(SequenceDataset):
    _name_ = "imdb"
    d_output = 2
    l_output = 0

    @property
    def init_defaults(self):
        return {
            "l_max": 4096,
            "level": "char",
            "min_freq": 15,
            "seed": 42,
            "val_split": 0.0,
            "append_bos": False,
            "append_eos": True,
            # 'max_vocab': 135,
            "n_workers": 4,  # Only used for tokenizing dataset before caching
        }

    @property
    def n_tokens(self):
        return len(self.vocab)

    def prepare_data(self):
        if self.cache_dir is None:  # Just download the dataset
            load_dataset(self._name_, cache_dir=self.data_dir)
        else:  # Process the dataset and save it
            self.process_dataset()

    def setup(self, stage=None):
        """If cache_dir is not None, we'll cache the processed dataset there."""
        self.data_dir = self.data_dir or default_data_path / self._name_
        self.cache_dir = self.data_dir / "cache"
        assert self.level in [
            "word",
            "char",
        ], f"level {self.level} not supported"

        if stage == "test" and hasattr(self, "dataset_test"):
            return
        dataset, self.tokenizer, self.vocab = self.process_dataset()
        print(
            f"IMDB {self.level} level | min_freq {self.min_freq} | vocab size {len(self.vocab)}"
        )
        dataset.set_format(type="torch", columns=["input_ids", "label"])

        # Create all splits
        dataset_train, self.dataset_test = dataset["train"], dataset["test"]
        if self.val_split == 0.0:
            # Use test set as val set, as done in the LRA paper
            self.dataset_train, self.dataset_val = dataset_train, None
        else:
            train_val = dataset_train.train_test_split(
                test_size=self.val_split, seed=self.seed
            )
            self.dataset_train, self.dataset_val = (
                train_val["train"],
                train_val["test"],
            )

    def _collate_fn(self, batch):
        xs, ys = zip(*[(data["input_ids"], data["label"]) for data in batch])
        lengths = torch.tensor([len(x) for x in xs])
        xs = nn.utils.rnn.pad_sequence(
            xs, padding_value=self.vocab["<pad>"], batch_first=True
        )
        ys = torch.tensor(ys)
        return xs, ys, {"lengths": lengths}

        # self._collate_fn = collate_batch

    def process_dataset(self):
        cache_dir = (
            None if self.cache_dir is None else self.cache_dir / self._cache_dir_name
        )
        if cache_dir is not None:
            if cache_dir.is_dir():
                return self._load_from_cache(cache_dir)

        dataset = load_dataset(self._name_, cache_dir=self.data_dir)
        dataset = DatasetDict(train=dataset["train"], test=dataset["test"])
        if self.level == "word":
            tokenizer = torchtext.data.utils.get_tokenizer(
                "spacy", language="en_core_web_sm"
            )
        else:  # self.level == 'char'
            tokenizer = list  # Just convert a string to a list of chars
        # Account for <bos> and <eos> tokens
        l_max = self.l_max - int(self.append_bos) - int(self.append_eos)
        tokenize = lambda example: {"tokens": tokenizer(example["text"])[:l_max]}
        dataset = dataset.map(
            tokenize,
            remove_columns=["text"],
            keep_in_memory=True,
            load_from_cache_file=False,
            num_proc=max(self.n_workers, 1),
        )
        vocab = torchtext.vocab.build_vocab_from_iterator(
            dataset["train"]["tokens"],
            min_freq=self.min_freq,
            specials=(
                    ["<pad>", "<unk>"]
                    + (["<bos>"] if self.append_bos else [])
                    + (["<eos>"] if self.append_eos else [])
            ),
        )
        vocab.set_default_index(vocab["<unk>"])

        numericalize = lambda example: {
            "input_ids": vocab(
                (["<bos>"] if self.append_bos else [])
                + example["tokens"]
                + (["<eos>"] if self.append_eos else [])
            )
        }
        dataset = dataset.map(
            numericalize,
            remove_columns=["tokens"],
            keep_in_memory=True,
            load_from_cache_file=False,
            num_proc=max(self.n_workers, 1),
        )

        if cache_dir is not None:
            self._save_to_cache(dataset, tokenizer, vocab, cache_dir)
        return dataset, tokenizer, vocab

    def _save_to_cache(self, dataset, tokenizer, vocab, cache_dir):
        cache_dir = self.cache_dir / self._cache_dir_name
        logger = logging.getLogger(__name__)
        logger.info(f"Saving to cache at {str(cache_dir)}")
        dataset.save_to_disk(str(cache_dir))
        with open(cache_dir / "tokenizer.pkl", "wb") as f:
            pickle.dump(tokenizer, f)
        with open(cache_dir / "vocab.pkl", "wb") as f:
            pickle.dump(vocab, f)

    def _load_from_cache(self, cache_dir):
        assert cache_dir.is_dir()
        logger = logging.getLogger(__name__)
        logger.info(f"Load from cache at {str(cache_dir)}")
        dataset = DatasetDict.load_from_disk(str(cache_dir))
        with open(cache_dir / "tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)
        with open(cache_dir / "vocab.pkl", "rb") as f:
            vocab = pickle.load(f)
        return dataset, tokenizer, vocab

    @property
    def _cache_dir_name(self):
        return f"l_max-{self.l_max}-level-{self.level}-min_freq-{self.min_freq}-append_bos-{self.append_bos}-append_eos-{self.append_eos}"


class TabularDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            path,
            format,
            col_idx=None,
            skip_header=False,
            csv_reader_params=None,
    ):
        """
        col_idx: the indices of the columns.
        """
        if csv_reader_params is None:
            csv_reader_params = {}
        format = format.lower()
        assert format in ["tsv", "csv"]
        with io.open(os.path.expanduser(path), encoding="utf8") as f:
            if format == "csv":
                reader = torchtext.utils.unicode_csv_reader(f, **csv_reader_params)
            elif format == "tsv":
                reader = torchtext.utils.unicode_csv_reader(
                    f, delimiter="\t", **csv_reader_params
                )
            else:
                reader = f
            if skip_header:
                next(reader)
            self._data = [
                line if col_idx is None else [line[c] for c in col_idx]
                for line in reader
            ]

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]


# LRA tokenizer renames ']' to 'X' and delete parentheses as their tokenizer removes
# non-alphanumeric characters.
# https://github.com/google-research/long-range-arena/blob/264227cbf9591e39dd596d2dc935297a2070bdfe/lra_benchmarks/listops/input_pipeline.py#L46
def listops_tokenizer(s):
    return s.translate({ord("]"): ord("X"), ord("("): None, ord(")"): None}).split()


class ListOps(SequenceDataset):
    _name_ = "listops"
    d_output = 10
    l_output = 0

    @property
    def init_defaults(self):
        return {
            "l_max": 2048,
            "append_bos": False,
            "append_eos": True,
            # 'max_vocab': 20, # Actual size 18
            "n_workers": 4,  # Only used for tokenizing dataset
        }

    @property
    def n_tokens(self):
        return len(self.vocab)

    @property
    def _cache_dir_name(self):
        return f"l_max-{self.l_max}-append_bos-{self.append_bos}-append_eos-{self.append_eos}"

    def init(self):
        if self.data_dir is None:
            self.data_dir = default_data_path / self._name_
        self.cache_dir = self.data_dir / self._cache_dir_name

    def prepare_data(self):
        if self.cache_dir is None:
            for split in ["train", "val", "test"]:
                split_path = self.data_dir / f"basic_{split}.tsv"
                if not split_path.is_file():
                    raise FileNotFoundError(
                        f"""
                    File {str(split_path)} not found.
                    To get the dataset, download lra_release.gz from
                    https://github.com/google-research/long-range-arena,
                    then unzip it with tar -xvf lra_release.gz.
                    Then point data_dir to the listops-1000 directory.
                    """
                    )
        else:  # Process the dataset and save it
            self.process_dataset()

    def setup(self, stage=None):
        if stage == "test" and hasattr(self, "dataset_test"):
            return
        dataset, self.tokenizer, self.vocab = self.process_dataset()
        self.vocab_size = len(self.vocab)
        dataset.set_format(type="torch", columns=["input_ids", "Target"])
        self.dataset_train, self.dataset_val, self.dataset_test = (
            dataset["train"],
            dataset["val"],
            dataset["test"],
        )

        def collate_batch(batch):
            xs, ys = zip(*[(data["input_ids"], data["Target"]) for data in batch])
            lengths = torch.tensor([len(x) for x in xs])
            xs = nn.utils.rnn.pad_sequence(
                xs, padding_value=self.vocab["<pad>"], batch_first=True
            )
            ys = torch.tensor(ys)
            return xs, ys, {"lengths": lengths}

        self._collate_fn = collate_batch

    def process_dataset(self):
        cache_dir = (
            None if self.cache_dir is None else self.cache_dir / self._cache_dir_name
        )
        if cache_dir is not None:
            if cache_dir.is_dir():
                return self._load_from_cache(cache_dir)

        dataset = load_dataset(
            "csv",
            data_files={
                "train": str(self.data_dir / "basic_train.tsv"),
                "val": str(self.data_dir / "basic_val.tsv"),
                "test": str(self.data_dir / "basic_test.tsv"),
            },
            delimiter="\t",
            keep_in_memory=True,
        )

        tokenizer = listops_tokenizer

        # Account for <bos> and <eos> tokens
        l_max = self.l_max - int(self.append_bos) - int(self.append_eos)
        tokenize = lambda example: {"tokens": tokenizer(example["Source"])[:l_max]}
        dataset = dataset.map(
            tokenize,
            remove_columns=["Source"],
            keep_in_memory=True,
            load_from_cache_file=False,
            num_proc=max(self.n_workers, 1),
        )
        vocab = torchtext.vocab.build_vocab_from_iterator(
            dataset["train"]["tokens"],
            specials=(
                    ["<pad>", "<unk>"]
                    + (["<bos>"] if self.append_bos else [])
                    + (["<eos>"] if self.append_eos else [])
            ),
        )
        vocab.set_default_index(vocab["<unk>"])

        numericalize = lambda example: {
            "input_ids": vocab(
                (["<bos>"] if self.append_bos else [])
                + example["tokens"]
                + (["<eos>"] if self.append_eos else [])
            )
        }
        dataset = dataset.map(
            numericalize,
            remove_columns=["tokens"],
            keep_in_memory=True,
            load_from_cache_file=False,
            num_proc=max(self.n_workers, 1),
        )

        if cache_dir is not None:
            self._save_to_cache(dataset, tokenizer, vocab, cache_dir)
        return dataset, tokenizer, vocab

    def _save_to_cache(self, dataset, tokenizer, vocab, cache_dir):
        cache_dir = self.cache_dir / self._cache_dir_name
        logger = logging.getLogger(__name__)
        logger.info(f"Saving to cache at {str(cache_dir)}")
        dataset.save_to_disk(str(cache_dir))
        with open(cache_dir / "tokenizer.pkl", "wb") as f:
            pickle.dump(tokenizer, f)
        with open(cache_dir / "vocab.pkl", "wb") as f:
            pickle.dump(vocab, f)

    def _load_from_cache(self, cache_dir):
        assert cache_dir.is_dir()
        logger = logging.getLogger(__name__)
        logger.info(f"Load from cache at {str(cache_dir)}")
        dataset = DatasetDict.load_from_disk(str(cache_dir))
        with open(cache_dir / "tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)
        with open(cache_dir / "vocab.pkl", "rb") as f:
            vocab = pickle.load(f)
        return dataset, tokenizer, vocab


class PathFinderDataset(torch.utils.data.Dataset):
    """Path Finder dataset."""

    # There's an empty file in the dataset
    blacklist = {"pathfinder32/curv_baseline/imgs/0/sample_172.png"}

    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_dir = Path(data_dir).expanduser()
        assert self.data_dir.is_dir(), f"data_dir {str(self.data_dir)} does not exist"
        self.transform = transform
        samples = []
        # for diff_level in ['curv_baseline', 'curv_contour_length_9', 'curv_contour_length_14']:
        for diff_level in ["curv_contour_length_14"]:
            path_list = sorted(
                list((self.data_dir / diff_level / "metadata").glob("*.npy")),
                key=lambda path: int(path.stem),
            )
            assert path_list, "No metadata found"
            for metadata_file in path_list:
                with open(metadata_file, "r") as f:
                    for metadata in f.read().splitlines():
                        metadata = metadata.split()
                        image_path = Path(diff_level) / metadata[0] / metadata[1]
                        if (
                                str(Path(self.data_dir.stem) / image_path)
                                not in self.blacklist
                        ):
                            label = int(metadata[3])
                            samples.append((image_path, label))
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        # https://github.com/pytorch/vision/blob/9b29f3f22783112406d9c1a6db47165a297c3942/torchvision/datasets/folder.py#L247
        with open(self.data_dir / path, "rb") as f:
            sample = Image.open(f).convert("L")  # Open in grayscale
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target


class PathFinder(ImageResolutionSequenceDataset):
    _name_ = "pathfinder"
    d_input = 1
    d_output = 2
    l_output = 0  # equivalent to length 1 and then squeezing

    @property
    def n_tokens(self):
        if self.tokenize:
            return 256

    @property
    def init_defaults(self):
        return {
            "resolution": 32,
            "sequential": True,
            "tokenize": False,
            "pool": 1,
            "val_split": 0.1,
            "test_split": 0.1,
            "seed": 42,  # Controls the train/val/test split
        }

    def default_transforms(self):
        transform_list = [torchvision.transforms.ToTensor()]
        if self.pool > 1:
            transform_list.append(
                Reduce(
                    "1 (h h2) (w w2) -> 1 h w",
                    "mean",
                    h2=self.pool,
                    w2=self.pool,
                )
            )
        if self.tokenize:
            transform_list.append(
                torchvision.transforms.Lambda(lambda x: (x * 255).long())
            )
        else:
            transform_list.append(torchvision.transforms.Normalize(mean=0.5, std=0.5))
        if self.sequential:
            # If tokenize, it makes more sense to get rid of the channel dimension
            transform_list.append(
                Rearrange("1 h w -> (h w)")
                if self.tokenize
                else Rearrange("1 h w -> (h w) 1")
            )
        else:
            transform_list.append(Rearrange("1 h w -> h w 1"))
        return torchvision.transforms.Compose(transform_list)

    def prepare_data(self):
        if not self.data_dir.is_dir():
            raise FileNotFoundError(
                f"""
            Directory {str(self.data_dir)} not found.
            To get the dataset, download lra_release.gz from
            https://github.com/google-research/long-range-arena,
            then unzip it with tar -xvf lra_release.gz.
            Then point data_dir to the pathfinderX directory, where X is either 32, 64, 128, or 256.
            """
            )

    def setup(self, stage=None):
        if self.data_dir is None:
            self.data_dir = (
                    default_data_path / self._name_ / f"pathfinder{self.resolution}"
            )

        if stage == "test" and hasattr(self, "dataset_test"):
            return
        # [2021-08-18] TD: I ran into RuntimeError: Too many open files.
        # https://github.com/pytorch/pytorch/issues/11201
        torch.multiprocessing.set_sharing_strategy("file_system")
        dataset = PathFinderDataset(self.data_dir, transform=self.default_transforms())
        len_dataset = len(dataset)
        val_len = int(self.val_split * len_dataset)
        test_len = int(self.test_split * len_dataset)
        train_len = len_dataset - val_len - test_len
        (
            self.dataset_train,
            self.dataset_val,
            self.dataset_test,
        ) = torch.utils.data.random_split(
            dataset,
            [train_len, val_len, test_len],
            generator=torch.Generator().manual_seed(self.seed),
        )


class AAN(SequenceDataset):
    _name_ = "aan"
    d_output = 2  # Use accuracy instead of binary_accuracy
    l_output = 0

    @property
    def n_tokens(self):
        return len(self.vocab)

    @property
    def init_defaults(self):
        return {
            "l_max": 4000,
            # 'max_vocab': 100, # Full size 98
            "append_bos": False,
            "append_eos": True,
            "n_workers": 4,  # For tokenizing only
        }

    @property
    def _cache_dir_name(self):
        return f"l_max-{self.l_max}-append_bos-{self.append_bos}-append_eos-{self.append_eos}"

    def init(self):
        if self.data_dir is None:
            self.data_dir = default_data_path / self._name_
        self.cache_dir = self.data_dir / self._cache_dir_name

    def prepare_data(self):
        if self.cache_dir is None:
            for split in ["train", "eval", "test"]:
                split_path = self.data_dir / f"new_aan_pairs.{split}.tsv"
                if not split_path.is_file():
                    raise FileNotFoundError(
                        f"""
                    File {str(split_path)} not found.
                    To get the dataset, download lra_release.gz from
                    https://github.com/google-research/long-range-arena,
                    then unzip it with tar -xvf lra_release.gz.
                    Then point data_dir to the tsv_data directory.
                    """
                    )
        else:  # Process the dataset and save it
            self.process_dataset()

    def setup(self, stage=None):
        if stage == "test" and hasattr(self, "dataset_test"):
            return

        # [2021-08-18] TD: I ran into RuntimeError: Too many open files.
        # https://github.com/pytorch/pytorch/issues/11201
        torch.multiprocessing.set_sharing_strategy("file_system")

        dataset, self.tokenizer, self.vocab = self.process_dataset()
        # self.vocab_size = len(self.vocab)
        print("AAN vocab size:", len(self.vocab))

        dataset.set_format(type="torch", columns=["input_ids1", "input_ids2", "label"])
        self.dataset_train, self.dataset_val, self.dataset_test = (
            dataset["train"],
            dataset["val"],
            dataset["test"],
        )

        def collate_batch(batch):
            xs1, xs2, ys = zip(
                *[
                    (data["input_ids1"], data["input_ids2"], data["label"])
                    for data in batch
                ]
            )
            lengths1 = torch.tensor([len(x) for x in xs1])
            lengths2 = torch.tensor([len(x) for x in xs2])
            xs1 = nn.utils.rnn.pad_sequence(
                xs1, padding_value=self.vocab["<pad>"], batch_first=True
            )
            xs2 = nn.utils.rnn.pad_sequence(
                xs2, padding_value=self.vocab["<pad>"], batch_first=True
            )
            # Pad both to same length
            # Shape (batch, length)
            L = max(xs1.size(1), xs2.size(1))
            xs1 = F.pad(xs1, (0, L - xs1.size(1)), value=self.vocab["<pad>"])
            xs2 = F.pad(xs2, (0, L - xs2.size(1)), value=self.vocab["<pad>"])
            ys = torch.tensor(ys)
            # return xs1, xs2, ys, lengths1, lengths2

            # Concatenate two batches
            xs = torch.cat([xs1, xs2], dim=0)
            lengths = torch.cat([lengths1, lengths2], dim=0)
            return xs, ys, {"lengths": lengths}

        self._collate_fn = collate_batch

    def process_dataset(self):
        cache_dir = (
            None if self.cache_dir is None else self.cache_dir / self._cache_dir_name
        )
        if cache_dir is not None:
            if cache_dir.is_dir():
                return self._load_from_cache(cache_dir)

        dataset = load_dataset(
            "csv",
            data_files={
                "train": str(self.data_dir / "new_aan_pairs.train.tsv"),
                "val": str(self.data_dir / "new_aan_pairs.eval.tsv"),
                "test": str(self.data_dir / "new_aan_pairs.test.tsv"),
            },
            delimiter="\t",
            column_names=["label", "input1_id", "input2_id", "text1", "text2"],
            keep_in_memory=True,
        )  # True)
        dataset = dataset.remove_columns(["input1_id", "input2_id"])
        new_features = dataset["train"].features.copy()
        new_features["label"] = Value("int32")
        dataset = dataset.cast(new_features)

        tokenizer = list  # Just convert a string to a list of chars
        # Account for <bos> and <eos> tokens
        l_max = self.l_max - int(self.append_bos) - int(self.append_eos)
        tokenize = lambda example: {
            "tokens1": tokenizer(example["text1"])[:l_max],
            "tokens2": tokenizer(example["text2"])[:l_max],
        }
        dataset = dataset.map(
            tokenize,
            remove_columns=["text1", "text2"],
            keep_in_memory=True,
            load_from_cache_file=False,
            num_proc=max(self.n_workers, 1),
        )
        vocab = torchtext.vocab.build_vocab_from_iterator(
            dataset["train"]["tokens1"] + dataset["train"]["tokens2"],
            specials=(
                    ["<pad>", "<unk>"]
                    + (["<bos>"] if self.append_bos else [])
                    + (["<eos>"] if self.append_eos else [])
            ),
        )
        vocab.set_default_index(vocab["<unk>"])

        encode = lambda text: vocab(
            (["<bos>"] if self.append_bos else [])
            + text
            + (["<eos>"] if self.append_eos else [])
        )
        numericalize = lambda example: {
            "input_ids1": encode(example["tokens1"]),
            "input_ids2": encode(example["tokens2"]),
        }
        dataset = dataset.map(
            numericalize,
            remove_columns=["tokens1", "tokens2"],
            keep_in_memory=True,
            load_from_cache_file=False,
            num_proc=max(self.n_workers, 1),
        )

        if cache_dir is not None:
            self._save_to_cache(dataset, tokenizer, vocab, cache_dir)
        return dataset, tokenizer, vocab

    def _save_to_cache(self, dataset, tokenizer, vocab, cache_dir):
        cache_dir = self.cache_dir / self._cache_dir_name
        logger = logging.getLogger(__name__)
        logger.info(f"Saving to cache at {str(cache_dir)}")
        dataset.save_to_disk(str(cache_dir))
        with open(cache_dir / "tokenizer.pkl", "wb") as f:
            pickle.dump(tokenizer, f)
        with open(cache_dir / "vocab.pkl", "wb") as f:
            pickle.dump(vocab, f)

    def _load_from_cache(self, cache_dir):
        assert cache_dir.is_dir()
        logger = logging.getLogger(__name__)
        logger.info(f"Load from cache at {str(cache_dir)}")
        dataset = DatasetDict.load_from_disk(str(cache_dir))
        with open(cache_dir / "tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)
        with open(cache_dir / "vocab.pkl", "rb") as f:
            vocab = pickle.load(f)
        return dataset, tokenizer, vocab


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class PathFinderLMDataset(torch.utils.data.Dataset):
    """Path Finder dataset."""

    # There's an empty file in the dataset
    blacklist = {"pathfinder32/curv_baseline/imgs/0/sample_172.png"}

    def __init__(self, data_dir, lm_args):
        """
        Args:
            data_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_dir = Path(data_dir).expanduser()
        assert self.data_dir.is_dir(), f"data_dir {str(self.data_dir)} does not exist"
        self.lm_args = lm_args
        samples = []
        # for diff_level in ['curv_baseline', 'curv_contour_length_9', 'curv_contour_length_14']:
        for diff_level in ["curv_contour_length_14"]:
            path_list = sorted(
                list((self.data_dir / diff_level / "metadata").glob("*.npy")),
                key=lambda path: int(path.stem),
            )
            assert path_list, "No metadata found"
            for metadata_file in path_list:
                with open(metadata_file, "r") as f:
                    for metadata in f.read().splitlines():
                        metadata = metadata.split()
                        image_path = Path(diff_level) / metadata[0] / metadata[1]
                        if (
                                str(Path(self.data_dir.stem) / image_path)
                                not in self.blacklist
                        ):
                            label = int(metadata[3])
                            samples.append((image_path, label))
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        # https://github.com/pytorch/vision/blob/9b29f3f22783112406d9c1a6db47165a297c3942/torchvision/datasets/folder.py#L247

        lm_args = self.lm_args
        mlm_prob, causal_lm, lm_loss, sequential, normalize = lm_args.mlm_prob, lm_args.causal_lm, lm_args.lm_loss, lm_args.sequential, lm_args.normalize
        lm_ignore_val, span_masking, span_length = lm_args.ignore_val, lm_args.span_masking, lm_args.span_length

        with open(self.data_dir / path, "rb") as f:
            sample = Image.open(f).convert("L")  # Open in grayscale

        sample = torchvision.transforms.ToTensor()(sample)  # [1 h w]
        sample_classes = (sample * 255).long()  # for ce loss
        if normalize:
            sample = torchvision.transforms.Normalize(mean=0.5, std=0.5)(sample)
        sample, sample_classes = map(Rearrange("r h w -> (h w) r"), (sample, sample_classes))

        assert sequential
        assert sample.ndim == 2 and sample.shape[-1] == 1  # [L 1]

        if mlm_prob:
            # if to apply random masking or in contiguous spans
            if span_masking:
                mask = masking.span_mask(x=sample, masking_ratio=mlm_prob, span_length=span_length)  # [L 1]
            else:
                mask = (torch.rand_like(sample[..., :1]) < mlm_prob).float()  # [L 1]

            masked_sample = torch.cat((sample * (1 - mask), mask), dim=-1)  # [L 2]  # 0 for unmasked, 1 for masked

            if lm_loss == 'l1':
                target = sample.clone()  # this was used by current version
                if lm_ignore_val:
                    target = target * mask - 10000.0 * (1 - mask)  # loss not computed at -10000 labels
                target = target.squeeze(-1)  # [L]

            elif lm_loss == 'ce':
                mask = mask.long()
                target = sample_classes.clone()  # this was used by current version
                if lm_ignore_val:
                    target = target * mask - 100 * (1 - mask)  # loss not computed at -100 labels
                target = target.squeeze(-1)  # [L]

            sample = masked_sample

        elif causal_lm:

            if lm_loss == 'l1':
                target = F.pad(sample[1:], (0, 0, 0, 1))  # [L 1]
                assert sample.shape == target.shape

            elif lm_loss == 'ce':
                target = F.pad(sample_classes[1:], (0, 0, 0, 1), value=-100)
                target = target.squeeze(-1)  # [L]

        return sample, target


class PathFinderLM(SequenceDataset):
    _name_ = "pathfinder_lm"

    @property
    def d_input(self):
        return 2 if self.mlm_prob else 1

    @property
    def l_output(self):
        return None if self.mlm_prob or self.causal_lm else 0

    @property
    def d_output(self):
        if self.mlm_prob or self.causal_lm:
            return 256 if self.lm_loss == 'ce' else 1
        return 2

    @property
    def init_defaults(self):
        return {
            "resolution": 32,
            "sequential": True,
            "val_split": 0.1,
            "normalize": True,  # normalize the image before masking
            "test_split": 0.1,
            "seed": 42,  # Controls the train/val/test split

            # pre training args
            "mlm_prob": 0,  # self pre-training : prob of masking out a pixel
            "causal_lm": False,  # self pre-training : predict next pixel in sequence
            "span_masking": False,
            "span_length": 0,
            "lm_loss": 'ce',  # self pre-training loss: cross-entropy (ce) or l1
            "ignore_val": True,  # ignore the loss at masked pixels
        }

    def init(self):
        if self.data_dir is None:
            self.data_dir = (
                    default_data_path / "pathfinder" / f"pathfinder{self.resolution}"
            )
        assert self.sequential

    def prepare_data(self):
        if not self.data_dir.is_dir():
            raise FileNotFoundError(
                f"""
            Directory {str(self.data_dir)} not found.
            To get the dataset, download lra_release.gz from
            https://github.com/google-research/long-range-arena,
            then unzip it with tar -xvf lra_release.gz.
            Then point data_dir to the pathfinderX directory, where X is either 32, 64, 128, or 256.
            """
            )

    def setup(self, stage=None):
        if stage == "test" and hasattr(self, "dataset_test"):
            return
        # [2021-08-18] TD: I ran into RuntimeError: Too many open files.
        # https://github.com/pytorch/pytorch/issues/11201
        torch.multiprocessing.set_sharing_strategy("file_system")

        lm_args = dotdict(mlm_prob=self.mlm_prob, causal_lm=self.causal_lm, lm_loss=self.lm_loss,
                          sequential=self.sequential, normalize=self.normalize, ignore_val=self.ignore_val,
                          span_masking=self.span_masking, span_length=self.span_length)
        dataset = PathFinderLMDataset(self.data_dir, lm_args)
        len_dataset = len(dataset)
        val_len = int(self.val_split * len_dataset)
        test_len = int(self.test_split * len_dataset)
        train_len = len_dataset - val_len - test_len
        (
            self.dataset_train,
            self.dataset_val,
            self.dataset_test,
        ) = torch.utils.data.random_split(
            dataset,
            [train_len, val_len, test_len],
            generator=torch.Generator().manual_seed(self.seed),
        )


class IMDB_New(SequenceDataset):
    """A copy of the IMDB dataset class with additional tokens <mask>, <cls> to
    match vocab of IMDBLM, duplicated the dataset to not touch source"""
    _name_ = "imdb_new"
    d_output = 2
    l_output = 0

    @property
    def init_defaults(self):
        return {
            "l_max": 4096,
            "level": "char",
            "min_freq": 15,
            "seed": 42,
            "val_split": 0.0,
            "append_bos": False,
            "append_eos": True,
            # 'max_vocab': 135,
            "n_workers": 4,  # Only used for tokenizing dataset before caching
            "append_cls": False
        }

    @property
    def n_tokens(self):
        return len(self.vocab)

    def prepare_data(self):
        if self.cache_dir is None:  # Just download the dataset
            load_dataset(self._name_, cache_dir=self.data_dir)
        else:  # Process the dataset and save it
            self.process_dataset()

    def setup(self, stage=None):
        """If cache_dir is not None, we'll cache the processed dataset there."""
        self.data_dir = self.data_dir or default_data_path / self._name_
        self.cache_dir = self.data_dir / "cache"
        assert self.level in [
            "word",
            "char",
        ], f"level {self.level} not supported"

        if stage == "test" and hasattr(self, "dataset_test"):
            return
        dataset, self.tokenizer, self.vocab = self.process_dataset()
        print(
            f"IMDB {self.level} level | min_freq {self.min_freq} | vocab size {len(self.vocab)}"
        )
        dataset.set_format(type="torch", columns=["input_ids", "label"])

        # Create all splits
        dataset_train, self.dataset_test = dataset["train"], dataset["test"]
        if self.val_split == 0.0:
            # Use test set as val set, as done in the LRA paper
            self.dataset_train, self.dataset_val = dataset_train, None
        else:
            train_val = dataset_train.train_test_split(
                test_size=self.val_split, seed=self.seed
            )
            self.dataset_train, self.dataset_val = (
                train_val["train"],
                train_val["test"],
            )

    def _collate_fn(self, batch):
        xs, ys = zip(*[(data["input_ids"], data["label"]) for data in batch])
        lengths = torch.tensor([len(x) for x in xs])
        xs = nn.utils.rnn.pad_sequence(
            xs, padding_value=self.vocab["<pad>"], batch_first=True
        )
        ys = torch.tensor(ys)
        return xs, ys, {"lengths": lengths}

        # self._collate_fn = collate_batch

    def process_dataset(self):
        cache_dir = (
            None if self.cache_dir is None else self.cache_dir / self._cache_dir_name
        )
        if cache_dir is not None:
            if cache_dir.is_dir():
                return self._load_from_cache(cache_dir)
        # have to change the name manually for it to load the dataset
        dataset = load_dataset('imdb', cache_dir=self.data_dir)
        dataset = DatasetDict(train=dataset["train"], test=dataset["test"])
        if self.level == "word":
            tokenizer = torchtext.data.utils.get_tokenizer(
                "spacy", language="en_core_web_sm"
            )
        else:  # self.level == 'char'
            tokenizer = list  # Just convert a string to a list of chars
        # Account for <bos> and <eos> tokens
        l_max = self.l_max - int(self.append_bos) - int(self.append_eos)
        tokenize = lambda example: {"tokens": tokenizer(example["text"])[:l_max]}
        dataset = dataset.map(
            tokenize,
            remove_columns=["text"],
            keep_in_memory=True,
            load_from_cache_file=False,
            num_proc=max(self.n_workers, 1),
        )
        # modified vocab with <mask> token for lm, <cls> token for downstream
        vocab = torchtext.vocab.build_vocab_from_iterator(
            dataset["train"]["tokens"],
            min_freq=self.min_freq,
            specials=(
                    ["<mask>", "<pad>", "<unk>"]
                    + (["<bos>"] if self.append_bos else [])
                    + (["<eos>"] if self.append_eos else [])
                    + (["<cls>"] if self.append_cls else [])
            ),
        )
        vocab.set_default_index(vocab["<unk>"])

        numericalize = lambda example: {
            "input_ids": vocab(
                (["<bos>"] if self.append_bos else [])
                + example["tokens"]
                + (["<eos>"] if self.append_eos else [])
            )
        }
        dataset = dataset.map(
            numericalize,
            remove_columns=["tokens"],
            keep_in_memory=True,
            load_from_cache_file=False,
            num_proc=max(self.n_workers, 1),
        )

        if cache_dir is not None:
            self._save_to_cache(dataset, tokenizer, vocab, cache_dir)
        return dataset, tokenizer, vocab

    def _save_to_cache(self, dataset, tokenizer, vocab, cache_dir):
        cache_dir = self.cache_dir / self._cache_dir_name
        logger = logging.getLogger(__name__)
        logger.info(f"Saving to cache at {str(cache_dir)}")
        dataset.save_to_disk(str(cache_dir))
        with open(cache_dir / "tokenizer.pkl", "wb") as f:
            pickle.dump(tokenizer, f)
        with open(cache_dir / "vocab.pkl", "wb") as f:
            pickle.dump(vocab, f)

    def _load_from_cache(self, cache_dir):
        assert cache_dir.is_dir()
        logger = logging.getLogger(__name__)
        logger.info(f"Load from cache at {str(cache_dir)}")
        dataset = DatasetDict.load_from_disk(str(cache_dir))
        with open(cache_dir / "tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)
        with open(cache_dir / "vocab.pkl", "rb") as f:
            vocab = pickle.load(f)
        return dataset, tokenizer, vocab

    @property
    def _cache_dir_name(self):
        return f"l_max-{self.l_max}-level-{self.level}-min_freq-{self.min_freq}-append_bos-{self.append_bos}-append_eos-{self.append_eos}"

class IMDBLM(SequenceDataset):
    """
    IMDB dataet modified for language modeling pre-training.
    Main Changes:
    1. datasets train, val, test are wrapped in a MaskedSequenceDataset class that allows a get_item method
    that applies masking.
    2. self.collate_fn modified to pad the sequences with the <pad> token and not to clash with <mask> token, return
    labels for lm loss and not for classification.
    3. self.vocab modified to include <mask>, <cls> tokens.
    """
    _name_ = "imdb_lm"

    @property
    def l_output(self):
        return None if self.mlm_prob or self.causal_lm else 0

    @property
    def d_output(self):
        return self.n_toknes

    # this should not be here but was used during fine-tuning, pre-training
    # doesn't seem to do anything so keeping for case of future bugs.
    @property
    def d_input(self):
        return 2 if self.mlm_prob else 1

    @property
    def init_defaults(self):
        return {
            ### IMDB params
            "l_max": 4096,
            "level": "char",
            "min_freq": 15,
            "seed": 42,
            "val_split": 0.0,
            "append_bos": False,
            "append_eos": True,
            # 'max_vocab': 135,
            "n_workers": 4,  # Only used for tokenizing dataset before caching
            # additoinal cls token for downstream classification - doesn't exist in source dataset
            "append_cls": False,

            ### Language modeling params - doesn't exist in source dataset
            "sequential": True,
            "mlm_prob": 0,  # self pre-training : prob of masking out a pixel
            "causal_lm": False,  # self pre-training : predict next pixel in sequence
            "lm_loss": 'ce',  # self pre-training loss: cross-entropy (ce) or l1
            "ignore_val": True,  # ignore the loss at masked pixels
            "span_masking": False,
            "span_length": 0,
        }

    @property
    def n_tokens(self):
        return len(self.vocab)

    def prepare_data(self):
        if self.cache_dir is None:  # Just download the dataset
            load_dataset(self._name_, cache_dir=self.data_dir)
        else:  # Process the dataset and save it
            self.process_dataset()

    def setup(self, stage=None):
        """If cache_dir is not None, we'll cache the processed dataset there."""
        self.data_dir = self.data_dir or default_data_path / self._name_
        self.cache_dir = self.data_dir / "cache"
        assert self.level in [
            "word",
            "char",
        ], f"level {self.level} not supported"

        if stage == "test" and hasattr(self, "dataset_test"):
            return
        dataset, self.tokenizer, self.vocab = self.process_dataset()
        print(
            f"IMDB {self.level} level | min_freq {self.min_freq} | vocab size {len(self.vocab)}"
        )
        dataset.set_format(type="torch", columns=["input_ids", "label"])

        # Create all splits
        dataset_train, self.dataset_test = dataset["train"], dataset["test"]
        if self.val_split == 0.0:
            # Use test set as val set, as done in the LRA paper
            self.dataset_train, self.dataset_val = dataset_train, None
        else:
            train_val = dataset_train.train_test_split(
                test_size=self.val_split, seed=self.seed
            )
            self.dataset_train, self.dataset_val = (
                train_val["train"],
                train_val["test"],
            )

        ## Modification to source dataset
        # Wrap the datasets with class that hs bult in masking functionality
        self.dataset_train = MaskedIMDB(dataset=self.dataset_train,
                                        mlm_prob=self.mlm_prob,
                                        causal_lm=self.causal_lm,
                                        ignore_val=self.ignore_val,
                                        span_masking=self.span_masking,
                                        span_length=self.span_length,
                                        lm_loss=self.lm_loss)
        self.dataset_test = MaskedIMDB(dataset=self.dataset_test,
                                       mlm_prob=self.mlm_prob,
                                       causal_lm=self.causal_lm,
                                       ignore_val=self.ignore_val,
                                       span_masking=self.span_masking,
                                       span_length=self.span_length,
                                       lm_loss=self.lm_loss)
        if self.dataset_val is not None:
            self.dataset_val = MaskedIMDB(dataset=self.dataset_val,
                                          mlm_prob=self.mlm_prob,
                                          causal_lm=self.causal_lm,
                                          ignore_val=self.ignore_val,
                                          span_masking=self.span_masking,
                                          span_length=self.span_length,
                                          lm_loss=self.lm_loss)

        if not self.ignore_val:
            raise NotImplementedError("LM on IMDB is not implemented without ignore_val == True since uses ignore val also for sequence padding values")

        # special value for padding in label sequence
        self.padding_ignore_value = self.dataset_train.ignore_val   # don't apply MLM on padding values

    def _collate_fn(self, batch):
        """modified from original to allow language modeling on the input sequence"""
        xs, ys = zip(*[(data["input_ids"], data["label"]) for data in batch])
        xs, ys = self._append_cls(xs, ys)
        lengths = torch.tensor([len(x) for x in xs])

        # inputs are padded with padding token
        xs = nn.utils.rnn.pad_sequence(
            xs, padding_value=self.vocab["<pad>"], batch_first=True
        )
        # labels are padded with ignore val - also in position of <cls>
        ys = nn.utils.rnn.pad_sequence(
            ys, padding_value=self.padding_ignore_value, batch_first=True
        )
        return xs, ys, {"lengths": lengths}

    def _append_cls(self, xs, ys):
        """Append <cls> token to the end of each sequence."""
        if self.append_cls:
            xs = [torch.concat([x,
                                torch.tensor([self.vocab["<cls>"]])
                                ], dim=-1) for x in xs]
            ys = [torch.concat([y,
                                torch.tensor([self.padding_ignore_value])
                                ], dim=-1) for y in ys]
        return xs, ys

    def process_dataset(self):
        cache_dir = (
            None if self.cache_dir is None else self.cache_dir / self._cache_dir_name
        )
        if cache_dir is not None:
            if cache_dir.is_dir():
                return self._load_from_cache(cache_dir)

        # have to change the name manually for it to load the dataset
        dataset = load_dataset('imdb', cache_dir=self.data_dir)
        dataset = DatasetDict(train=dataset["train"], test=dataset["test"])
        if self.level == "word":
            tokenizer = torchtext.data.utils.get_tokenizer(
                "spacy", language="en_core_web_sm"
            )
        else:  # self.level == 'char'
            tokenizer = list  # Just convert a string to a list of chars

        # Account for <bos> and <eos> tokens in sequence length
        l_max = self.l_max - int(self.append_bos) - int(self.append_eos)
        tokenize = lambda example: {"tokens": tokenizer(example["text"])[:l_max]}
        dataset = dataset.map(
            tokenize,
            remove_columns=["text"],
            keep_in_memory=True,
            load_from_cache_file=False,
            num_proc=max(self.n_workers, 1),
        )

        # modified vocab with <mask> token for lm, <cls> token for downstream
        vocab = torchtext.vocab.build_vocab_from_iterator(
            dataset["train"]["tokens"],
            min_freq=self.min_freq,
            specials=(
                    ["<mask>", "<pad>", "<unk>"]
                    + (["<bos>"] if self.append_bos else [])
                    + (["<eos>"] if self.append_eos else [])
                    + (["<cls>"] if self.append_cls else [])
            ),
        )
        vocab.set_default_index(vocab["<unk>"])

        numericalize = lambda example: {
            "input_ids": vocab(
                (["<bos>"] if self.append_bos else [])
                + example["tokens"]
                + (["<eos>"] if self.append_eos else [])
            )
        }
        dataset = dataset.map(
            numericalize,
            remove_columns=["tokens"],
            keep_in_memory=True,
            load_from_cache_file=False,
            num_proc=max(self.n_workers, 1),
        )

        if cache_dir is not None:
            self._save_to_cache(dataset, tokenizer, vocab, cache_dir)
        return dataset, tokenizer, vocab

    def _save_to_cache(self, dataset, tokenizer, vocab, cache_dir):
        cache_dir = self.cache_dir / self._cache_dir_name
        logger = logging.getLogger(__name__)
        logger.info(f"Saving to cache at {str(cache_dir)}")
        dataset.save_to_disk(str(cache_dir))
        with open(cache_dir / "tokenizer.pkl", "wb") as f:
            pickle.dump(tokenizer, f)
        with open(cache_dir / "vocab.pkl", "wb") as f:
            pickle.dump(vocab, f)

    def _load_from_cache(self, cache_dir):
        assert cache_dir.is_dir()
        logger = logging.getLogger(__name__)
        logger.info(f"Load from cache at {str(cache_dir)}")
        dataset = DatasetDict.load_from_disk(str(cache_dir))
        with open(cache_dir / "tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)
        with open(cache_dir / "vocab.pkl", "rb") as f:
            vocab = pickle.load(f)
        return dataset, tokenizer, vocab

    @property
    def _cache_dir_name(self):
        return f"l_max-{self.l_max}-level-{self.level}-min_freq-{self.min_freq}-append_bos-{self.append_bos}-append_eos-{self.append_eos}"


class MaskedIMDB(MaskedSequenceDatasetWrapper):

    def __getitem__(self, index):
        # get item from the warpped dataset
        sample = self.dataset[index]
        input_ids = sample["input_ids"]
        label = sample["label"]

        # samples are 1D and mask expects [L 1] so expand
        input_ids = input_ids.unsqueeze(1).float()

        # apply masking - the classes are the token values
        inputs, target = self.mask(input_ids, sample_classes=input_ids)

        # since using an nn.Embedding module can throw away the mask
        inputs = inputs[..., 0]
        # move everything to long
        inputs = inputs.long()
        target = target.long()

        # return the sample and the target
        sample["input_ids"] = inputs
        sample["label"] = target

        return sample


class ListOps_New(SequenceDataset):
    _name_ = "listops_new"
    d_output = 10
    l_output = 0

    @property
    def init_defaults(self):
        return {
            "l_max": 2048,
            "append_bos": False,
            "append_eos": True,
            # 'max_vocab': 20, # Actual size 18
            "n_workers": 4,  # Only used for tokenizing dataset
        }

    @property
    def n_tokens(self):
        return len(self.vocab)

    @property
    def _cache_dir_name(self):
        return f"l_max-{self.l_max}-append_bos-{self.append_bos}-append_eos-{self.append_eos}"

    def init(self):
        if self.data_dir is None:
            self.data_dir = default_data_path / self._name_
        self.cache_dir = self.data_dir / self._cache_dir_name

    def prepare_data(self):
        if self.cache_dir is None:
            for split in ["train", "val", "test"]:
                split_path = self.data_dir / f"basic_{split}.tsv"
                if not split_path.is_file():
                    raise FileNotFoundError(
                        f"""
                    File {str(split_path)} not found.
                    To get the dataset, download lra_release.gz from
                    https://github.com/google-research/long-range-arena,
                    then unzip it with tar -xvf lra_release.gz.
                    Then point data_dir to the listops-1000 directory.
                    """
                    )
        else:  # Process the dataset and save it
            self.process_dataset()

    def setup(self, stage=None):
        if stage == "test" and hasattr(self, "dataset_test"):
            return
        dataset, self.tokenizer, self.vocab = self.process_dataset()
        self.vocab_size = len(self.vocab)
        dataset.set_format(type="torch", columns=["input_ids", "Target"])
        self.dataset_train, self.dataset_val, self.dataset_test = (
            dataset["train"],
            dataset["val"],
            dataset["test"],
        )

        def collate_batch(batch):
            xs, ys = zip(*[(data["input_ids"], data["Target"]) for data in batch])
            lengths = torch.tensor([len(x) for x in xs])
            xs = nn.utils.rnn.pad_sequence(
                xs, padding_value=self.vocab["<pad>"], batch_first=True
            )
            ys = torch.tensor(ys)
            return xs, ys, {"lengths": lengths}

        self._collate_fn = collate_batch

    def process_dataset(self):
        cache_dir = (
            None if self.cache_dir is None else self.cache_dir / self._cache_dir_name
        )
        if cache_dir is not None:
            if cache_dir.is_dir():
                return self._load_from_cache(cache_dir)

        dataset = load_dataset(
            "csv",
            data_files={
                "train": str(self.data_dir / "basic_train.tsv"),
                "val": str(self.data_dir / "basic_val.tsv"),
                "test": str(self.data_dir / "basic_test.tsv"),
            },
            delimiter="\t",
            keep_in_memory=True,
        )

        tokenizer = listops_tokenizer

        # Account for <bos> and <eos> tokens
        l_max = self.l_max - int(self.append_bos) - int(self.append_eos)
        tokenize = lambda example: {"tokens": tokenizer(example["Source"])[:l_max]}
        dataset = dataset.map(
            tokenize,
            remove_columns=["Source"],
            keep_in_memory=True,
            load_from_cache_file=False,
            num_proc=max(self.n_workers, 1),
        )
        vocab = torchtext.vocab.build_vocab_from_iterator(
            dataset["train"]["tokens"],
            specials=(
                    ["<mask>","<pad>", "<unk>"]
                    + (["<bos>"] if self.append_bos else [])
                    + (["<eos>"] if self.append_eos else [])
            ),
        )
        vocab.set_default_index(vocab["<unk>"])

        numericalize = lambda example: {
            "input_ids": vocab(
                (["<bos>"] if self.append_bos else [])
                + example["tokens"]
                + (["<eos>"] if self.append_eos else [])
            )
        }
        dataset = dataset.map(
            numericalize,
            remove_columns=["tokens"],
            keep_in_memory=True,
            load_from_cache_file=False,
            num_proc=max(self.n_workers, 1),
        )

        if cache_dir is not None:
            self._save_to_cache(dataset, tokenizer, vocab, cache_dir)
        return dataset, tokenizer, vocab

    def _save_to_cache(self, dataset, tokenizer, vocab, cache_dir):
        cache_dir = self.cache_dir / self._cache_dir_name
        logger = logging.getLogger(__name__)
        logger.info(f"Saving to cache at {str(cache_dir)}")
        dataset.save_to_disk(str(cache_dir))
        with open(cache_dir / "tokenizer.pkl", "wb") as f:
            pickle.dump(tokenizer, f)
        with open(cache_dir / "vocab.pkl", "wb") as f:
            pickle.dump(vocab, f)

    def _load_from_cache(self, cache_dir):
        assert cache_dir.is_dir()
        logger = logging.getLogger(__name__)
        logger.info(f"Load from cache at {str(cache_dir)}")
        dataset = DatasetDict.load_from_disk(str(cache_dir))
        with open(cache_dir / "tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)
        with open(cache_dir / "vocab.pkl", "rb") as f:
            vocab = pickle.load(f)
        return dataset, tokenizer, vocab



class ListOpsLM(SequenceDataset):
    _name_ = "listops_lm"

    @property
    def l_output(self):
        return None if self.mlm_prob or self.causal_lm else 0

    @property
    def d_output(self):
        return self.n_tokens

    @property
    def init_defaults(self):
        return {
            "l_max": 2048,
            "append_bos": False,
            "append_eos": True,
            # 'max_vocab': 20, # Actual size 18
            "n_workers": 4,  # Only used for tokenizing dataset

            ### Language modeling params - doesn't exist in source dataset
            "sequential": True,
            "mlm_prob": 0,  # self pre-training : prob of masking out a pixel
            "causal_lm": False,  # self pre-training : predict next pixel in sequence
            "lm_loss": 'ce',  # self pre-training loss: cross-entropy (ce) or l1
            "ignore_val": True,  # ignore the loss at masked pixels
            "span_masking": False,
            "span_length": 0,
            "nesting_lvl": False
        }


    @property
    def n_tokens(self):
        return len(self.vocab)

    @property
    def _cache_dir_name(self):
        return f"l_max-{self.l_max}-append_bos-{self.append_bos}-append_eos-{self.append_eos}"

    def init(self):
        if self.data_dir is None:
            self.data_dir = default_data_path / self._name_
        self.cache_dir = self.data_dir / self._cache_dir_name

    def prepare_data(self):
        if self.cache_dir is None:
            for split in ["train", "val", "test"]:
                split_path = self.data_dir / f"basic_{split}.tsv"
                if not split_path.is_file():
                    raise FileNotFoundError(
                        f"""
                    File {str(split_path)} not found.
                    To get the dataset, download lra_release.gz from
                    https://github.com/google-research/long-range-arena,
                    then unzip it with tar -xvf lra_release.gz.
                    Then point data_dir to the listops-1000 directory.
                    """
                    )
        else:  # Process the dataset and save it
            self.process_dataset()

    def setup(self, stage=None):
        if stage == "test" and hasattr(self, "dataset_test"):
            return
        dataset, self.tokenizer, self.vocab = self.process_dataset()
        self.vocab_size = len(self.vocab)
        dataset.set_format(type="torch", columns=["input_ids", "Target"])
        self.dataset_train, self.dataset_val, self.dataset_test = (
            dataset["train"],
            dataset["val"],
            dataset["test"],
        )

        # Wrap datasets in a MaskedSequenceDatasetWrapper
        self.dataset_train = MaskedListOps(dataset=self.dataset_train,
                                           mlm_prob=self.mlm_prob,
                                           causal_lm=self.causal_lm,
                                           ignore_val=self.ignore_val,
                                           span_masking=self.span_masking,
                                           span_length=self.span_length,
                                           lm_loss=self.lm_loss,
                                           tag_nesting_lvl=self.nesting_lvl,
                                           vocab=self.vocab)

        self.dataset_val = MaskedListOps(dataset=self.dataset_val,
                                         mlm_prob=self.mlm_prob,
                                         causal_lm=self.causal_lm,
                                         ignore_val=self.ignore_val,
                                         span_masking=self.span_masking,
                                         span_length=self.span_length,
                                         lm_loss=self.lm_loss,
                                         tag_nesting_lvl=self.nesting_lvl,
                                         vocab=self.vocab)

        self.dataset_test = MaskedListOps(dataset=self.dataset_test,
                                          mlm_prob=self.mlm_prob,
                                          causal_lm=self.causal_lm,
                                          ignore_val=self.ignore_val,
                                          span_masking=self.span_masking,
                                          span_length=self.span_length,
                                          lm_loss=self.lm_loss,
                                          tag_nesting_lvl=self.nesting_lvl,
                                          vocab=self.vocab)

        if not self.ignore_val:
            raise NotImplementedError("LM on ListOps is not implemented without ignore_val == True since uses ignore val also for sequence padding values")

        # special value for padding in label sequence
        self.padding_ignore_value = self.dataset_train.ignore_val   # don't apply MLM on padding values

        def collate_batch(batch):
            xs, ys = zip(*[(data["input_ids"], data["Target"]) for data in batch])
            lengths = torch.tensor([len(x) for x in xs])

            # xs padded with <pad> token
            xs = nn.utils.rnn.pad_sequence(
                xs, padding_value=self.vocab["<pad>"], batch_first=True
            )

            # ys are padded with ignore value - don't want loss on padding
            ys = nn.utils.rnn.pad_sequence(
                ys, padding_value=self.padding_ignore_value, batch_first=True
            )
            return xs, ys, {"lengths": lengths}

        self._collate_fn = collate_batch

    def process_dataset(self):
        cache_dir = (
            None if self.cache_dir is None else self.cache_dir / self._cache_dir_name
        )
        if cache_dir is not None:
            if cache_dir.is_dir():
                return self._load_from_cache(cache_dir)

        # changing data_dir manually to correct one
        dataset = load_dataset(
            "csv",
            data_files={
                "train": str("data/listops/basic_train.tsv"),
                "val": str("data/listops/basic_val.tsv"),
                "test": str("data/listops/basic_test.tsv"),
            },
            delimiter="\t",
            keep_in_memory=True,
        )

        tokenizer = listops_tokenizer

        # Account for <bos> and <eos> tokens
        l_max = self.l_max - int(self.append_bos) - int(self.append_eos)
        tokenize = lambda example: {"tokens": tokenizer(example["Source"])[:l_max]}
        dataset = dataset.map(
            tokenize,
            remove_columns=["Source"],
            keep_in_memory=True,
            load_from_cache_file=False,
            num_proc=max(self.n_workers, 1),
        )

        # modified vocab with <mask> token for lm
        vocab = torchtext.vocab.build_vocab_from_iterator(
            dataset["train"]["tokens"],
            specials=(
                    ["<mask>", "<pad>", "<unk>"]
                    + (["<bos>"] if self.append_bos else [])
                    + (["<eos>"] if self.append_eos else [])
            ),
        )
        vocab.set_default_index(vocab["<unk>"])

        numericalize = lambda example: {
            "input_ids": vocab(
                (["<bos>"] if self.append_bos else [])
                + example["tokens"]
                + (["<eos>"] if self.append_eos else [])
            )
        }
        dataset = dataset.map(
            numericalize,
            remove_columns=["tokens"],
            keep_in_memory=True,
            load_from_cache_file=False,
            num_proc=max(self.n_workers, 1),
        )

        if cache_dir is not None:
            self._save_to_cache(dataset, tokenizer, vocab, cache_dir)
        return dataset, tokenizer, vocab

    def _save_to_cache(self, dataset, tokenizer, vocab, cache_dir):
        cache_dir = self.cache_dir / self._cache_dir_name
        logger = logging.getLogger(__name__)
        logger.info(f"Saving to cache at {str(cache_dir)}")
        dataset.save_to_disk(str(cache_dir))
        with open(cache_dir / "tokenizer.pkl", "wb") as f:
            pickle.dump(tokenizer, f)
        with open(cache_dir / "vocab.pkl", "wb") as f:
            pickle.dump(vocab, f)

    def _load_from_cache(self, cache_dir):
        assert cache_dir.is_dir()
        logger = logging.getLogger(__name__)
        logger.info(f"Load from cache at {str(cache_dir)}")
        dataset = DatasetDict.load_from_disk(str(cache_dir))
        with open(cache_dir / "tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)
        with open(cache_dir / "vocab.pkl", "rb") as f:
            vocab = pickle.load(f)
        return dataset, tokenizer, vocab


class MaskedListOps(MaskedIMDB):

    def __init__(self, tag_nesting_lvl=False, vocab=None, **kwargs):
        super().__init__(**kwargs)
        self.nesting_lvl = tag_nesting_lvl
        self.vocab = vocab

        if self.nesting_lvl:
            assert self.vocab is not None, "to tag nesting lvl must pass vocab to MaskedListOps"

    def __getitem__(self, index):
        # get item from the warpped dataset
        sample = self.dataset[index]
        input_ids = sample["input_ids"]
        label = sample["Target"]

        # samples are 1D and mask expects [L 1] so expand
        input_ids = input_ids.unsqueeze(1).float()

        # apply masking - the classes are the token values
        inputs, target = self.mask(input_ids, sample_classes=input_ids)

        # move everything to long
        inputs = inputs.long()
        target = target.long()

        if self.nesting_lvl:
            # we want unsmasked input for this op.
            target = self.nesting_lvl_tags(sequence=input_ids.long().squeeze(-1), mask=inputs[..., 1])

        # since using an nn.Embedding module can throw away the mask dimension
        inputs = inputs[..., 0]

        # return the sample and the target
        sample["input_ids"] = inputs
        sample["Target"] = target

        return sample

    def nesting_lvl_tags(self, sequence, mask):
        """
        Create token lvl labels that are the nesting lvl 1-9, e.g.
        [MAX 1 2 3 [MIN 7 8 9] 4 5] will be labeled as: ignr 1 1 1 ignr 2 2 2 ignr 1 ignr
        mask is used o only tag some of the elements in the sequence.
        where [Operator, ] are ignored
        Args:
            sequence, mask - 1D tensors
        """
        labels = torch.zeros_like(sequence)
        nesting_lvl = 0
        for i, s in enumerate(sequence):
            s = self.vocab.lookup_token(s.item())
            if s.startswith('['):
                nesting_lvl += 1
                labels[i] = self.ignore_val
            # listops tokenizer is sometimes set to represent ']' as 'X'
            elif s == 'X' or s == ']':
                nesting_lvl -= 1
                labels[i] = self.ignore_val
            else:
                labels[i] = nesting_lvl

        labels[mask == 0] = self.ignore_val
        return labels


class AAN_New(SequenceDataset):
    _name_ = "aan_new"
    d_output = 2  # Use accuracy instead of binary_accuracy
    l_output = 0

    @property
    def n_tokens(self):
        return len(self.vocab)

    @property
    def init_defaults(self):
        return {
            "l_max": 4000,
            # 'max_vocab': 100, # Full size 98
            "append_bos": False,
            "append_eos": True,
            "n_workers": 4,  # For tokenizing only
        }

    @property
    def _cache_dir_name(self):
        return f"l_max-{self.l_max}-append_bos-{self.append_bos}-append_eos-{self.append_eos}"

    def init(self):
        if self.data_dir is None:
            self.data_dir = default_data_path / self._name_
        self.cache_dir = self.data_dir / self._cache_dir_name

    def prepare_data(self):
        if self.cache_dir is None:
            for split in ["train", "eval", "test"]:
                split_path = self.data_dir / f"new_aan_pairs.{split}.tsv"
                if not split_path.is_file():
                    raise FileNotFoundError(
                        f"""
                    File {str(split_path)} not found.
                    To get the dataset, download lra_release.gz from
                    https://github.com/google-research/long-range-arena,
                    then unzip it with tar -xvf lra_release.gz.
                    Then point data_dir to the tsv_data directory.
                    """
                    )
        else:  # Process the dataset and save it
            self.process_dataset()

    def setup(self, stage=None):
        if stage == "test" and hasattr(self, "dataset_test"):
            return

        # [2021-08-18] TD: I ran into RuntimeError: Too many open files.
        # https://github.com/pytorch/pytorch/issues/11201
        torch.multiprocessing.set_sharing_strategy("file_system")

        dataset, self.tokenizer, self.vocab = self.process_dataset()
        # self.vocab_size = len(self.vocab)
        print("AAN vocab size:", len(self.vocab))

        dataset.set_format(type="torch", columns=["input_ids1", "input_ids2", "label"])
        self.dataset_train, self.dataset_val, self.dataset_test = (
            dataset["train"],
            dataset["val"],
            dataset["test"],
        )

        def collate_batch(batch):
            xs1, xs2, ys = zip(
                *[
                    (data["input_ids1"], data["input_ids2"], data["label"])
                    for data in batch
                ]
            )
            lengths1 = torch.tensor([len(x) for x in xs1])
            lengths2 = torch.tensor([len(x) for x in xs2])
            xs1 = nn.utils.rnn.pad_sequence(
                xs1, padding_value=self.vocab["<pad>"], batch_first=True
            )
            xs2 = nn.utils.rnn.pad_sequence(
                xs2, padding_value=self.vocab["<pad>"], batch_first=True
            )
            # Pad both to same length
            # Shape (batch, length)
            L = max(xs1.size(1), xs2.size(1))
            xs1 = F.pad(xs1, (0, L - xs1.size(1)), value=self.vocab["<pad>"])
            xs2 = F.pad(xs2, (0, L - xs2.size(1)), value=self.vocab["<pad>"])
            ys = torch.tensor(ys)
            # return xs1, xs2, ys, lengths1, lengths2

            # Concatenate two batches
            xs = torch.cat([xs1, xs2], dim=0)
            lengths = torch.cat([lengths1, lengths2], dim=0)
            return xs, ys, {"lengths": lengths}

        self._collate_fn = collate_batch

    def process_dataset(self):
        cache_dir = (
            None if self.cache_dir is None else self.cache_dir / self._cache_dir_name
        )
        if cache_dir is not None:
            if cache_dir.is_dir():
                return self._load_from_cache(cache_dir)

        dataset = load_dataset(
            "csv",
            data_files={
                "train": str(self.data_dir / "new_aan_pairs.train.tsv"),
                "val": str(self.data_dir / "new_aan_pairs.eval.tsv"),
                "test": str(self.data_dir / "new_aan_pairs.test.tsv"),
            },
            delimiter="\t",
            column_names=["label", "input1_id", "input2_id", "text1", "text2"],
            keep_in_memory=True,
            cache_dir=AAN_CACHE_DIR,
        )  # True)
        dataset = dataset.remove_columns(["input1_id", "input2_id"])
        new_features = dataset["train"].features.copy()
        new_features["label"] = Value("int32")
        dataset = dataset.cast(new_features)

        tokenizer = list  # Just convert a string to a list of chars
        # Account for <bos> and <eos> tokens
        l_max = self.l_max - int(self.append_bos) - int(self.append_eos)
        tokenize = lambda example: {
            "tokens1": tokenizer(example["text1"])[:l_max],
            "tokens2": tokenizer(example["text2"])[:l_max],
        }
        dataset = dataset.map(
            tokenize,
            remove_columns=["text1", "text2"],
            keep_in_memory=True,
            load_from_cache_file=False,
            num_proc=max(self.n_workers, 1),
        )
        vocab = torchtext.vocab.build_vocab_from_iterator(
            dataset["train"]["tokens1"] + dataset["train"]["tokens2"],
            specials=(
                    ["<mask>", "<pad>", "<unk>"]
                    + (["<bos>"] if self.append_bos else [])
                    + (["<eos>"] if self.append_eos else [])
            ),
        )
        vocab.set_default_index(vocab["<unk>"])

        encode = lambda text: vocab(
            (["<bos>"] if self.append_bos else [])
            + text
            + (["<eos>"] if self.append_eos else [])
        )
        numericalize = lambda example: {
            "input_ids1": encode(example["tokens1"]),
            "input_ids2": encode(example["tokens2"]),
        }
        dataset = dataset.map(
            numericalize,
            remove_columns=["tokens1", "tokens2"],
            keep_in_memory=True,
            load_from_cache_file=False,
            num_proc=max(self.n_workers, 1),
        )

        if cache_dir is not None:
            self._save_to_cache(dataset, tokenizer, vocab, cache_dir)
        return dataset, tokenizer, vocab

    def _save_to_cache(self, dataset, tokenizer, vocab, cache_dir):
        cache_dir = self.cache_dir / self._cache_dir_name
        logger = logging.getLogger(__name__)
        logger.info(f"Saving to cache at {str(cache_dir)}")
        dataset.save_to_disk(str(cache_dir))
        with open(cache_dir / "tokenizer.pkl", "wb") as f:
            pickle.dump(tokenizer, f)
        with open(cache_dir / "vocab.pkl", "wb") as f:
            pickle.dump(vocab, f)

    def _load_from_cache(self, cache_dir):
        assert cache_dir.is_dir()
        logger = logging.getLogger(__name__)
        logger.info(f"Load from cache at {str(cache_dir)}")
        dataset = DatasetDict.load_from_disk(str(cache_dir))
        with open(cache_dir / "tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)
        with open(cache_dir / "vocab.pkl", "rb") as f:
            vocab = pickle.load(f)
        return dataset, tokenizer, vocab



class AANLM(SequenceDataset):
    _name_ = "aan_lm"

    @property
    def l_output(self):
        return None if self.mlm_prob or self.causal_lm else 0

    @property
    def d_output(self):
        return self.n_toknes

    @property
    def n_tokens(self):
        return len(self.vocab)

    @property
    def init_defaults(self):
        return {
            "l_max": 4000,
            # 'max_vocab': 100, # Full size 98
            "append_bos": False,
            "append_eos": True,
            "n_workers": 4,  # For tokenizing only

            ### Language modeling params - doesn't exist in source dataset
            "sequential": True,
            "mlm_prob": 0,  # self pre-training : prob of masking out a pixel
            "causal_lm": False,  # self pre-training : predict next pixel in sequence
            "lm_loss": 'ce',  # self pre-training loss: cross-entropy (ce) or l1
            "ignore_val": True,  # ignore the loss at masked pixels
            "span_masking": False,
            "span_length": 0,
        }

    @property
    def _cache_dir_name(self):
        return f"l_max-{self.l_max}-append_bos-{self.append_bos}-append_eos-{self.append_eos}"

    def init(self):
        if self.data_dir is None:
            self.data_dir = default_data_path / self._name_
        self.cache_dir = self.data_dir / self._cache_dir_name

    def prepare_data(self):
        if self.cache_dir is None:
            for split in ["train", "eval", "test"]:
                split_path = self.data_dir / f"new_aan_pairs.{split}.tsv"
                if not split_path.is_file():
                    raise FileNotFoundError(
                        f"""
                    File {str(split_path)} not found.
                    To get the dataset, download lra_release.gz from
                    https://github.com/google-research/long-range-arena,
                    then unzip it with tar -xvf lra_release.gz.
                    Then point data_dir to the tsv_data directory.
                    """
                    )
        else:  # Process the dataset and save it
            self.process_dataset()

    def setup(self, stage=None):
        if stage == "test" and hasattr(self, "dataset_test"):
            return

        # [2021-08-18] TD: I ran into RuntimeError: Too many open files.
        # https://github.com/pytorch/pytorch/issues/11201
        torch.multiprocessing.set_sharing_strategy("file_system")

        dataset, self.tokenizer, self.vocab = self.process_dataset()
        # self.vocab_size = len(self.vocab)
        print("AAN vocab size:", len(self.vocab))

        dataset.set_format(type="torch", columns=["input_ids1", "input_ids2", "label"])
        self.dataset_train, self.dataset_val, self.dataset_test = (
            dataset["train"],
            dataset["val"],
            dataset["test"],
        )

        # Wrap datasets in a MaskedSequenceDatasetWrapper
        self.dataset_train = MaskedAAN(dataset=self.dataset_train,
                                           mlm_prob=self.mlm_prob,
                                           causal_lm=self.causal_lm,
                                           ignore_val=self.ignore_val,
                                           span_masking=self.span_masking,
                                           span_length=self.span_length,
                                           lm_loss=self.lm_loss)

        self.dataset_val = MaskedAAN(dataset=self.dataset_val,
                                         mlm_prob=self.mlm_prob,
                                         causal_lm=self.causal_lm,
                                         ignore_val=self.ignore_val,
                                         span_masking=self.span_masking,
                                         span_length=self.span_length,
                                         lm_loss=self.lm_loss)

        self.dataset_test = MaskedAAN(dataset=self.dataset_test,
                                          mlm_prob=self.mlm_prob,
                                          causal_lm=self.causal_lm,
                                          ignore_val=self.ignore_val,
                                          span_masking=self.span_masking,
                                          span_length=self.span_length,
                                          lm_loss=self.lm_loss)

        if not self.ignore_val:
            raise NotImplementedError(
                "LM on ListOps is not implemented without ignore_val == True since uses ignore val also for sequence padding values")

        # special value for padding in label sequence
        self.padding_ignore_value = self.dataset_train.ignore_val  # don't apply MLM on padding values

        def collate_batch(batch):
            xs1, xs2, ys1, ys2 = zip(
                *[
                    (data["input_ids1"], data["input_ids2"], data["label1"], data["label2"])
                    for data in batch
                ]
            )
            lengths1 = torch.tensor([len(x) for x in xs1])
            lengths2 = torch.tensor([len(x) for x in xs2])

            # pad sequence & target sequence
            xs1 = nn.utils.rnn.pad_sequence(
                xs1, padding_value=self.vocab["<pad>"], batch_first=True
            )
            xs2 = nn.utils.rnn.pad_sequence(
                xs2, padding_value=self.vocab["<pad>"], batch_first=True
            )
            # padding_value for target sequence is set to ignore_val of the loss function
            ys1 = nn.utils.rnn.pad_sequence(
                ys1, padding_value=self.padding_ignore_value, batch_first=True
            )
            ys2 = nn.utils.rnn.pad_sequence(
                ys2, padding_value=self.padding_ignore_value, batch_first=True
            )

            # Pad all to same length
            # Shape (batch, length)
            L = max(xs1.size(1), xs2.size(1))
            xs1 = F.pad(xs1, (0, L - xs1.size(1)), value=self.vocab["<pad>"])
            xs2 = F.pad(xs2, (0, L - xs2.size(1)), value=self.vocab["<pad>"])
            ys1 = F.pad(ys1, (0, L - ys1.size(1)), value=self.padding_ignore_value)
            ys2 = F.pad(ys2, (0, L - ys2.size(1)), value=self.padding_ignore_value)
            # return xs1, xs2, ys, lengths1, lengths2

            # Concatenate two batches
            xs = torch.cat([xs1, xs2], dim=0)
            ys = torch.cat([ys1, ys2], dim=0)
            lengths = torch.cat([lengths1, lengths2], dim=0)
            return xs, ys, {"lengths": lengths}

        self._collate_fn = collate_batch

    def process_dataset(self):
        cache_dir = (
            None if self.cache_dir is None else self.cache_dir / self._cache_dir_name
        )
        if cache_dir is not None:
            if cache_dir.is_dir():
                return self._load_from_cache(cache_dir)

        dataset = load_dataset(
            "csv",
            data_files={
                "train": str(self.data_dir / "new_aan_pairs.train.tsv"),
                "val": str(self.data_dir / "new_aan_pairs.eval.tsv"),
                "test": str(self.data_dir / "new_aan_pairs.test.tsv"),
            },
            delimiter="\t",
            column_names=["label", "input1_id", "input2_id", "text1", "text2"],
            keep_in_memory=True,
            cache_dir=AAN_CACHE_DIR,  # manually changed the cache dir location since it was too big for the default cache dir
        )  # True)
        dataset = dataset.remove_columns(["input1_id", "input2_id"])
        new_features = dataset["train"].features.copy()
        new_features["label"] = Value("int32")
        dataset = dataset.cast(new_features)

        tokenizer = list  # Just convert a string to a list of chars
        # Account for <bos> and <eos> tokens
        l_max = self.l_max - int(self.append_bos) - int(self.append_eos)
        tokenize = lambda example: {
            "tokens1": tokenizer(example["text1"])[:l_max],
            "tokens2": tokenizer(example["text2"])[:l_max],
        }
        dataset = dataset.map(
            tokenize,
            remove_columns=["text1", "text2"],
            keep_in_memory=True,
            load_from_cache_file=False,
            num_proc=max(self.n_workers, 1),
        )
        vocab = torchtext.vocab.build_vocab_from_iterator(
            dataset["train"]["tokens1"] + dataset["train"]["tokens2"],
            specials=(
                    ["<mask>", "<pad>", "<unk>"]
                    + (["<bos>"] if self.append_bos else [])
                    + (["<eos>"] if self.append_eos else [])
            ),
        )
        vocab.set_default_index(vocab["<unk>"])

        encode = lambda text: vocab(
            (["<bos>"] if self.append_bos else [])
            + text
            + (["<eos>"] if self.append_eos else [])
        )
        numericalize = lambda example: {
            "input_ids1": encode(example["tokens1"]),
            "input_ids2": encode(example["tokens2"]),
        }
        dataset = dataset.map(
            numericalize,
            remove_columns=["tokens1", "tokens2"],
            keep_in_memory=True,
            load_from_cache_file=False,
            num_proc=max(self.n_workers, 1),
        )

        if cache_dir is not None:
            self._save_to_cache(dataset, tokenizer, vocab, cache_dir)
        return dataset, tokenizer, vocab

    def _save_to_cache(self, dataset, tokenizer, vocab, cache_dir):
        cache_dir = self.cache_dir / self._cache_dir_name
        logger = logging.getLogger(__name__)
        logger.info(f"Saving to cache at {str(cache_dir)}")
        dataset.save_to_disk(str(cache_dir))
        with open(cache_dir / "tokenizer.pkl", "wb") as f:
            pickle.dump(tokenizer, f)
        with open(cache_dir / "vocab.pkl", "wb") as f:
            pickle.dump(vocab, f)

    def _load_from_cache(self, cache_dir):
        assert cache_dir.is_dir()
        logger = logging.getLogger(__name__)
        logger.info(f"Load from cache at {str(cache_dir)}")
        dataset = DatasetDict.load_from_disk(str(cache_dir))
        with open(cache_dir / "tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)
        with open(cache_dir / "vocab.pkl", "rb") as f:
            vocab = pickle.load(f)
        return dataset, tokenizer, vocab



class MaskedAAN(MaskedSequenceDatasetWrapper):

    def __getitem__(self, index):
        # get item from the warpped dataset
        sample = self.dataset[index]
        xs1, xs2, ys = sample["input_ids1"], sample["input_ids2"], sample["label"]

        # samples are 1D and mask expects [L 1] so expand
        tnsr_dtype = xs1.dtype
        xs1 = xs1.unsqueeze(1).float()
        xs2 = xs2.unsqueeze(1).float()

        # apply masking - the classes are the token values
        # mask token is set to be 0
        xs1, ys1 = self.mask(xs1, sample_classes=xs1)
        xs2, ys2 = self.mask(xs2, sample_classes=xs2)

        # since using an nn.Embedding module can throw away the mask
        xs1 = xs1[..., 0]
        xs2 = xs2[..., 0]

        # move everything to long
        xs1 = xs1.to(tnsr_dtype)
        xs2 = xs2.to(tnsr_dtype)
        ys1 = ys1.to(tnsr_dtype)
        ys2 = ys2.to(tnsr_dtype)

        # return the sample and the target, remove the label key from sample
        sample.pop("label")
        sample["input_ids1"] = xs1
        sample["input_ids2"] = xs2
        sample["label1"] = ys1
        sample["label2"] = ys2

        return sample


class PathFinderSegmentationDataset(torch.utils.data.Dataset):
    """Path Finder dataset with extra supervision."""

    def __init__(self, data_dir, input_transform, prepare_target):
        """
        Args:
            data_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            prepare_target: function for preparing target from path supervision file
        """
        self.data_dir = Path(data_dir).expanduser()
        assert self.data_dir.is_dir(), f"data_dir {str(self.data_dir)} does not exist"
        self.input_transform = input_transform
        self.prepare_target = prepare_target
        samples = []
        path_list = sorted(
            list((self.data_dir / "metadata").glob("*.npy")),
            key=lambda path: int(path.stem),
        )
        assert path_list, "No metadata found"
        for metadata_file in path_list:
            for metadata in np.load(metadata_file).tolist():
                image_path = Path(metadata[0]) / metadata[1]  # 'imgs/0', 'sample_0.png'
                label = int(metadata[3])
                segments_path = Path(metadata[0].replace('imgs/', 'paths/')) / metadata[1].replace('.png', '.pkl')
                # 'paths/0', 'sample_0.pkl'
                samples.append((image_path, label, segments_path))
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label, segments_path = self.samples[idx]
        # https://github.com/pytorch/vision/blob/9b29f3f22783112406d9c1a6db47165a297c3942/torchvision/datasets/folder.py#L247
        with open(self.data_dir / image_path, "rb") as f:
            orig_sample = Image.open(f).convert("L")  # Open in grayscale
        if self.input_transform is not None:
            sample = self.input_transform(orig_sample)

        segmentation = pickle.load(open(self.data_dir / segments_path, 'rb'))
        target = self.prepare_target(segmentation, label, orig_sample)
        return sample, target


class PathFinderSegmentation(SequenceDataset):
    _name_ = "pathfinder_segmentation"
    d_output = 3
    l_output = None

    @property
    def d_input(self):
        if self.pos_info:
            return 4 + 2 if self.all_corners else 1 + 2
        return 4 if self.all_corners else 1

    @property
    def n_tokens(self):
        if self.tokenize:
            return 256

    @property
    def init_defaults(self):
        return {
            "resolution": 128,
            "sequential": True,
            "tokenize": False,
            "autoregressive": False,  # if sequential, pad by L 0's on right and take last L outputs
            "pool": 1,
            "all_corners": False,  # stack 0,90,180,270 degree rotations
            "val_split": 0.1,
            "test_split": 0.1,
            "seed": 42,  # Controls the train/val/test split
            "pos_info": False,
        }

    def init(self):
        if self.data_dir is None:
            self.data_dir = (
                    default_data_path / self._name_ / f"pathfinder{self.resolution}_segmentation"
            )
        if self.autoregressive:
            assert self.sequential
            self.l_output = self.resolution ** 2

    def rotations(self, x, dim=-3):
        assert x.shape[-2] == x.shape[-1], 'must be square'
        assert x.ndim >= 3
        rotations = [x] + [torchvision.transforms.functional.rotate(x, 90 * i) for i in range(1, 4)]
        return torch.cat(rotations, dim=dim)

    def concat_pos(self, x):
        """ append in last dim the position info of second-last dim
        """
        L = x.shape[-2]  # (... L d)
        pos = (2 * np.pi * torch.arange(L, device=x.device) / L).view(-1, 1)
        cos = torch.zeros_like(x[..., :1]) + pos.cos()
        sin = torch.zeros_like(x[..., :1]) + pos.sin()
        return torch.cat((x, cos, sin), dim=-1)  # (... L d+2)

    def zero_pad_right(self, x, dim=-2):
        assert dim < 0
        L = x.shape[dim]
        assert self.l_output == L
        return F.pad(x, (0, 0) * abs(dim + 1) + (0, L))

    def input_transform(self):
        transform_list = [torchvision.transforms.ToTensor()]
        if self.pool > 1:
            transform_list.append(
                Reduce(
                    "1 (h h2) (w w2) -> 1 h w",
                    "mean",
                    h2=self.pool,
                    w2=self.pool,
                )
            )
        if self.tokenize:
            transform_list.append(
                torchvision.transforms.Lambda(lambda x: (x * 255).long())
            )
        else:
            transform_list.append(torchvision.transforms.Normalize(mean=0.5, std=0.5))
        if self.all_corners:
            transform_list.append(self.rotations)  # (4 h w)
        if self.sequential:
            # If tokenize, it makes more sense to get rid of the channel dimension
            transform_list.append(
                Rearrange("1 h w -> (h w)")
                if self.tokenize and not self.all_corners
                else Rearrange("r h w -> (h w) r")
            )
            if not self.tokenize and self.pos_info:
                transform_list.append(self.concat_pos)
            if self.autoregressive:
                transform_list.append(
                    partial(self.zero_pad_right, dim=-1)
                    if self.tokenize
                    else partial(self.zero_pad_right, dim=-2)
                )
        return torchvision.transforms.Compose(transform_list)

    def prepare_target(self, d, label=None, orig_sample=None):
        # d.keys(): 'segs', 'origin_tips', 'terminal_tips', 'marker_indices', 'image_size'
        [origin_mark_idx, terminal_mark_idx] = d['marker_indices']

        if label is not None:
            assert label == (origin_mark_idx == terminal_mark_idx)

        paths = []
        for i, path_inds in enumerate(d['segs']):
            path = np.zeros(d['image_size'], dtype=np.int16)
            path[path_inds] = 1 + (origin_mark_idx == terminal_mark_idx == i)
            paths.append(path)
        # 0: not on a long path, 1: on a long path but not any connecting path, 2: on a connecting path
        label_mask = torch.LongTensor(np.maximum(*paths))

        # sanity
        # label_mask = (torchvision.transforms.ToTensor()(orig_sample) > 0).long().squeeze(0)

        if self.sequential:
            label_mask = rearrange(label_mask, "h w -> (h w)")
        return label_mask

    def prepare_data(self):
        if not self.data_dir.is_dir():
            raise FileNotFoundError(
                f"""
            Directory {str(self.data_dir)} not found.
            To get the dataset, generate pathfinder data using /src/dataloaders/prepare/pathfinder .
            Then point data_dir to the pathfinderX_segmentation directory, where X is either 128, or 256.
            """
            )

    def setup(self, stage=None):
        if stage == "test" and hasattr(self, "dataset_test"):
            return

        # [2021-08-18] TD: I ran into RuntimeError: Too many open files.
        # https://github.com/pytorch/pytorch/issues/11201
        torch.multiprocessing.set_sharing_strategy("file_system")
        dataset = PathFinderSegmentationDataset(self.data_dir, self.input_transform(), self.prepare_target)
        len_dataset = len(dataset)
        print(f'Total num of samples = {len_dataset}')
        val_len = int(self.val_split * len_dataset)
        test_len = int(self.test_split * len_dataset)
        train_len = len_dataset - val_len - test_len
        (
            self.dataset_train,
            self.dataset_val,
            self.dataset_test,
        ) = torch.utils.data.random_split(
            dataset,
            [train_len, val_len, test_len],
            generator=torch.Generator().manual_seed(self.seed),
        )

