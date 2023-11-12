"""Implementation of basic benchmark datasets used in S4 experiments: MNIST, CIFAR10 and Speech Commands."""
import numpy as np
import torch
import torchvision
from einops.layers.torch import Rearrange
from src.utils import permutations

from src.dataloaders.base import default_data_path, ImageResolutionSequenceDataset, ResolutionSequenceDataset, SequenceDataset, MaskedSequenceDatasetWrapper


class MNIST(SequenceDataset):
    _name_ = "mnist"
    d_input = 1
    d_output = 10
    l_output = 0
    L = 784

    @property
    def init_defaults(self):
        return {
            "permute": True,
            "val_split": 0.1,
            "seed": 42,  # For train/val split
        }

    def setup(self):
        self.data_dir = self.data_dir or default_data_path / self._name_

        transform_list = [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(lambda x: x.view(self.d_input, self.L).t()),
        ]  # (L, d_input)
        if self.permute:
            # below is another permutation that other works have used
            # permute = np.random.RandomState(92916)
            # permutation = torch.LongTensor(permute.permutation(784))
            permutation = permutations.bitreversal_permutation(self.L)
            transform_list.append(
                torchvision.transforms.Lambda(lambda x: x[permutation])
            )
        # TODO does MNIST need normalization?
        # torchvision.transforms.Normalize((0.1307,), (0.3081,)) # normalize inputs
        transform = torchvision.transforms.Compose(transform_list)
        self.dataset_train = torchvision.datasets.MNIST(
            self.data_dir,
            train=True,
            download=True,
            transform=transform,
        )
        self.dataset_test = torchvision.datasets.MNIST(
            self.data_dir,
            train=False,
            transform=transform,
        )
        self.split_train_val(self.val_split)

    def __str__(self):
        return f"{'p' if self.permute else 's'}{self._name_}"


class CIFAR10(ImageResolutionSequenceDataset):
    _name_ = "cifar"
    d_output = 10
    l_output = 0

    @property
    def init_defaults(self):
        return {
            "permute": None,
            "grayscale": False,
            "tokenize": False,  # if grayscale, tokenize into discrete byte inputs
            "augment": False,
            "cutout": False,
            "rescale": None,
            "random_erasing": False,
            "val_split": 0.1,
            "seed": 42,  # For validation split
        }

    @property
    def d_input(self):
        if self.grayscale:
            if self.tokenize:
                return 256
            else:
                return 1
        else:
            assert not self.tokenize
            return 3

    def setup(self):
        img_size = 32
        if self.rescale:
            img_size //= self.rescale

        if self.grayscale:
            preprocessors = [
                torchvision.transforms.Grayscale(),
                torchvision.transforms.ToTensor(),
            ]
            permutations_list = [
                torchvision.transforms.Lambda(
                    lambda x: x.view(1, img_size * img_size).t()
                )  # (L, d_input)
            ]

            if self.tokenize:
                preprocessors.append(
                    torchvision.transforms.Lambda(lambda x: (x * 255).long())
                )
                permutations_list.append(Rearrange("l 1 -> l"))
            else:
                preprocessors.append(
                    torchvision.transforms.Normalize(
                        mean=122.6 / 255.0, std=61.0 / 255.0
                    )
                )
        else:
            preprocessors = [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)
                ),
            ]
            permutations_list = [
                torchvision.transforms.Lambda(
                    Rearrange("z h w -> (h w) z", z=3, h=img_size, w=img_size)
                )  # (L, d_input)
            ]

        # Permutations and reshaping
        if self.permute == "br":
            permutation = permutations.bitreversal_permutation(img_size * img_size)
            print("bit reversal", permutation)
            permutations_list.append(torchvision.transforms.Lambda(lambda x: x[permutation]))
        elif self.permute == "snake":
            permutation = permutations.snake_permutation(img_size, img_size)
            print("snake", permutation)
            permutations_list.append(torchvision.transforms.Lambda(lambda x: x[permutation]))
        elif self.permute == "hilbert":
            permutation = permutations.hilbert_permutation(img_size)
            print("hilbert", permutation)
            permutations_list.append(torchvision.transforms.Lambda(lambda x: x[permutation]))
        elif self.permute == "transpose":
            permutation = permutations.transpose_permutation(img_size, img_size)
            transform = torchvision.transforms.Lambda(
                lambda x: torch.cat([x, x[permutation]], dim=-1)
            )
            permutations_list.append(transform)
        elif self.permute == "2d":  # h, w, c
            permutation = torchvision.transforms.Lambda(
                    Rearrange("(h w) c -> h w c", h=img_size, w=img_size)
                )
            permutations_list.append(permutation)
        elif self.permute == "2d_transpose":  # c, h, w
            permutation = torchvision.transforms.Lambda(
                    Rearrange("(h w) c -> c h w", h=img_size, w=img_size)
                )
            permutations_list.append(permutation)

        # Augmentation
        if self.augment:
            augmentations = [
                torchvision.transforms.RandomCrop(
                    img_size, padding=4, padding_mode="symmetric"
                ),
                torchvision.transforms.RandomHorizontalFlip(),
            ]

            post_augmentations = []
            if self.cutout:
                post_augmentations.append(Cutout(1, img_size // 2))
                pass
            if self.random_erasing:
                # augmentations.append(RandomErasing())
                pass
        else:
            augmentations, post_augmentations = [], []
        transforms_train = (
            augmentations + preprocessors + post_augmentations + permutations_list
        )
        transforms_eval = preprocessors + permutations_list

        transform_train = torchvision.transforms.Compose(transforms_train)
        transform_eval = torchvision.transforms.Compose(transforms_eval)
        self.dataset_train = torchvision.datasets.CIFAR10(
            f"{default_data_path}/{self._name_}",
            train=True,
            download=True,
            transform=transform_train,
        )
        self.dataset_test = torchvision.datasets.CIFAR10(
            f"{default_data_path}/{self._name_}", train=False, transform=transform_eval
        )

        if self.rescale:
            print(f"Resizing all images to {img_size} x {img_size}.")
            self.dataset_train.data = self.dataset_train.data.reshape((self.dataset_train.data.shape[0], 32 // self.rescale, self.rescale, 32 // self.rescale, self.rescale, 3)).max(4).max(2).astype(np.uint8)
            self.dataset_test.data = self.dataset_test.data.reshape((self.dataset_test.data.shape[0], 32 // self.rescale, self.rescale, 32 // self.rescale, self.rescale, 3)).max(4).max(2).astype(np.uint8)

        self.split_train_val(self.val_split)

    def __str__(self):
        return f"{'p' if self.permute else 's'}{self._name_}"

class MaskedCIFAR10(MaskedSequenceDatasetWrapper):
    """
    A custom masked dataset wrapper for the CIFAR10 dataset.
    outputs masked samples after applying transforms
    """

    def __getitem__(self, index):
        source_sample, source_target = self.dataset.data[index], self.dataset.targets[index]
        sample = torchvision.transforms.ToPILImage()(source_sample)
        sample_classes = None

        # if loss is cross entropy need to extract the source pixel values in [0 255]
        if self.lm_loss == 'ce':
            data_to_tensor = torchvision.transforms.Compose([torchvision.transforms.Grayscale(),
                                                             torchvision.transforms.ToTensor()])
            sample_classes = (data_to_tensor(sample) * 255).long()


        # apply source transforms on the sample
        sample, _ = self.transforms(sample, source_target)
        if self.lm_loss == 'ce':
            sample_classes = sample_classes.reshape(sample.shape)

        # mask out tokens randomly
        sample, target = self.mask(sample, sample_classes, allow_features=True)

        return sample, target

class CIFAR10LM(ImageResolutionSequenceDataset):
    _name_ = "cifar_lm"

    @property
    def l_output(self):
        return None if self.mlm_prob or self.causal_lm else 0

    @property
    def init_defaults(self):
        return {
            "permute": None,
            "grayscale": False,
            "tokenize": False,  # if grayscale, tokenize into discrete byte inputs
            "augment": False,
            "cutout": False,
            "rescale": None,
            "random_erasing": False,
            "val_split": 0.1,
            "seed": 42,  # For validation split

            # Masked LM parameters
            "mlm_prob": 0,       # self pre-training : prob of masking out a pixel
            "causal_lm": False,  # self pre-training : predict next pixel in sequence
            "lm_loss": 'l1',     # self pre-training loss: cross-entropy (ce) or l1
            "ignore_val": True,  # self pre-training: ignore the loss at masked pixels
            "span_masking": False,
            "span_length": 0,
        }

    @property
    def d_input(self):
        if self.grayscale:
            if self.tokenize:
                # return 256
                raise NotImplementedError("Tokenization is not supported for now for CIFAR10LM task")
            else:
                return 1 if self.causal_lm else 2
        else:
            return 3 if self.causal_lm else 4

    @property
    def d_output(self):
        if self.lm_loss == 'ce':
            return 256
        else:
            return 1 if self.grayscale else 3

    def setup(self):
        # if not self.grayscale:
        #     return ValueError("Only grayscale is supported for now for masked CIFAR10")

        img_size = 32
        if self.rescale:
            img_size //= self.rescale

        ## prepare preprocessing and permutations to be applied to each image

        if self.grayscale:
            preprocessors = [
                torchvision.transforms.Grayscale(),
                torchvision.transforms.ToTensor(),
            ]
            permutations_list = [
                torchvision.transforms.Lambda(
                    lambda x: x.view(1, img_size * img_size).t()
                )  # (L, d_input)
            ]

            if self.tokenize:
                preprocessors.append(
                    torchvision.transforms.Lambda(lambda x: (x * 255).long())
                )
                permutations_list.append(Rearrange("l 1 -> l"))
            else:
                preprocessors.append(
                    torchvision.transforms.Normalize(
                        mean=122.6 / 255.0, std=61.0 / 255.0
                    )
                )
        else:
            preprocessors = [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)
                ),
            ]
            permutations_list = [
                torchvision.transforms.Lambda(
                    Rearrange("z h w -> (h w) z", z=3, h=img_size, w=img_size)
                )  # (L, d_input)
            ]

        # Permutations and reshaping
        if self.permute == "br":
            permutation = permutations.bitreversal_permutation(img_size * img_size)
            print("bit reversal", permutation)
            permutations_list.append(torchvision.transforms.Lambda(lambda x: x[permutation]))
        elif self.permute == "snake":
            permutation = permutations.snake_permutation(img_size, img_size)
            print("snake", permutation)
            permutations_list.append(torchvision.transforms.Lambda(lambda x: x[permutation]))
        elif self.permute == "hilbert":
            permutation = permutations.hilbert_permutation(img_size)
            print("hilbert", permutation)
            permutations_list.append(torchvision.transforms.Lambda(lambda x: x[permutation]))
        elif self.permute == "transpose":
            permutation = permutations.transpose_permutation(img_size, img_size)
            transform = torchvision.transforms.Lambda(
                lambda x: torch.cat([x, x[permutation]], dim=-1)
            )
            permutations_list.append(transform)
        elif self.permute == "2d":  # h, w, c
            permutation = torchvision.transforms.Lambda(
                    Rearrange("(h w) c -> h w c", h=img_size, w=img_size)
                )
            permutations_list.append(permutation)
        elif self.permute == "2d_transpose":  # c, h, w
            permutation = torchvision.transforms.Lambda(
                    Rearrange("(h w) c -> c h w", h=img_size, w=img_size)
                )
            permutations_list.append(permutation)

        ## add data augmentations

        # Augmentation
        if self.augment:
            augmentations = [
                torchvision.transforms.RandomCrop(
                    img_size, padding=4, padding_mode="symmetric"
                ),
                torchvision.transforms.RandomHorizontalFlip(),
            ]

            post_augmentations = []
            if self.cutout:
                post_augmentations.append(Cutout(1, img_size // 2))
                pass
            if self.random_erasing:
                # augmentations.append(RandomErasing())
                pass
        else:
            augmentations, post_augmentations = [], []

        ## construct datasets

        transforms_train = (
            augmentations + preprocessors + post_augmentations + permutations_list
        )
        transforms_eval = preprocessors + permutations_list

        transform_train = torchvision.transforms.Compose(transforms_train)
        transform_eval = torchvision.transforms.Compose(transforms_eval)
        self.dataset_train = torchvision.datasets.CIFAR10(
            f"{default_data_path}/{self._name_}",
            train=True,
            download=True,
            transform=transform_train,
        )
        self.dataset_test = torchvision.datasets.CIFAR10(
            f"{default_data_path}/{self._name_}", train=False, transform=transform_eval
        )

        if self.rescale:
            print(f"Resizing all images to {img_size} x {img_size}.")
            self.dataset_train.data = self.dataset_train.data.reshape((self.dataset_train.data.shape[0], 32 // self.rescale, self.rescale, 32 // self.rescale, self.rescale, 3)).max(4).max(2).astype(np.uint8)
            self.dataset_test.data = self.dataset_test.data.reshape((self.dataset_test.data.shape[0], 32 // self.rescale, self.rescale, 32 // self.rescale, self.rescale, 3)).max(4).max(2).astype(np.uint8)

        # Change dataset to a masked dataset
        self.dataset_train = MaskedCIFAR10(dataset=self.dataset_train,
                                           mlm_prob=self.mlm_prob,
                                           causal_lm=self.causal_lm,
                                           lm_loss=self.lm_loss,
                                           ignore_val=self.ignore_val)
        self.dataset_test = MaskedCIFAR10(dataset=self.dataset_test,
                                          mlm_prob=self.mlm_prob,
                                          causal_lm=self.causal_lm,
                                          lm_loss=self.lm_loss,
                                          ignore_val=self.ignore_val)

        self.split_train_val(self.val_split)

    def __str__(self):
        return f"{'p' if self.permute else 's'}{self._name_}"


class SpeechCommands(ResolutionSequenceDataset):
    _name_ = "sc"

    @property
    def init_defaults(self):
        return {
            "mfcc": False,
            "dropped_rate": 0.0,
            "length": 16000,
            "all_classes": False,
        }

    @property
    def d_input(self):
        _d_input = 20 if self.mfcc else 1
        _d_input += 1 if self.dropped_rate > 0.0 else 0
        return _d_input

    @property
    def d_output(self):
        return 10 if not self.all_classes else 35

    @property
    def l_output(self):
        return 0

    @property
    def L(self):
        return 161 if self.mfcc else self.length


    def setup(self):
        self.data_dir = self.data_dir or default_data_path # TODO make same logic as other classes

        from src.dataloaders.datasets.sc import _SpeechCommands

        # TODO refactor with data_dir argument
        self.dataset_train = _SpeechCommands(
            partition="train",
            length=self.L,
            mfcc=self.mfcc,
            sr=1,
            dropped_rate=self.dropped_rate,
            path=self.data_dir,
            all_classes=self.all_classes,
        )

        self.dataset_val = _SpeechCommands(
            partition="val",
            length=self.L,
            mfcc=self.mfcc,
            sr=1,
            dropped_rate=self.dropped_rate,
            path=self.data_dir,
            all_classes=self.all_classes,
        )

        self.dataset_test = _SpeechCommands(
            partition="test",
            length=self.L,
            mfcc=self.mfcc,
            sr=1,
            dropped_rate=self.dropped_rate,
            path=self.data_dir,
            all_classes=self.all_classes,
        )


class MaskedSc(MaskedSequenceDatasetWrapper):

    def __getitem__(self, item):
        sample, label = self.dataset.__getitem__(item)

        sample, target = self.mask(sample)

        return sample, target

class SpeechCommandsLM(ResolutionSequenceDataset):
    _name_ = "sc_lm"

    @property
    def init_defaults(self):
        return {
            "mfcc": False,
            "dropped_rate": 0.0,
            "length": 16000,
            "all_classes": False,

            # Masked LM parameters
            "mlm_prob": 0,  # self pre-training : prob of masking out a pixel
            "causal_lm": False,  # self pre-training : predict next pixel in sequence
            "lm_loss": 'l1',  # self pre-training loss: cross-entropy (ce) or l1
            "ignore_val": True,  # self pre-training: ignore the loss at masked pixels
            "span_masking": False,
            "span_length": 0,
        }

    @property
    def d_input(self):
        return 1 if self.causal_lm else 2

    @property
    def d_output(self):
        return 1

    @property
    def l_output(self):
        return None

    @property
    def L(self):
        return 161 if self.mfcc else self.length


    def setup(self):
        assert not self.mfcc, "MFCC not supported for LM on SC"
        assert self.dropped_rate == 0.0, "Dropped rate not supported for LM on SC"
        assert self.lm_loss != 'ce', "Cross-entropy not supported for LM on SC"

        self.data_dir = self.data_dir or default_data_path # TODO make same logic as other classes

        from src.dataloaders.datasets.sc import _SpeechCommands

        # TODO refactor with data_dir argument
        self.dataset_train = _SpeechCommands(
            partition="train",
            length=self.L,
            mfcc=self.mfcc,
            sr=1,
            dropped_rate=self.dropped_rate,
            path=self.data_dir,
            all_classes=self.all_classes,
        )

        self.dataset_val = _SpeechCommands(
            partition="val",
            length=self.L,
            mfcc=self.mfcc,
            sr=1,
            dropped_rate=self.dropped_rate,
            path=self.data_dir,
            all_classes=self.all_classes,
        )

        self.dataset_test = _SpeechCommands(
            partition="test",
            length=self.L,
            mfcc=self.mfcc,
            sr=1,
            dropped_rate=self.dropped_rate,
            path=self.data_dir,
            all_classes=self.all_classes,
        )

        self.dataset_train = MaskedSc(dataset=self.dataset_train,
                                        mlm_prob=self.mlm_prob,
                                        causal_lm=self.causal_lm,
                                        lm_loss=self.lm_loss,
                                        ignore_val=self.ignore_val)

        self.dataset_test = MaskedSc(dataset=self.dataset_test,
                                        mlm_prob=self.mlm_prob,
                                        causal_lm=self.causal_lm,
                                        lm_loss=self.lm_loss,
                                        ignore_val=self.ignore_val)
        self.dataset_val = MaskedSc(dataset=self.dataset_val,
                                        mlm_prob=self.mlm_prob,
                                        causal_lm=self.causal_lm,
                                        lm_loss=self.lm_loss,
                                        ignore_val=self.ignore_val)


