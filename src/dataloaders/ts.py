"""Time series datasets, especially for medical time series"""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.dataloaders.base import default_data_path, SequenceDataset, deprecated, MaskedSequenceDatasetWrapper
from src.models.functional import masking


class BIDMC(SequenceDataset):
    """BIDMC datasets for Respiratory Rate / Heart Rate / Oxygen Saturation regression"""

    _name_ = "bidmc"
    d_input = 2

    @property
    def d_output(self):
        return 2 if self.prediction else 1

    @property
    def l_output(self):
        return 4000 if self.prediction else 0

    @property
    def init_defaults(self):
        return {
            "target": "RR",  # 'RR' | 'HR' | 'SpO2'
            "prediction": False,
            "reshuffle": True,
            "rescale_targets": False
        }

    def setup(self):
        self.data_dir = self.data_dir or default_data_path / self._name_

        split = "reshuffle" if self.reshuffle else "original"
        # X: (dataset_size, length, d_input)
        # y: (dataset_size)
        X_train = np.load(self.data_dir / self.target / split / "trainx.npy")
        y_train = np.load(self.data_dir / self.target / split / "trainy.npy")
        X_val = np.load(self.data_dir / self.target / split / "validx.npy")
        y_val = np.load(self.data_dir / self.target / split / "validy.npy")
        X_test = np.load(self.data_dir / self.target / split / "testx.npy")
        y_test = np.load(self.data_dir / self.target / split / "testy.npy")

        if self.rescale_targets:
            y_train, y_val, y_test = self._rescale_target_values(y_train, y_val, y_test)

        if self.prediction:
            y_train = np.pad(X_train[:, 1:, :], ((0, 0), (0, 1), (0, 0)))
            y_val = np.pad(X_val[:, 1:, :], ((0, 0), (0, 1), (0, 0)))
            y_test = np.pad(X_test[:, 1:, :], ((0, 0), (0, 1), (0, 0)))

        self.dataset_train = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_train), torch.FloatTensor(y_train)
        )

        self.dataset_val = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_val), torch.FloatTensor(y_val)
        )

        self.dataset_test = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_test), torch.FloatTensor(y_test)
        )

    def __str__(self):
        split = "reshuffle" if self.reshuffle else "original"
        return f"BIDMC{self.target}_{split}"

    def _rescale_target_values(self, y_train, y_val, y_test):
        max_val = np.max(np.abs(y_train))
        min_val = np.min(np.abs(y_train))

        # rescale to [-1, 1]
        y_train = (y_train - min_val) / (max_val - min_val) * 2 - 1
        y_val = (y_val - min_val) / (max_val - min_val) * 2 - 1
        y_test = (y_test - min_val) / (max_val - min_val) * 2 - 1


        return y_train, y_val, y_test


class MaskedBIDMC(MaskedSequenceDatasetWrapper):

    def __getitem__(self, item):
        sample, label = self.dataset.__getitem__(item)

        sample, target = self.mask(sample)

        return sample, target

    def mask(self, sample, sample_classes=None):
        """
        Applies masking to a given sample.
        Args:
            sample (torch.Tensor): The input sample to be masked.
                Should be a 2-dimensional tensor with shape [L, 1], where L is the sequence length.
                Sample has to be a float tensor.
            sample_classes (torch.Tensor, optional): Target classes for the sample of each token.
                e.g. for images the pixel values between [0 255] of each pixel
                Should be provided when using cross-entropy loss.
        Returns:
            tuple: A tuple containing the masked sample and the target tensor for the masked sample.
                - sample (torch.Tensor): The masked sample after applying the masking operation.
                - target (torch.Tensor): The target tensor for the masked sample,
                    which corresponds to the original sample with masked tokens or transformed labels.
        """

        mlm_prob = self.mlm_prob
        causal_lm = self.causal_lm
        lm_loss = self.lm_loss
        lm_ignore_val = self.ignore_val
        lm_ignore_none_masked = self.ignore_non_masked
        span_masking = self.span_masking
        span_length = self.span_length

        assert not lm_loss == 'ce', "Cross entropy loss not supported for BIDMCLM yet"
        assert not span_masking, "Span masking not supported for BIDMCLM yet"

        if lm_loss == 'ce' and sample_classes is None:
            raise ValueError("sample_classes should be provided when using cross entropy loss but got None")

        if causal_lm:
            if lm_loss == 'l1':
                target = F.pad(sample[1:], (0, 0, 0, 1), value=lm_ignore_val)  # [L 1]
                assert sample.shape == target.shape

            elif lm_loss == 'ce':
                target = F.pad(sample_classes[1:], (0, 0, 0, 1), value=lm_ignore_val)
                target = target.squeeze(-1)  # [L]

        else:
            if span_masking:
                mask = masking.span_mask(x=sample, masking_ratio=mlm_prob, span_length=span_length)  # [L 1]
            else:
                mask = (torch.rand_like(sample[..., :1]) < mlm_prob).float()  # [L 1]

            masked_sample = torch.cat((sample * (1 - mask), mask), dim=-1)  # [L 2]  # 0 for unmasked, 1 for masked

            if lm_loss == 'l1':
                target = sample.clone()
                if lm_ignore_none_masked:
                    target = target * mask + lm_ignore_val * (1 - mask)  # loss not computed at lm_ignore_val
                target = target.squeeze(-1)  # [L]

            elif lm_loss == 'ce':
                mask = mask.long()
                target = sample_classes.clone()
                if lm_ignore_none_masked:
                    target = target * mask + lm_ignore_val * (1 - mask)  # loss not computed at lm_ignore_val
                target = target.squeeze(-1)  # [L]

            sample = masked_sample

        return sample, target


class BIDMCLM(SequenceDataset):
    """BIDMC datasets for Respiratory Rate / Heart Rate / Oxygen Saturation regression"""

    _name_ = "bidmc_lm"
    d_input = 2

    @property
    def d_output(self):
        return 2

    @property
    def d_input(self):
        return 3  # additional dim for mask

    @property
    def l_output(self):
        return None

    @property
    def init_defaults(self):
        return {
            "target": "RR",  # 'RR' | 'HR' | 'SpO2'
            "prediction": False,
            "reshuffle": True,

            # Masked LM parameters
            "mlm_prob": 0,  # self pre-training : prob of masking out a pixel
            "causal_lm": False,  # self pre-training : predict next pixel in sequence
            "lm_loss": 'l1',  # self pre-training loss: cross-entropy (ce) or l1
            "ignore_val": True,  # self pre-training: ignore the loss at masked pixels
            "span_masking": False,
            "span_length": 0,
        }

    def setup(self):
        assert not self.prediction, "Prediction not supported for BIDMCLM"

        self.data_dir = default_data_path / 'bidmc'  # self.data_dir or default_data_path / self._name_

        split = "reshuffle" if self.reshuffle else "original"
        # X: (dataset_size, length, d_input)
        # y: (dataset_size)
        X_train = np.load(self.data_dir / self.target / split / "trainx.npy")
        y_train = np.load(self.data_dir / self.target / split / "trainy.npy")
        X_val = np.load(self.data_dir / self.target / split / "validx.npy")
        y_val = np.load(self.data_dir / self.target / split / "validy.npy")
        X_test = np.load(self.data_dir / self.target / split / "testx.npy")
        y_test = np.load(self.data_dir / self.target / split / "testy.npy")

        # if self.prediction:
        #     y_train = np.pad(X_train[:, 1:, :], ((0, 0), (0, 1), (0, 0)))
        #     y_val = np.pad(X_val[:, 1:, :], ((0, 0), (0, 1), (0, 0)))
        #     y_test = np.pad(X_test[:, 1:, :], ((0, 0), (0, 1), (0, 0)))

        self.dataset_train = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_train), torch.FloatTensor(y_train)
        )

        self.dataset_val = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_val), torch.FloatTensor(y_val)
        )

        self.dataset_test = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_test), torch.FloatTensor(y_test)
        )

        self.dataset_train = MaskedBIDMC(dataset=self.dataset_train,
                                         mlm_prob=self.mlm_prob,
                                         causal_lm=self.causal_lm,
                                         lm_loss=self.lm_loss,
                                         ignore_val=self.ignore_val,
                                         span_masking=self.span_masking,
                                         span_length=self.span_length)
        self.dataset_val = MaskedBIDMC(dataset=self.dataset_val,
                                       mlm_prob=self.mlm_prob,
                                       causal_lm=self.causal_lm,
                                       lm_loss=self.lm_loss,
                                       ignore_val=self.ignore_val,
                                       span_masking=self.span_masking,
                                       span_length=self.span_length)
        self.dataset_test = MaskedBIDMC(dataset=self.dataset_test,
                                        mlm_prob=self.mlm_prob,
                                        causal_lm=self.causal_lm,
                                        lm_loss=self.lm_loss,
                                        ignore_val=self.ignore_val,
                                        span_masking=self.span_masking,
                                        span_length=self.span_length)

    def __str__(self):
        split = "reshuffle" if self.reshuffle else "original"
        return f"BIDMC{self.target}_{split}"
