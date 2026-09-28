from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torchaudio
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from transformers import AutoFeatureExtractor


class RADARDataset(Dataset):
    """RADAR dev/eval dataset backed by the challenge label file.

    Expected layout:
        RADAR2026-dev/
          flac/<utt_id>.flac
          label_RADAR2026-dev.txt

    Protocol rows are expected to start with:
        <utt_id> <duration> <bonafide|spoof> ...
    """

    LABEL_MAP = {"bonafide": 0, "spoof": 1, "0": 0, "1": 1}

    def __init__(
        self,
        root_dir: str,
        protocol_file: str = "label_RADAR2026-dev.txt",
        audio_dir: str = "flac",
        audio_ext: str = ".flac",
    ) -> None:
        self.root_dir = Path(root_dir)
        self.protocol_file = Path(protocol_file)
        self.audio_dir = Path(audio_dir)
        self.audio_ext = audio_ext
        self.samples = self._load_samples()

    def _resolve_protocol_path(self) -> Path:
        if self.protocol_file.is_absolute():
            return self.protocol_file
        return self.root_dir / self.protocol_file

    def _resolve_audio_dir(self) -> Path:
        if self.audio_dir.is_absolute():
            return self.audio_dir
        return self.root_dir / self.audio_dir

    def _parse_label(self, raw_label: str, protocol_path: Path) -> int:
        label_token = raw_label.strip().lower()
        if label_token not in self.LABEL_MAP:
            raise ValueError(
                f"Invalid label '{raw_label}' in {protocol_path}. "
                "Expected one of: bonafide, spoof, 0, 1."
            )
        return self.LABEL_MAP[label_token]

    def _load_samples(self) -> List[Tuple[str, int]]:
        protocol_path = self._resolve_protocol_path()
        audio_dir = self._resolve_audio_dir()

        if not protocol_path.exists():
            raise FileNotFoundError(f"RADAR protocol not found: {protocol_path}")
        if not audio_dir.exists():
            raise FileNotFoundError(f"RADAR audio directory not found: {audio_dir}")

        samples: List[Tuple[str, int]] = []
        missing_audio = 0
        with protocol_path.open("r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                parts = line.strip().split()
                if not parts:
                    continue
                if len(parts) < 3:
                    raise ValueError(f"Invalid RADAR protocol row {line_idx}: {line.strip()}")

                utt_id = parts[0]
                label = self._parse_label(parts[2], protocol_path)
                audio_path = audio_dir / f"{utt_id}{self.audio_ext}"
                if not audio_path.exists():
                    missing_audio += 1
                    continue
                samples.append((str(audio_path), label))

        print(
            f"Loaded {len(samples)} RADAR samples from {protocol_path}"
            + (f" ({missing_audio} missing audio files skipped)" if missing_audio else "")
        )

        if not samples:
            raise RuntimeError(f"No RADAR samples were loaded from protocol {protocol_path}")

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        audio_path, label = self.samples[idx]
        waveform, sample_rate = torchaudio.load(audio_path)

        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        waveform = waveform.squeeze(0)
        return waveform, sample_rate, label


class RADARDataModule(LightningDataModule):
    """LightningDataModule for using RADAR dev as a test set."""

    def __init__(
        self,
        data_dir: str,
        protocol_file: str = "label_RADAR2026-dev.txt",
        audio_dir: str = "flac",
        audio_ext: str = ".flac",
        batch_size: int = 32,
        num_workers: int = 4,
        persistent_workers: bool = False,
        pin_memory: bool = False,
        model_name: str = "facebook/wav2vec2-xls-r-300m",
        target_sample_rate: int = 16_000,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size
        self.feature_extractor = None

    @property
    def num_classes(self) -> int:
        return 2

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by world size ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        if self.data_test is None:
            self.data_test = RADARDataset(
                root_dir=self.hparams.data_dir,
                protocol_file=self.hparams.protocol_file,
                audio_dir=self.hparams.audio_dir,
                audio_ext=self.hparams.audio_ext,
            )
            self.data_train = self.data_test
            self.data_val = self.data_test

        if self.feature_extractor is None:
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(
                self.hparams.model_name,
                trust_remote_code=True,
            )

    def _collate_fn(
        self,
        batch: List[Tuple[torch.Tensor, int, int]],
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        waveforms: List[torch.Tensor] = []
        labels: List[int] = []

        for waveform, sample_rate, label in batch:
            if sample_rate != self.hparams.target_sample_rate:
                waveform = torchaudio.functional.resample(
                    waveform,
                    sample_rate,
                    self.hparams.target_sample_rate,
                )
            waveforms.append(waveform)
            labels.append(label)

        if self.hparams.model_name == "labhamlet/gramt-binaural-time":
            waveforms = [torch.stack([w, w, w, w], dim=0) for w in waveforms]
        else:
            waveforms = [w.numpy() for w in waveforms]

        features = self.feature_extractor(
            waveforms,
            sampling_rate=self.hparams.target_sample_rate,
            padding=True,
            return_tensors="pt",
        )
        targets = torch.tensor(labels, dtype=torch.long)
        return features, targets

    def train_dataloader(self) -> DataLoader[Any]:
        return self.test_dataloader()

    def val_dataloader(self) -> DataLoader[Any]:
        return self.test_dataloader()

    def test_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            persistent_workers=self.hparams.persistent_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=partial(self._collate_fn),
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        pass

    def state_dict(self) -> Dict[Any, Any]:
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        pass


if __name__ == "__main__":
    dm = RADARDataModule(data_dir="/workspace/datasets/RADAR_dev/RADAR2026-dev")
    dm.setup(stage="test")
    print(f"Loaded RADAR test samples: {len(dm.data_test)}", flush=True)
