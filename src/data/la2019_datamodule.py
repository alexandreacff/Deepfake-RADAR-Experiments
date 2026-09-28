from functools import partial
import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torchaudio
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from transformers import AutoFeatureExtractor

from src.data.components import data_augmentation as aug


class LA2019Dataset(Dataset):
    """ASVspoof 2019 LA split dataset backed by protocol files."""

    LABEL_MAP = {"bonafide": 0, "spoof": 1}

    def __init__(self, root_dir: str, split: str) -> None:
        self.root_dir = Path(root_dir)
        self.split = split
        self.samples = self._load_samples()

    def _protocol_file(self) -> Path:
        protocol_dir = self.root_dir / "ASVspoof2019_LA_cm_protocols"
        split_to_protocol = {
            "train": "ASVspoof2019.LA.cm.train.trn.txt",
            "dev": "ASVspoof2019.LA.cm.dev.trl.txt",
            "eval": "ASVspoof2019.LA.cm.eval.trl.txt",
        }
        return protocol_dir / split_to_protocol[self.split]

    def _audio_dir(self) -> Path:
        split_to_audio = {
            "train": "ASVspoof2019_LA_train/flac",
            "dev": "ASVspoof2019_LA_dev/flac",
            "eval": "ASVspoof2019_LA_eval/flac",
        }
        return self.root_dir / split_to_audio[self.split]

    def _load_samples(self) -> List[Tuple[str, int]]:
        protocol_path = self._protocol_file()
        audio_dir = self._audio_dir()

        if not protocol_path.exists():
            raise FileNotFoundError(f"Protocol not found: {protocol_path}")
        if not audio_dir.exists():
            raise FileNotFoundError(f"Audio directory not found: {audio_dir}")

        samples: List[Tuple[str, int]] = []
        with protocol_path.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue

                utt_id = parts[1]
                label_token = parts[-1].lower()
                if label_token not in self.LABEL_MAP:
                    continue

                label = self.LABEL_MAP[label_token]
                audio_path = audio_dir / f"{utt_id}.flac"
                samples.append((str(audio_path), label))
        
        print(f"Loaded {len(samples)} samples from protocol {protocol_path}")

        if not samples:
            raise RuntimeError(f"No samples were loaded from protocol {protocol_path}")

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        audio_path, label = self.samples[idx]
        waveform, sample_rate = torchaudio.load(audio_path)

        # Convert to mono if needed.
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        waveform = waveform.squeeze(0)
        return waveform, sample_rate, label


class LA2019DataModule(LightningDataModule):
    """LightningDataModule for ASVspoof 2019 Logical Access (LA)."""

    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        num_workers: int = 4,
        persistent_workers: bool = True,
        pin_memory: bool = True,
        model_name: str = "microsoft/wavlm-large",
        target_sample_rate: int = 16_000,
        augmentation: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size
        self.feature_extractor = None
        self.augmentation_cfg = augmentation or {}
        self.noise_paths: List[str] = []
        self.music_paths: List[str] = []
        self.rir_paths: List[str] = []

    def _expand_audio_paths(self, path_specs: Optional[List[str]]) -> List[str]:
        if not path_specs:
            return []

        resolved_paths: List[str] = []
        valid_ext = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac"}

        for spec in path_specs:
            p = Path(spec)
            if p.is_file() and p.suffix.lower() in valid_ext:
                resolved_paths.append(str(p))
                continue

            if p.is_dir():
                for child in p.rglob("*"):
                    if child.is_file() and child.suffix.lower() in valid_ext:
                        resolved_paths.append(str(child))
                continue

            for matched_str in glob.glob(spec, recursive=True):
                matched = Path(matched_str)
                if matched.is_file() and matched.suffix.lower() in valid_ext:
                    resolved_paths.append(str(matched))

        # Keep insertion order but remove duplicates.
        return list(dict.fromkeys(resolved_paths))

    def _sample_external_audio(self, file_paths: List[str]) -> Optional[torch.Tensor]:
        if not file_paths:
            return None

        idx = int(torch.randint(0, len(file_paths), (1,)).item())
        sample_path = file_paths[idx]
        waveform, sample_rate = torchaudio.load(sample_path)

        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform.squeeze(0)

        if sample_rate != self.hparams.target_sample_rate:
            waveform = torchaudio.functional.resample(
                waveform,
                sample_rate,
                self.hparams.target_sample_rate,
            )

        return waveform

    def _rand_uniform(self, low: float, high: float) -> float:
        return float(torch.empty(1).uniform_(low, high).item())

    def _maybe(self, probability: float) -> bool:
        return bool(torch.rand(1).item() < probability)

    def _apply_augmentations(self, waveform: torch.Tensor) -> torch.Tensor:
        cfg = self.augmentation_cfg
        if not cfg.get("enabled", False):
            return waveform

        global_prob = float(cfg.get("apply_probability", 1.0))
        if not self._maybe(global_prob):
            return waveform

        # Signal level operations
        if self._maybe(float(cfg.get("peak_level_prob", 0.0))):
            peak_range = cfg.get("peak_level_dbfs_range", [-6.0, -1.0])
            target_peak = self._rand_uniform(float(peak_range[0]), float(peak_range[1]))
            waveform = aug.peak_level_adjustment(waveform, target_peak_dbfs=target_peak)

        if self._maybe(float(cfg.get("fade_prob", 0.0))):
            fade_ms_range = cfg.get("fade_ms_range", [5.0, 40.0])
            fade_ms = self._rand_uniform(float(fade_ms_range[0]), float(fade_ms_range[1]))
            waveform = aug.fade_in_fade_out(
                waveform,
                sample_rate=self.hparams.target_sample_rate,
                fade_in_ms=fade_ms,
                fade_out_ms=fade_ms,
                fade_shape=str(cfg.get("fade_shape", "linear")),
            )

        # Signal structure modifications
        if self._maybe(float(cfg.get("trim_silence_prob", 0.0))):
            threshold_dbfs = float(cfg.get("trim_silence_threshold_dbfs", -40.0))
            waveform = aug.trim_silence(waveform, threshold_dbfs=threshold_dbfs)

        if self._maybe(float(cfg.get("zero_padding_prob", 0.0))):
            target_num_samples = int(cfg.get("zero_padding_target_num_samples", 0))
            if target_num_samples > 0:
                waveform = aug.zero_padding(
                    waveform,
                    target_num_samples=target_num_samples,
                    mode=str(cfg.get("zero_padding_mode", "right")),
                )

        # Environmental conditions
        if self._maybe(float(cfg.get("env_noise_prob", 0.0))) and self.noise_paths:
            snr_range = cfg.get("env_noise_snr_db_range", [10.0, 25.0])
            snr_db = self._rand_uniform(float(snr_range[0]), float(snr_range[1]))
            noise_waveform = self._sample_external_audio(self.noise_paths)
            if noise_waveform is not None:
                waveform = aug.add_environmental_noise(waveform, noise_waveform, snr_db=snr_db)

        if self._maybe(float(cfg.get("background_music_prob", 0.0))) and self.music_paths:
            snr_range = cfg.get("background_music_snr_db_range", [15.0, 30.0])
            snr_db = self._rand_uniform(float(snr_range[0]), float(snr_range[1]))
            music_waveform = self._sample_external_audio(self.music_paths)
            if music_waveform is not None:
                waveform = aug.add_background_music(waveform, music_waveform, snr_db=snr_db)

        if self._maybe(float(cfg.get("rir_prob", 0.0))) and self.rir_paths:
            rir_waveform = self._sample_external_audio(self.rir_paths)
            if rir_waveform is not None:
                waveform = aug.rir_convolution(waveform, rir_waveform)

        # Media channel effects
        if self._maybe(float(cfg.get("codec_prob", 0.0))):
            codec = str(cfg.get("codec_format", "mp3"))
            codec_compression = cfg.get("codec_compression", "64k")
            waveform = aug.audio_codec_compression(
                waveform,
                sample_rate=self.hparams.target_sample_rate,
                codec=codec,
                compression=codec_compression,
            )

        if self._maybe(float(cfg.get("resample_prob", 0.0))):
            low_rate, high_rate = cfg.get("resample_temp_rate_range", [8000, 12000])
            temp_rate = int(self._rand_uniform(float(low_rate), float(high_rate)))
            waveform = aug.resample_audio(
                waveform,
                orig_sample_rate=self.hparams.target_sample_rate,
                new_sample_rate=temp_rate,
            )
            waveform = aug.resample_audio(
                waveform,
                orig_sample_rate=temp_rate,
                new_sample_rate=self.hparams.target_sample_rate,
            )

        if self._maybe(float(cfg.get("drc_prob", 0.0))):
            drc_amount = float(cfg.get("drc_amount", 6.0))
            waveform = aug.dynamic_range_compression(
                waveform,
                sample_rate=self.hparams.target_sample_rate,
                amount=drc_amount,
            )

        if self._maybe(float(cfg.get("bandwidth_prob", 0.0))):
            high_range = cfg.get("bandwidth_high_freq_range", [3000.0, 7000.0])
            high_freq = self._rand_uniform(float(high_range[0]), float(high_range[1]))
            waveform = aug.bandwidth_limitation(
                waveform,
                sample_rate=self.hparams.target_sample_rate,
                low_freq=cfg.get("bandwidth_low_freq", None),
                high_freq=high_freq,
            )

        return waveform

    @property
    def num_classes(self) -> int:
        return 2

    def prepare_data(self) -> None:
        # This also pre-downloads extractor artifacts when needed.
        # _ = AutoFeatureExtractor.from_pretrained(self.hparams.model_name)
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by world size ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        if self.data_train is None and self.data_val is None and self.data_test is None:
            self.data_train = LA2019Dataset(self.hparams.data_dir, split="train")
            self.data_val = LA2019Dataset(self.hparams.data_dir, split="dev")
            self.data_test = LA2019Dataset(self.hparams.data_dir, split="eval")

        if self.feature_extractor is None:
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.hparams.model_name , trust_remote_code=True, force_download=True)

        if self.augmentation_cfg.get("enabled", False):
            self.noise_paths = self._expand_audio_paths(self.augmentation_cfg.get("noise_paths", []))
            self.music_paths = self._expand_audio_paths(self.augmentation_cfg.get("music_paths", []))
            self.rir_paths = self._expand_audio_paths(self.augmentation_cfg.get("rir_paths", []))

    def _collate_fn(
        self,
        batch: List[Tuple[torch.Tensor, int, int]],
        apply_augment: bool = False,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        # if self.feature_extractor is None:
        #     self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.hparams.model_name)

        waveforms: List[torch.Tensor] = []
        labels: List[int] = []

        for waveform, sample_rate, label in batch:
            if sample_rate != self.hparams.target_sample_rate:
                waveform = torchaudio.functional.resample(
                    waveform, sample_rate, self.hparams.target_sample_rate
                )

            if apply_augment:
                waveform = self._apply_augmentations(waveform)

            waveforms.append(waveform)
            labels.append(label)

        if self.hparams.model_name == "labhamlet/gramt-binaural-time":
            # Duplicate mono to 4 channels for GRAMT binaural-time model.
            print(waveforms[0].shape)
            waveforms = [torch.stack([w, w, w, w], dim=0) for w in waveforms]
            print(waveforms[0].shape)
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
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            persistent_workers=self.hparams.persistent_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            collate_fn=partial(
                self._collate_fn,
                apply_augment=bool(self.augmentation_cfg.get("enabled", False)),
            ),
        )

    def val_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            persistent_workers=self.hparams.persistent_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=partial(self._collate_fn, apply_augment=False),
        )

    def test_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            persistent_workers=self.hparams.persistent_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=partial(self._collate_fn, apply_augment=False),
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        pass

    def state_dict(self) -> Dict[Any, Any]:
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        pass


if __name__ == "__main__":
    dm = LA2019DataModule(data_dir="/workspace/datasets/LA")
    dm.setup(stage="fit")
    print(
        f"Loaded splits -> train: {len(dm.data_train)}, dev: {len(dm.data_val)}, eval: {len(dm.data_test)}",
        flush=True,
    )
