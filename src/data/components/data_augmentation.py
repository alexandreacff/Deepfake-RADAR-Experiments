from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Literal, Tuple

import torch
import torchaudio

def _to_2d(waveform: torch.Tensor) -> Tuple[torch.Tensor, bool]:
	if waveform.ndim == 1:
		return waveform.unsqueeze(0), True
	if waveform.ndim == 2:
		return waveform, False
	raise ValueError("Expected waveform with shape [T] or [C, T].")


def _restore_shape(waveform: torch.Tensor, was_1d: bool) -> torch.Tensor:
	return waveform.squeeze(0) if was_1d else waveform


def peak_level_adjustment(
	waveform: torch.Tensor,
	target_peak_dbfs: float = -1.0,
	eps: float = 1e-12,
) -> torch.Tensor:
	"""Adjust waveform peak to a target dBFS value."""
	x, was_1d = _to_2d(waveform)
	peak = x.abs().max().clamp_min(eps)
	target_peak = 10.0 ** (target_peak_dbfs / 20.0)
	y = x * (target_peak / peak)
	return _restore_shape(y, was_1d)


def fade_in_fade_out(
	waveform: torch.Tensor,
	sample_rate: int,
	fade_in_ms: float = 10.0,
	fade_out_ms: float = 10.0,
	fade_shape: Literal["linear", "exponential", "logarithmic", "quarter_sine", "half_sine"] = "linear",
) -> torch.Tensor:
	"""Apply fade-in and fade-out envelopes."""
	x, was_1d = _to_2d(waveform)
	fade_in_len = max(0, int(sample_rate * fade_in_ms / 1000.0))
	fade_out_len = max(0, int(sample_rate * fade_out_ms / 1000.0))
	fade = torchaudio.transforms.Fade(
		fade_in_len=fade_in_len,
		fade_out_len=fade_out_len,
		fade_shape=fade_shape,
	)
	y = fade(x)
	return _restore_shape(y, was_1d)


def trim_silence(
	waveform: torch.Tensor,
	threshold_dbfs: float = -40.0,
) -> torch.Tensor:
	"""Trim leading and trailing silence using an absolute dBFS threshold."""
	x, was_1d = _to_2d(waveform)
	mono_energy = x.abs().amax(dim=0)
	threshold = 10.0 ** (threshold_dbfs / 20.0)
	non_silent = torch.nonzero(mono_energy > threshold, as_tuple=False).squeeze(-1)

	if non_silent.numel() == 0:
		# Keep at least one sample to avoid empty tensors downstream.
		y = x[:, :1]
	else:
		start = int(non_silent[0].item())
		end = int(non_silent[-1].item()) + 1
		y = x[:, start:end]
	return _restore_shape(y, was_1d)


def zero_padding(
	waveform: torch.Tensor,
	target_num_samples: int,
	mode: Literal["left", "right", "both"] = "right",
) -> torch.Tensor:
	"""Pad waveform with zeros until target length. If already longer, keep as-is."""
	x, was_1d = _to_2d(waveform)
	current_len = x.shape[-1]
	pad_total = target_num_samples - current_len
	if pad_total <= 0:
		return waveform

	if mode == "left":
		left, right = pad_total, 0
	elif mode == "both":
		left = pad_total // 2
		right = pad_total - left
	else:
		left, right = 0, pad_total

	y = torch.nn.functional.pad(x, (left, right), mode="constant", value=0.0)
	return _restore_shape(y, was_1d)


def _mix_with_snr(
	speech: torch.Tensor,
	noise: torch.Tensor,
	snr_db: float,
	eps: float = 1e-12,
) -> torch.Tensor:
	speech_rms = speech.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(eps)
	noise_rms = noise.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(eps)
	target_noise_rms = speech_rms / (10.0 ** (snr_db / 20.0))
	scaled_noise = noise * (target_noise_rms / noise_rms)
	return speech + scaled_noise


def _fit_reference_length(
	reference: torch.Tensor, source: torch.Tensor, random_offset: bool = True
) -> torch.Tensor:
	"""Tile/crop source to match reference length."""
	ref_len = reference.shape[-1]
	src_len = source.shape[-1]

	if src_len < ref_len:
		reps = (ref_len + src_len - 1) // src_len
		source = source.repeat(1, reps)
	if source.shape[-1] > ref_len:
		max_start = source.shape[-1] - ref_len
		if random_offset and max_start > 0:
			start = int(torch.randint(0, max_start + 1, (1,)).item())
		else:
			start = 0
		source = source[:, start : start + ref_len]
	return source


def add_environmental_noise(
	waveform: torch.Tensor,
	noise_waveform: torch.Tensor,
	snr_db: float = 15.0,
	random_offset: bool = True,
	clamp: bool = True,
) -> torch.Tensor:
	"""Mix environmental noise at a controlled SNR."""
	x, was_1d = _to_2d(waveform)
	n, _ = _to_2d(noise_waveform)

	if n.shape[0] == 1 and x.shape[0] > 1:
		n = n.repeat(x.shape[0], 1)
	elif n.shape[0] != x.shape[0]:
		raise ValueError("noise_waveform channel count must match waveform or be mono.")

	n = _fit_reference_length(x, n, random_offset=random_offset)
	y = _mix_with_snr(x, n, snr_db=snr_db)
	if clamp:
		y = y.clamp(-1.0, 1.0)
	return _restore_shape(y, was_1d)


def add_background_music(
	waveform: torch.Tensor,
	music_waveform: torch.Tensor,
	snr_db: float = 20.0,
	random_offset: bool = True,
	clamp: bool = True,
) -> torch.Tensor:
	"""Mix background music at a controlled SNR (usually higher than noise SNR)."""
	return add_environmental_noise(
		waveform=waveform,
		noise_waveform=music_waveform,
		snr_db=snr_db,
		random_offset=random_offset,
		clamp=clamp,
	)


def rir_convolution(
	waveform: torch.Tensor,
	rir_waveform: torch.Tensor,
	normalize_rir: bool = True,
	trim_to_input: bool = True,
) -> torch.Tensor:
	"""Apply room impulse response convolution."""
	x, was_1d = _to_2d(waveform)
	h, _ = _to_2d(rir_waveform)

	if h.shape[0] == 1 and x.shape[0] > 1:
		h = h.repeat(x.shape[0], 1)
	elif h.shape[0] != x.shape[0]:
		raise ValueError("rir_waveform channel count must match waveform or be mono.")

	if normalize_rir:
		denom = h.abs().sum(dim=-1, keepdim=True).clamp_min(1e-12)
		h = h / denom

	y_channels = []
	for c in range(x.shape[0]):
		y_c = torch.nn.functional.conv1d(
			x[c : c + 1].unsqueeze(0),
			h[c : c + 1].flip(-1).unsqueeze(1),
			padding=h.shape[-1] - 1,
		).squeeze(0)
		y_channels.append(y_c)
	y = torch.cat(y_channels, dim=0)

	if trim_to_input:
		y = y[:, : x.shape[-1]]
	return _restore_shape(y, was_1d)


def audio_codec_compression(
	waveform: torch.Tensor,
	sample_rate: int,
	codec: Literal["mp3", "ogg", "flac", "wav"] = "mp3",
	compression: str | None = "64k",
) -> torch.Tensor:
	"""Apply codec compression by encoding and decoding through torchaudio backends."""
	x, was_1d = _to_2d(waveform)
	suffix = f".{codec}"

	with tempfile.TemporaryDirectory(prefix="audio_aug_codec_") as tmp_dir:
		tmp_path = Path(tmp_dir) / f"temp{suffix}"

		save_kwargs = {"format": codec}
		if compression is not None:
			save_kwargs["compression"] = compression

		try:
			torchaudio.save(str(tmp_path), x.cpu(), sample_rate, **save_kwargs)
		except Exception:
			# Fallback when the selected backend/codec does not support `compression`.
			torchaudio.save(str(tmp_path), x.cpu(), sample_rate, format=codec)

		y, _ = torchaudio.load(str(tmp_path))

	if y.shape[0] != x.shape[0] and y.shape[0] == 1 and x.shape[0] > 1:
		y = y.repeat(x.shape[0], 1)
	y = _fit_reference_length(x, y, random_offset=False)
	return _restore_shape(y.to(waveform.device), was_1d)


def resample_audio(
	waveform: torch.Tensor,
	orig_sample_rate: int,
	new_sample_rate: int,
) -> torch.Tensor:
	"""Resample waveform to a different sample rate."""
	x, was_1d = _to_2d(waveform)
	y = torchaudio.functional.resample(x, orig_sample_rate, new_sample_rate)
	return _restore_shape(y, was_1d)


def dynamic_range_compression(
	waveform: torch.Tensor,
	sample_rate: int,
	amount: float = 6.0,
) -> torch.Tensor:
	"""Apply dynamic range compression (sox compand when available, otherwise soft-knee tanh)."""
	x, was_1d = _to_2d(waveform)

	try:
		effects = [["compand", "0.3,1", "6:-70,-60,-20", "-5", "-90", "0.2"]]
		y, _ = torchaudio.sox_effects.apply_effects_tensor(x, sample_rate, effects)
	except Exception:
		# Backend-agnostic fallback: simple soft compression curve.
		y = torch.tanh(amount * x) / torch.tanh(torch.tensor(amount, device=x.device))

	return _restore_shape(y, was_1d)


def bandwidth_limitation(
	waveform: torch.Tensor,
	sample_rate: int,
	low_freq: float | None = None,
	high_freq: float | None = 4000.0,
) -> torch.Tensor:
	"""Apply bandwidth limitation via high-pass + low-pass filtering."""
	x, was_1d = _to_2d(waveform)
	y = x

	if low_freq is not None and low_freq > 0:
		y = torchaudio.functional.highpass_biquad(y, sample_rate=sample_rate, cutoff_freq=low_freq)
	if high_freq is not None and high_freq < sample_rate / 2:
		y = torchaudio.functional.lowpass_biquad(y, sample_rate=sample_rate, cutoff_freq=high_freq)

	return _restore_shape(y, was_1d)

