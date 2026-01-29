# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any, List, Tuple, Union

import librosa
import numpy as np
import torch
from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig
from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram
from torch.utils.data import Dataset

AudioLike = Union[
    str,                     # wav path, URL, base64
    np.ndarray,              # waveform (requires sr)
    Tuple[np.ndarray, int],  # (waveform, sr)
]

MaybeList = Union[Any, List[Any]]

class TTSDataset(Dataset):
    def __init__(
        self,
        data_list,
        processor,
        config: Qwen3TTSConfig,
        lag_num: int = -1,
        language: str = "english",
        codec_format: str = "inference",
    ):
        self.data_list = data_list
        self.processor = processor
        self.lag_num = lag_num
        self.config = config
        self.language = language
        self.codec_format = codec_format
        # Get language ID from config (english=2050)
        self.language_id = config.talker_config.codec_language_id.get(language.lower(), 2050)

    def __len__(self):
        return len(self.data_list)

    def _load_audio_to_np(self, x: str) -> Tuple[np.ndarray, int]:
        audio, sr = librosa.load(x, sr=None, mono=True)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=-1)
        return audio.astype(np.float32), int(sr)

    def _normalize_audio_inputs(self, audios: Union[AudioLike, List[AudioLike]]) -> List[Tuple[np.ndarray, int]]:
        if isinstance(audios, list):
            items = audios
        else:
            items = [audios]

        out: List[Tuple[np.ndarray, int]] = []
        for a in items:
            if isinstance(a, str):
                out.append(self._load_audio_to_np(a))
            elif isinstance(a, tuple) and len(a) == 2 and isinstance(a[0], np.ndarray):
                out.append((a[0].astype(np.float32), int(a[1])))
            elif isinstance(a, np.ndarray):
                raise ValueError("For numpy waveform input, pass a tuple (audio, sr).")
            else:
                raise TypeError(f"Unsupported audio input type: {type(a)}")
        return out

    def _build_assistant_text(self, text: str) -> str:
        return f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"

    def _ensure_list(self, x: MaybeList) -> List[Any]:
        return x if isinstance(x, list) else [x]

    def _tokenize_texts(self, text) -> List[torch.Tensor]:
        input = self.processor(text=text, return_tensors="pt", padding=True)
        input_id = input["input_ids"]
        input_id = input_id.unsqueeze(0) if input_id.dim() == 1 else input_id
        return input_id

    @torch.inference_mode()
    def extract_mels(self, audio, sr):
        if sr != 24000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=24000)
            sr = 24000
        mels = mel_spectrogram(
            torch.from_numpy(audio).unsqueeze(0),
            n_fft=1024,
            num_mels=128,
            sampling_rate=24000,
            hop_size=256,
            win_size=1024,
            fmin=0,
            fmax=12000
        ).transpose(1, 2)
        return mels

    def __getitem__(self, idx):
        item = self.data_list[idx]

        audio_path = item["audio"]
        text = item["text"]
        audio_codes = item["audio_codes"]
        ref_audio_path = item['ref_audio']

        text = self._build_assistant_text(text)
        text_ids = self._tokenize_texts(text)

        audio_codes = torch.tensor(audio_codes, dtype=torch.long)

        ref_audio_list = self._ensure_list(ref_audio_path)
        normalized = self._normalize_audio_inputs(ref_audio_list)
        wav, sr = normalized[0]

        ref_mel = self.extract_mels(audio=wav, sr=sr)

        return {
            "text_ids": text_ids[:, :-5],
            "audio_codes": audio_codes,
            "ref_mel": ref_mel
        }

    def collate_fn(self, batch):
        """
        Build the 2-channel (text/codecs) sequence used by Qwen3-TTS.

        Note: Qwen3-TTS has had small variations in how the codec prefix / BOS is placed across
        releases. This dataset supports two layouts via `codec_format`:
          - "inference": matches this repo's current "prefix contains codec_bos" layout.
          - "teacher_forcing": keeps codec_bos immediately before the first codec token label
            (classic LM teacher-forcing), while still adding think+language tokens in the prefix.
        """
        assert self.lag_num == -1

        item_length = [b['text_ids'].shape[1] + b['audio_codes'].shape[0] for b in batch]
        max_length = max(item_length) + 12  # Extra space for longer prefix
        b, t = len(batch), max_length

        input_ids = torch.zeros((b, t, 2), dtype=torch.long)
        codec_ids = torch.zeros((b, t, 16), dtype=torch.long)
        text_embedding_mask = torch.zeros((b, t), dtype=torch.bool)
        codec_embedding_mask = torch.zeros((b, t), dtype=torch.bool)
        codec_mask = torch.zeros((b, t), dtype=torch.bool)
        attention_mask = torch.zeros((b, t), dtype=torch.long)
        codec_0_labels = torch.full((b, t), -100, dtype=torch.long)

        for i, data in enumerate(batch):
            text_ids = data['text_ids']
            audio_codec_0 = data['audio_codes'][:, 0]
            audio_codecs = data['audio_codes']

            text_ids_len = text_ids.shape[1]
            codec_ids_len = audio_codec_0.shape[0]

            # Initialize both channels with pad ids to avoid accidental "0" tokens participating
            # in attention when sequence length is shorter than the max_length buffer.
            input_ids[i, :, 0] = self.config.tts_pad_token_id
            input_ids[i, :, 1] = self.config.talker_config.codec_pad_id

            if self.codec_format == "teacher_forcing":
                # Prefix (codec channel, fixed at pos>=3). Keep codec_bos immediately before the
                # first audio codec label position (classic LM teacher forcing).
                codec_prefix_start = 3
                codec_prefix = [
                    self.config.talker_config.codec_think_id,
                    self.config.talker_config.codec_think_bos_id,
                    self.language_id,
                    self.config.talker_config.codec_think_eos_id,
                    0,  # SPEAKER placeholder (masked; replaced in training)
                    self.config.talker_config.codec_pad_id,
                ]
                speaker_pos = codec_prefix_start + 4
                codec_prefix_end = codec_prefix_start + len(codec_prefix) - 1

                # Text channel alignment: tts_bos aligned to last prefix token (as in the
                # official dataset layout this repo started from).
                text_bos_pos = codec_prefix_end
                text_start = text_bos_pos + 1
                text_tail_len = text_ids_len - 3
                text_eos_pos = text_start + text_tail_len

                codec_bos_pos = text_eos_pos + 1
                audio_start = codec_bos_pos + 1
                codec_eos_pos = audio_start + codec_ids_len
                seq_len = codec_eos_pos + 1

                # Text channel
                input_ids[i, :3, 0] = text_ids[0, :3]
                input_ids[i, text_bos_pos, 0] = self.config.tts_bos_token_id
                input_ids[i, text_start:text_start + text_tail_len, 0] = text_ids[0, 3:]
                input_ids[i, text_eos_pos, 0] = self.config.tts_eos_token_id
                input_ids[i, text_eos_pos + 1:seq_len, 0] = self.config.tts_pad_token_id
                text_embedding_mask[i, :seq_len] = True

                # Codec channel
                input_ids[i, codec_prefix_start:codec_prefix_end + 1, 1] = torch.tensor(codec_prefix, dtype=torch.long)
                input_ids[i, codec_prefix_end + 1:codec_bos_pos, 1] = self.config.talker_config.codec_pad_id
                input_ids[i, codec_bos_pos, 1] = self.config.talker_config.codec_bos_id
                input_ids[i, audio_start:audio_start + codec_ids_len, 1] = audio_codec_0
                input_ids[i, codec_eos_pos, 1] = self.config.talker_config.codec_eos_token_id

                codec_0_labels[i, audio_start:audio_start + codec_ids_len] = audio_codec_0
                codec_0_labels[i, codec_eos_pos] = self.config.talker_config.codec_eos_token_id

                codec_ids[i, audio_start:audio_start + codec_ids_len, :] = audio_codecs

                codec_embedding_mask[i, codec_prefix_start:seq_len] = True
                codec_embedding_mask[i, speaker_pos] = False
                codec_mask[i, audio_start:audio_start + codec_ids_len] = True
                attention_mask[i, :seq_len] = True
            elif self.codec_format == "inference":
                # Current repo layout: codec_bos is inside the fixed prefix after the speaker slot.
                codec_prefix_start = 3
                codec_prefix = [
                    self.config.talker_config.codec_think_id,
                    self.config.talker_config.codec_think_bos_id,
                    self.language_id,
                    self.config.talker_config.codec_think_eos_id,
                    0,  # SPEAKER placeholder (masked; replaced in training)
                    self.config.talker_config.codec_pad_id,
                    self.config.talker_config.codec_bos_id,
                ]
                speaker_pos = codec_prefix_start + 4

                # Text channel (repo's current offsets)
                text_bos_pos = 10
                text_start = 11
                text_tail_len = text_ids_len - 3
                text_eos_pos = text_start + text_tail_len
                seq_len = 11 + text_ids_len + codec_ids_len

                # Text channel
                input_ids[i, :3, 0] = text_ids[0, :3]
                input_ids[i, 3:text_bos_pos, 0] = self.config.tts_pad_token_id
                input_ids[i, text_bos_pos, 0] = self.config.tts_bos_token_id
                input_ids[i, text_start:text_start + text_tail_len, 0] = text_ids[0, 3:]
                input_ids[i, text_eos_pos, 0] = self.config.tts_eos_token_id
                input_ids[i, text_eos_pos + 1:seq_len, 0] = self.config.tts_pad_token_id
                text_embedding_mask[i, :seq_len] = True

                # Codec channel
                input_ids[i, codec_prefix_start:codec_prefix_start + len(codec_prefix), 1] = torch.tensor(codec_prefix, dtype=torch.long)
                input_ids[i, 10:10 + text_ids_len - 2, 1] = self.config.talker_config.codec_pad_id
                audio_start = 10 + text_ids_len - 2
                codec_eos_pos = audio_start + codec_ids_len
                input_ids[i, audio_start:audio_start + codec_ids_len, 1] = audio_codec_0
                input_ids[i, codec_eos_pos, 1] = self.config.talker_config.codec_eos_token_id
                # Fill any remaining positions within seq_len (there are typically 2) with codec_pad.
                input_ids[i, codec_eos_pos + 1:seq_len, 1] = self.config.talker_config.codec_pad_id

                codec_0_labels[i, audio_start:audio_start + codec_ids_len] = audio_codec_0
                codec_0_labels[i, codec_eos_pos] = self.config.talker_config.codec_eos_token_id

                codec_ids[i, audio_start:audio_start + codec_ids_len, :] = audio_codecs

                codec_embedding_mask[i, codec_prefix_start:seq_len] = True
                codec_embedding_mask[i, speaker_pos] = False
                codec_mask[i, audio_start:audio_start + codec_ids_len] = True
                attention_mask[i, :seq_len] = True
            else:
                raise ValueError(f"Unknown codec_format: {self.codec_format!r}")

        ref_mels = [data['ref_mel'] for data in batch]
        max_mel_len = max(m.shape[1] for m in ref_mels)
        ref_mels = [torch.nn.functional.pad(m, (0, 0, 0, max_mel_len - m.shape[1])) for m in ref_mels]
        ref_mels = torch.cat(ref_mels, dim=0)

        return {
            'input_ids': input_ids,
            'ref_mels': ref_mels,
            'attention_mask': attention_mask,
            'text_embedding_mask': text_embedding_mask.unsqueeze(-1),
            'codec_embedding_mask': codec_embedding_mask.unsqueeze(-1),
            'codec_0_labels': codec_0_labels,
            'codec_ids': codec_ids,
            'codec_mask': codec_mask
        }
