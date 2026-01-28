#!/usr/bin/env python3
# Qwen3-TTS Fine-tuning Script - Optimized for Cloud GPU (A100/4090)
import argparse
import json
import os
import shutil

import torch
from accelerate import Accelerator
from dataset import TTSDataset
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from safetensors.torch import save_file
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoConfig, get_linear_schedule_with_warmup

target_speaker_embedding = None

def train():
    global target_speaker_embedding

    parser = argparse.ArgumentParser()
    parser.add_argument("--init_model_path", type=str, default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    parser.add_argument("--output_model_path", type=str, default="output")
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--speaker_name", type=str, default="michael_douglas")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--save_every_n_steps", type=int, default=500)
    args = parser.parse_args()

    # GPU-optimized settings
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="bf16",
        log_with="tensorboard",
        project_dir="./logs"
    )

    accelerator.print(f"Training with {accelerator.num_processes} GPU(s)")
    accelerator.print(f"Batch size: {args.batch_size}, Gradient accumulation: {args.gradient_accumulation_steps}")
    accelerator.print(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps * accelerator.num_processes}")

    MODEL_PATH = args.init_model_path

    # Load model with flash attention for speed
    qwen3tts = Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    config = AutoConfig.from_pretrained(MODEL_PATH)

    train_data = open(args.train_jsonl).readlines()
    train_data = [json.loads(line) for line in train_data]
    accelerator.print(f"Loaded {len(train_data)} training samples")

    dataset = TTSDataset(train_data, qwen3tts.processor, config)
    train_dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
        num_workers=4,
        pin_memory=True
    )

    optimizer = AdamW(qwen3tts.model.parameters(), lr=args.lr, weight_decay=0.01)

    num_training_steps = len(train_dataloader) * args.num_epochs
    num_warmup_steps = min(100, num_training_steps // 10)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    model, optimizer, train_dataloader, scheduler = accelerator.prepare(
        qwen3tts.model, optimizer, train_dataloader, scheduler
    )

    num_epochs = args.num_epochs
    model.train()

    global_step = 0
    best_loss = float('inf')

    accelerator.print(f"Starting training for {num_epochs} epochs, {len(train_dataloader)} steps per epoch")

    for epoch in range(num_epochs):
        epoch_loss = 0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                input_ids = batch['input_ids']
                codec_ids = batch['codec_ids']
                ref_mels = batch['ref_mels']
                text_embedding_mask = batch['text_embedding_mask']
                codec_embedding_mask = batch['codec_embedding_mask']
                attention_mask = batch['attention_mask']
                codec_0_labels = batch['codec_0_labels']
                codec_mask = batch['codec_mask']

                speaker_embedding = model.speaker_encoder(ref_mels.to(model.device).to(model.dtype)).detach()
                if target_speaker_embedding is None:
                    target_speaker_embedding = speaker_embedding

                input_text_ids = input_ids[:, :, 0]
                input_codec_ids = input_ids[:, :, 1]

                input_text_embedding = model.talker.model.text_embedding(input_text_ids) * text_embedding_mask
                input_codec_embedding = model.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask
                input_codec_embedding[:, 6, :] = speaker_embedding

                input_embeddings = input_text_embedding + input_codec_embedding

                for i in range(1, 16):
                    codec_i_embedding = model.talker.code_predictor.get_input_embeddings()[i - 1](codec_ids[:, :, i])
                    codec_i_embedding = codec_i_embedding * codec_mask.unsqueeze(-1)
                    input_embeddings = input_embeddings + codec_i_embedding

                outputs = model.talker(
                    inputs_embeds=input_embeddings[:, :-1, :],
                    attention_mask=attention_mask[:, :-1],
                    labels=codec_0_labels[:, 1:],
                    output_hidden_states=True
                )

                hidden_states = outputs.hidden_states[0][-1]
                talker_hidden_states = hidden_states[codec_mask[:, 1:]]
                talker_codec_ids = codec_ids[codec_mask]

                sub_talker_logits, sub_talker_loss = model.talker.forward_sub_talker_finetune(talker_codec_ids, talker_hidden_states)

                loss = outputs.loss + sub_talker_loss
                epoch_loss += loss.item()

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            global_step += 1

            if step % 10 == 0:
                accelerator.print(f"Epoch {epoch} | Step {step}/{len(train_dataloader)} | Loss: {loss.item():.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")

            # Save checkpoint every N steps
            if args.save_every_n_steps > 0 and global_step % args.save_every_n_steps == 0:
                if accelerator.is_main_process:
                    save_checkpoint(accelerator, model, qwen3tts, MODEL_PATH, args, f"checkpoint-step-{global_step}", target_speaker_embedding)

        avg_epoch_loss = epoch_loss / len(train_dataloader)
        accelerator.print(f"Epoch {epoch} completed | Avg Loss: {avg_epoch_loss:.4f}")

        # Save only best checkpoint to save disk space
        if accelerator.is_main_process:
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                save_checkpoint(accelerator, model, qwen3tts, MODEL_PATH, args, "checkpoint-best", target_speaker_embedding)

    accelerator.print("Training complete!")

def save_checkpoint(accelerator, model, qwen3tts, model_path, args, checkpoint_name, speaker_embedding):
    output_dir = os.path.join(args.output_model_path, checkpoint_name)
    accelerator.print(f"Saving checkpoint to {output_dir}")

    shutil.copytree(model_path, output_dir, dirs_exist_ok=True)

    input_config_file = os.path.join(model_path, "config.json")
    output_config_file = os.path.join(output_dir, "config.json")
    with open(input_config_file, 'r', encoding='utf-8') as f:
        config_dict = json.load(f)
    config_dict["tts_model_type"] = "custom_voice"
    talker_config = config_dict.get("talker_config", {})
    talker_config["spk_id"] = {args.speaker_name: 3000}
    talker_config["spk_is_dialect"] = {args.speaker_name: False}
    config_dict["talker_config"] = talker_config

    with open(output_config_file, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)

    unwrapped_model = accelerator.unwrap_model(model)
    state_dict = {k: v.detach().to("cpu") for k, v in unwrapped_model.state_dict().items()}

    drop_prefix = "speaker_encoder"
    keys_to_drop = [k for k in state_dict.keys() if k.startswith(drop_prefix)]
    for k in keys_to_drop:
        del state_dict[k]

    weight = state_dict['talker.model.codec_embedding.weight']
    state_dict['talker.model.codec_embedding.weight'][3000] = speaker_embedding[0].detach().to(weight.device).to(weight.dtype)
    save_path = os.path.join(output_dir, "model.safetensors")
    save_file(state_dict, save_path)

if __name__ == "__main__":
    train()
