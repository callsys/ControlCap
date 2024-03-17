import math
import copy
import random
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torchvision
from textblob import TextBlob
from torchvision.models.vision_transformer import MLPBlock
from peft import LoraConfig, get_peft_model

from lavis.common.registry import registry
from lavis.models.blip2_models.blip2_t5 import Blip2T5
from controlcap.models.tagging_heads.bert import BertConfig, BertModel
from controlcap.models.tagging_heads.asymmetric_loss import AsymmetricLoss


version = "blip2"

if version == "blip2":
    @registry.register_model("dev_t5")
    class DevT5(Blip2T5):
        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs
            base_kwargs = copy.deepcopy(kwargs)
            base_kwargs_keys = ["vit_model", "img_size", "drop_path_rate", "use_grad_checkpoint", "vit_precision", "freeze_vit",
                                "num_query_token", "t5_model", "prompt", "max_txt_len", "apply_lemmatizer"]
            for key in kwargs.keys():
                if key not in base_kwargs_keys:
                    base_kwargs.pop(key)
            super().__init__(*args, **base_kwargs)

            # Trainable parameters
            names = ["Qformer", "t5_proj"]
            self.finetune_llm = kwargs.get("finetune_llm", False)
            if self.finetune_llm:
                lora_config = LoraConfig(
                    r=64, lora_alpha=128, lora_dropout=0.0,
                    target_modules=["embed_tokens", "lm_head", "q", "k", "v"]
                )

                self.t5_model = get_peft_model(self.t5_model, lora_config)
                self.t5_model.to(torch.float32)
                names.extend(["lora"])
            params = [0] * len(names)

            trainable_params = 0
            all_params = 0
            for param_name, param in self.named_parameters():
                all_params += param.numel()
                param.requires_grad = False
                for idx, name in enumerate(names):
                    if name in param_name:
                        param.requires_grad = True
                        trainable_params += param.numel()
                        params[idx] += param.numel()
                        break
            print(f"[ trainable ratio : {trainable_params / all_params}]")
            for idx, name in enumerate(names):
                print(f"[{name} ratio : {params[idx] / all_params}")

        def forward(self, samples):
            image = samples["region_images"]

            with self.maybe_autocast(dtype=torch.float16):
                embeds = self.ln_vision(self.visual_encoder(image))
                control_words = ["a photo of"] * len(embeds)
                control_tokens = self.t5_tokenizer(
                    control_words,
                    padding="longest",
                    truncation=True,
                    max_length=self.max_txt_len,
                    return_tensors="pt",
                ).to(embeds.device)
                control_embeds = self.t5_model.encoder.embed_tokens(control_tokens.input_ids)

            with self.maybe_autocast(dtype=torch.bfloat16):
                object_atts = torch.ones(embeds.size()[:-1], dtype=torch.long).to(
                    image.device
                )
                query_tokens = self.query_tokens.expand(embeds.shape[0], -1, -1)
                query_output = self.Qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=embeds,
                    encoder_attention_mask=object_atts,
                    return_dict=True,
                )
                inputs_t5 = self.t5_proj(query_output.last_hidden_state)
                atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(image.device)
                encoder_atts = torch.cat([atts_t5, control_tokens.attention_mask], dim=1)
                inputs_embeds = torch.cat([inputs_t5, control_embeds], dim=1)

                output_tokens = self.t5_tokenizer(
                    samples["caps"],
                    padding="longest",
                    truncation=True,
                    max_length=self.max_txt_len,
                    return_tensors="pt",
                ).to(inputs_embeds.device)

                targets = output_tokens.input_ids.masked_fill(
                    output_tokens.input_ids == self.t5_tokenizer.pad_token_id, -100)

                outputs = self.t5_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=encoder_atts,
                    decoder_attention_mask=output_tokens.attention_mask,
                    return_dict=True,
                    labels=targets,
                )
                loss_llm = outputs.loss

                return {"loss": loss_llm, "loss_llm": loss_llm.detach(), "loss_tag": 0}

        def predict_answers(
                self,
                samples,
                num_beams=5,
                inference_method="generate",
                max_len=10,
                min_len=1,
                num_ans_candidates=128,
                answer_list=None,
                prompt="",
                length_penalty=-1,
                **kwargs
        ):
            image = samples["region_images"]

            with self.maybe_autocast(dtype=torch.float16):
                embeds = self.ln_vision(self.visual_encoder(image))
                control_words = ["a photo of"] * len(embeds)
                control_tokens = self.t5_tokenizer(
                    control_words,
                    padding="longest",
                    truncation=True,
                    max_length=self.max_txt_len,
                    return_tensors="pt",
                ).to(embeds.device)
                control_embeds = self.t5_model.encoder.embed_tokens(control_tokens.input_ids)

            with self.maybe_autocast(dtype=torch.bfloat16):
                object_atts = torch.ones(embeds.size()[:-1], dtype=torch.long).to(
                    image.device
                )
                query_tokens = self.query_tokens.expand(embeds.shape[0], -1, -1)
                query_output = self.Qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=embeds,
                    encoder_attention_mask=object_atts,
                    return_dict=True,
                )
                inputs_t5 = self.t5_proj(query_output.last_hidden_state)
                atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(image.device)
                encoder_atts = torch.cat([atts_t5, control_tokens.attention_mask], dim=1)
                inputs_embeds = torch.cat([inputs_t5, control_embeds], dim=1)

                outputs = self.t5_model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=encoder_atts,
                    do_sample=False,
                    num_beams=num_beams,
                    max_new_tokens=max_len,
                    min_length=min_len,
                    length_penalty=length_penalty,
                    num_return_sequences=1,
                    output_scores=True,
                    return_dict_in_generate=True
                )

                sequences = outputs["sequences"]
                scores = outputs["sequences_scores"]
                scores = torch.exp(scores)
                l = sequences.shape[1]
                sequences = sequences.reshape(-1, l)
                scores = scores.reshape(-1).cpu().numpy().tolist()
                captions = self.t5_tokenizer.batch_decode(
                    sequences, skip_special_tokens=True
                )

            if self._apply_lemmatizer:
                captions = self._lemmatize(captions)

            output = []
            for id, caption, score in zip(samples["ids"], captions, scores):
                output.append(
                    {"id": id, "caption": caption, "score": score}
                )

            return output

        @classmethod
        def from_config(cls, cfg):
            model = cls(**cfg)
            if cfg.pretrained is not None:
                model.load_checkpoint(url_or_filename=cfg.pretrained)
            return model