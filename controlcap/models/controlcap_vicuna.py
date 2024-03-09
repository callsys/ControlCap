import time
import cv2
import math
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import random
import copy
from functools import partial
from textblob import TextBlob
from torchvision.models.vision_transformer import MLPBlock

from reem.models.tag_heads.bert import BertConfig, BertModel
from reem.models.tag_heads.asymmetric_loss import AsymmetricLoss
from reem.models.blip2_models.blip2_vicuna_instruct import Blip2VicunaInstruct
from reem.models.cve_vit.cve_vit import create_cve_vit_g
from lavis.common.registry import registry
from lavis.models.eva_vit import create_eva_vit_g

from peft import LoraConfig, get_peft_model, TaskType

version = "controlcap"

if version == "baseline":
    @registry.register_model("reem_vicuna")
    class ReemVicuna(Blip2VicunaInstruct):
        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs
            base_kwargs = copy.deepcopy(kwargs)
            base_kwargs_keys = ["vit_model", "img_size", "drop_path_rate", "use_grad_checkpoint", "freeze_vit",
                                "num_query_token", "llm_model", "prompt", "max_txt_len", "max_output_txt_len",
                                "qformer_text_input"]
            for key in kwargs.keys():
                if key not in base_kwargs_keys:
                    base_kwargs.pop(key)
            super().__init__(*args, **base_kwargs)


            # Contextual visual embedding module
            self.roi_size = (16, 16)
            self._roi_align = torchvision.ops.RoIAlign(output_size=self.roi_size, spatial_scale=1 / 14,
                                                       sampling_ratio=2)

            self.cve_cap_mlp = nn.Sequential(nn.Linear(self.visual_encoder.embed_dim*2, self.visual_encoder.embed_dim),
                                             nn.GELU(),
                                             nn.Linear(self.visual_encoder.embed_dim, self.visual_encoder.embed_dim))


            # Trainable parameters
            names = ["cve", "Qformer", "llm_proj"]
            trainable_params = 0
            all_params = 0
            for param_name, param in self.named_parameters():
                all_params += param.numel()
                param.requires_grad = False
                for name in names:
                    if name in param_name:
                        param.requires_grad = True
                        trainable_params += param.numel()
                        break
            print(f"[all params : {all_params}][trainable params : {trainable_params}][ratio : {trainable_params/all_params}]")

        @classmethod
        def from_config(cls, cfg):
            model = cls(**cfg)
            if cfg.pretrained is not None:
                model.load_checkpoint(url_or_filename=cfg.pretrained)
            return model

        def sequence2spatio(self, embeds):
            spatio_image_embeds = embeds[:, 1:]
            cls_embeds = embeds[:, 0][:, None]
            b, hw, c = spatio_image_embeds.shape
            h, w = int(math.sqrt(hw)), int(math.sqrt(hw))
            spatio_image_embeds = spatio_image_embeds.reshape(b, h, w, c).permute(0, 3, 1, 2)
            return spatio_image_embeds, cls_embeds

        def spatio2sequence(self, spatio_image_embeds, cls_embeds):
            b, c, h, w = spatio_image_embeds.shape
            spatio_image_embeds = spatio_image_embeds.permute(0, 2, 3, 1).reshape(b, -1, c)
            return torch.cat([cls_embeds, spatio_image_embeds], 1)

        def roi_align(self, image_embeds, samples):
            # prepare spatio image embeddings
            spatio_image_embeds, rois_cls_embeds = self.sequence2spatio(image_embeds)

            # mask the padding bbox of the recognition task to save memory
            bboxes = samples["bboxes"]
            ids = samples["batch_idx"].to(torch.int64)
            rois = torch.cat([ids[:, None], bboxes], -1)

            # instance level feature encoder
            rois_embeds = self._roi_align(spatio_image_embeds, rois)
            rois_cls_embeds = rois_cls_embeds[ids]

            # back to sequence
            rois_embeds = self.spatio2sequence(rois_embeds, rois_cls_embeds)
            return rois_embeds

        def cve_forward(self, samples, embeds):
            bz = len(samples["image"])
            image_embeds = embeds[:bz]
            region_embeds = embeds[bz:]

            # local-roi fusion
            rois_embeds = self.roi_align(image_embeds, samples)
            object_embeds = torch.cat([rois_embeds, region_embeds], -1)
            cap_embeds = self.cve_cap_mlp(object_embeds)

            return cap_embeds

        def cap_forward(self, samples, cap_embeds, control_tags):
            if self.training:
                device = cap_embeds.device
                cap_atts = torch.ones(cap_embeds.size()[:-1], dtype=torch.long).to(device)
                query_tokens = self.query_tokens.expand(cap_embeds.shape[0], -1, -1)

                if self.qformer_text_input:
                    text_Qformer = self.tokenizer(
                        control_tags,
                        padding='longest',
                        truncation=True,
                        max_length=self.max_txt_len,
                        return_tensors="pt",
                    ).to(cap_embeds.device)
                    query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(device)
                    Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask], dim=1)

                    query_output = self.Qformer.bert(
                        text_Qformer.input_ids,
                        attention_mask=Qformer_atts,
                        query_embeds=query_tokens,
                        encoder_hidden_states=cap_embeds,
                        encoder_attention_mask=cap_atts,
                        return_dict=True,
                    )
                else:
                    query_output = self.Qformer.bert(
                        query_embeds=query_tokens,
                        encoder_hidden_states=cap_embeds,
                        encoder_attention_mask=cap_atts,
                        return_dict=True,
                    )

                inputs_llm = self.llm_proj(query_output.last_hidden_state[:, :query_tokens.size(1), :])
                atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(device)

                self.llm_tokenizer.padding_side = "right"
                self.llm_tokenizer.truncation_side = 'left'
                text_input_tokens = self.llm_tokenizer(
                    control_tags,
                    return_tensors="pt",
                    padding="longest",
                    truncation=True,
                    max_length=self.max_txt_len,
                ).to(device)

                self.llm_tokenizer.truncation_side = 'right'
                text_output_tokens = self.llm_tokenizer(
                    [t + self.llm_tokenizer.eos_token for t in samples['caps']],
                    return_tensors="pt",
                    padding="longest",
                    truncation=True,
                    max_length=self.max_output_txt_len,
                ).to(device)

                llm_tokens, input_part_targets_len = self.concat_text_input_output(
                    text_input_tokens.input_ids,
                    text_input_tokens.attention_mask,
                    text_output_tokens.input_ids,
                    text_output_tokens.attention_mask,
                )

                # do not apply loss to the padding
                targets = llm_tokens['input_ids'].masked_fill(
                    llm_tokens['input_ids'] == self.llm_tokenizer.pad_token_id, -100
                )

                # do not apply loss to the text input (i.e., instruction)
                for i, l in enumerate(input_part_targets_len):
                    targets[i][:l] = -100

                # do not apply loss to the query tokens
                empty_targets = (
                    torch.ones(atts_llm.size(), dtype=torch.long).to(device).fill_(-100)
                )
                targets = torch.cat([empty_targets, targets], dim=1)

                inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens['input_ids'])
                inputs_embeds = torch.cat([inputs_llm, inputs_embeds], dim=1)
                attention_mask = torch.cat([atts_llm, llm_tokens['attention_mask']], dim=1)

                with self.maybe_autocast():
                    outputs = self.llm_model(
                        inputs_embeds=inputs_embeds,
                        attention_mask=attention_mask,
                        return_dict=True,
                        labels=targets,
                    )

                loss = outputs.loss

                return loss
            else:
                device = cap_embeds.device
                self.llm_tokenizer.padding_side = "left"
                bs = cap_embeds.size(0)
                cap_atts = torch.ones(cap_embeds.size()[:-1], dtype=torch.long).to(device)
                query_tokens = self.query_tokens.expand(bs, -1, -1)
                if self.qformer_text_input:
                    text_Qformer = self.tokenizer(
                        control_tags,
                        padding='longest',
                        truncation=True,
                        max_length=self.max_txt_len,
                        return_tensors="pt",
                    ).to(cap_embeds.device)
                    query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(device)
                    Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask], dim=1)

                    query_output = self.Qformer.bert(
                        text_Qformer.input_ids,
                        attention_mask=Qformer_atts,
                        query_embeds=query_tokens,
                        encoder_hidden_states=cap_embeds,
                        encoder_attention_mask=cap_atts,
                        return_dict=True,
                    )
                else:
                    query_output = self.Qformer.bert(
                        query_embeds=query_tokens,
                        encoder_hidden_states=cap_embeds,
                        encoder_attention_mask=cap_atts,
                        return_dict=True,
                    )

                inputs_llm = self.llm_proj(query_output.last_hidden_state[:, :query_tokens.size(1), :])
                atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(device)

                llm_tokens = self.llm_tokenizer(
                    control_tags,
                    padding="longest",
                    return_tensors="pt"
                ).to(device)

                with self.maybe_autocast():
                    inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens.input_ids)
                    inputs_embeds = torch.cat([inputs_llm, inputs_embeds], dim=1)
                    attention_mask = torch.cat([atts_llm, llm_tokens.attention_mask], dim=1)

                    llm_kwargs = copy.deepcopy(self.kwargs)
                    llm_kwargs_keys = ["do_sample", "top_p", "temperature", "num_beams", "max_length",
                                       "min_length", "repetition_penalty", "length_penalty", "num_return_sequences"]
                    for key in self.kwargs.keys():
                        if key not in llm_kwargs_keys:
                            llm_kwargs.pop(key)

                    outputs = self.llm_model.generate(
                        inputs_embeds=inputs_embeds,
                        attention_mask=attention_mask,
                        output_scores=True,
                        return_dict_in_generate=True,
                        **llm_kwargs
                    )

                sequences = outputs.sequences
                scores = outputs.sequences_scores
                scores = torch.exp(scores).cpu().numpy().tolist()

                sequences[sequences == 0] = 2  # convert output id 0 to 2 (eos_token_id)
                captions = self.llm_tokenizer.batch_decode(sequences, skip_special_tokens=True)
                captions = [caption.strip() for caption in captions]

                return scores, captions

        def forward(self, samples):
            image = torch.cat([samples["image"], samples["region_images"]], 0)

            with self.maybe_autocast():
                embeds = self.ln_vision(self.visual_encoder(image))

            cap_embeds = self.cve_forward(samples, embeds)

            control_tags = [""] * len(cap_embeds)

            loss_llm = self.cap_forward(samples, cap_embeds, control_tags)

            return {"loss": loss_llm, "loss_llm": loss_llm.detach(), "loss_tag": 0}

        def predict_answers_memory_efficient(self, samples, **kwargs):
            split_size = int(self.kwargs.get("split_size", int(1e7)))
            l_samples = len(samples["ids"])
            chunk_samples = []
            if l_samples <= split_size:
                chunk_samples = [samples]
            else:
                idxs = torch.LongTensor(range(l_samples))
                chunk_idxs = torch.split(idxs, split_size_or_sections=split_size)
                for chunk_idx in chunk_idxs:
                    chunk_sample = dict()
                    for key, value in samples.items():
                        if len(value) != l_samples:
                            chunk_sample[key] = value
                        elif isinstance(value, list):
                            chunk_sample[key] = [value[idx] for idx in chunk_idx]
                        else:
                            chunk_sample[key] = value[chunk_idx]
                    chunk_samples.append(chunk_sample)

            output = []
            for chunk_sample in chunk_samples:
                result = self.predict_answers(chunk_sample, **kwargs)
                output.extend(result)

            return output

        def predict_answers(self, samples, **kwargs):
            split_size = int(self.kwargs.get("split_size", int(1e7)))
            l_samples = len(samples["ids"])
            if l_samples > split_size:
                return self.predict_answers_memory_efficient(samples, **kwargs)

            image = torch.cat([samples["image"], samples["region_images"]], 0)

            with self.maybe_autocast():
                embeds = self.ln_vision(self.visual_encoder(image))

            cap_embeds = self.cve_forward(samples, embeds)

            control_tags = [""] * len(cap_embeds)

            scores, captions = self.cap_forward(samples, cap_embeds, control_tags)

            if self.kwargs.get("apply_lemmatizer", False):
                captions = self._lemmatize(captions)

            # format the predictions
            name = "reem_vicuna"
            output = []
            for id, caption, score in zip(samples["ids"], captions, scores):
                output.append(
                    {"id": id, "caption": {name: [{"caption": caption, "score": score}]}}
                )

            # # format the predictions
            # name = "reem_vicuna"
            # output = []
            # for id, stag, tag, caption, score in zip(samples["ids"], stags, tags, captions, scores):
            #     output.append(
            #         {"id": id, "caption": caption, "score": score, "tag": {"subj": stag, "full": tag}, "annotator": name}
            #     )

            return output

if version == "baseline with cvit":
    @registry.register_model("reem_vicuna")
    class ReemVicuna(Blip2VicunaInstruct):
        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs
            base_kwargs = copy.deepcopy(kwargs)
            base_kwargs_keys = ["vit_model", "img_size", "drop_path_rate", "use_grad_checkpoint", "freeze_vit",
                                "num_query_token", "llm_model", "prompt", "max_txt_len", "max_output_txt_len",
                                "qformer_text_input"]
            for key in kwargs.keys():
                if key not in base_kwargs_keys:
                    base_kwargs.pop(key)
            super().__init__(*args, **base_kwargs)

            # Trainable parameters
            names = ["cve", "Qformer", "llm_proj"]
            trainable_params = 0
            all_params = 0
            for param_name, param in self.named_parameters():
                all_params += param.numel()
                param.requires_grad = False
                for name in names:
                    if name in param_name:
                        param.requires_grad = True
                        trainable_params += param.numel()
                        break
            print(f"[all params : {all_params}][trainable params : {trainable_params}][ratio : {trainable_params/all_params}]")

        @classmethod
        def init_vision_encoder(
                cls, model_name, img_size, drop_path_rate, use_grad_checkpoint, precision
        ):
            assert model_name in [
                "eva_clip_g",
                "clip_L",
            ], "vit model must be eva_clip_g or clip_L"
            visual_encoder = create_cve_vit_g(
                img_size, drop_path_rate, use_grad_checkpoint, precision
            )
            visual_encoder.to(torch.float32)
            ln_vision = nn.LayerNorm(visual_encoder.num_features)
            return visual_encoder, ln_vision

        @classmethod
        def from_config(cls, cfg):
            model = cls(**cfg)
            if cfg.pretrained is not None:
                model.load_checkpoint(url_or_filename=cfg.pretrained)
            return model

        def cap_forward(self, samples, cap_embeds, control_tags):
            if self.training:
                device = cap_embeds.device
                cap_atts = torch.ones(cap_embeds.size()[:-1], dtype=torch.long).to(device)
                query_tokens = self.query_tokens.expand(cap_embeds.shape[0], -1, -1)

                if self.qformer_text_input:
                    text_Qformer = self.tokenizer(
                        control_tags,
                        padding='longest',
                        truncation=True,
                        max_length=self.max_txt_len,
                        return_tensors="pt",
                    ).to(cap_embeds.device)
                    query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(device)
                    Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask], dim=1)

                    query_output = self.Qformer.bert(
                        text_Qformer.input_ids,
                        attention_mask=Qformer_atts,
                        query_embeds=query_tokens,
                        encoder_hidden_states=cap_embeds,
                        encoder_attention_mask=cap_atts,
                        return_dict=True,
                    )
                else:
                    query_output = self.Qformer.bert(
                        query_embeds=query_tokens,
                        encoder_hidden_states=cap_embeds,
                        encoder_attention_mask=cap_atts,
                        return_dict=True,
                    )

                inputs_llm = self.llm_proj(query_output.last_hidden_state[:, :query_tokens.size(1), :])
                atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(device)

                self.llm_tokenizer.padding_side = "right"
                self.llm_tokenizer.truncation_side = 'left'
                text_input_tokens = self.llm_tokenizer(
                    control_tags,
                    return_tensors="pt",
                    padding="longest",
                    truncation=True,
                    max_length=self.max_txt_len,
                ).to(device)

                self.llm_tokenizer.truncation_side = 'right'
                text_output_tokens = self.llm_tokenizer(
                    [t + self.llm_tokenizer.eos_token for t in samples['caps']],
                    return_tensors="pt",
                    padding="longest",
                    truncation=True,
                    max_length=self.max_output_txt_len,
                ).to(device)

                llm_tokens, input_part_targets_len = self.concat_text_input_output(
                    text_input_tokens.input_ids,
                    text_input_tokens.attention_mask,
                    text_output_tokens.input_ids,
                    text_output_tokens.attention_mask,
                )

                # do not apply loss to the padding
                targets = llm_tokens['input_ids'].masked_fill(
                    llm_tokens['input_ids'] == self.llm_tokenizer.pad_token_id, -100
                )

                # do not apply loss to the text input (i.e., instruction)
                for i, l in enumerate(input_part_targets_len):
                    targets[i][:l] = -100

                # do not apply loss to the query tokens
                empty_targets = (
                    torch.ones(atts_llm.size(), dtype=torch.long).to(device).fill_(-100)
                )
                targets = torch.cat([empty_targets, targets], dim=1)

                inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens['input_ids'])
                inputs_embeds = torch.cat([inputs_llm, inputs_embeds], dim=1)
                attention_mask = torch.cat([atts_llm, llm_tokens['attention_mask']], dim=1)

                with self.maybe_autocast():
                    outputs = self.llm_model(
                        inputs_embeds=inputs_embeds,
                        attention_mask=attention_mask,
                        return_dict=True,
                        labels=targets,
                    )

                loss = outputs.loss

                return loss
            else:
                device = cap_embeds.device
                self.llm_tokenizer.padding_side = "left"
                bs = cap_embeds.size(0)
                cap_atts = torch.ones(cap_embeds.size()[:-1], dtype=torch.long).to(device)
                query_tokens = self.query_tokens.expand(bs, -1, -1)
                if self.qformer_text_input:
                    text_Qformer = self.tokenizer(
                        control_tags,
                        padding='longest',
                        truncation=True,
                        max_length=self.max_txt_len,
                        return_tensors="pt",
                    ).to(cap_embeds.device)
                    query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(device)
                    Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask], dim=1)

                    query_output = self.Qformer.bert(
                        text_Qformer.input_ids,
                        attention_mask=Qformer_atts,
                        query_embeds=query_tokens,
                        encoder_hidden_states=cap_embeds,
                        encoder_attention_mask=cap_atts,
                        return_dict=True,
                    )
                else:
                    query_output = self.Qformer.bert(
                        query_embeds=query_tokens,
                        encoder_hidden_states=cap_embeds,
                        encoder_attention_mask=cap_atts,
                        return_dict=True,
                    )

                inputs_llm = self.llm_proj(query_output.last_hidden_state[:, :query_tokens.size(1), :])
                atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(device)

                llm_tokens = self.llm_tokenizer(
                    control_tags,
                    padding="longest",
                    return_tensors="pt"
                ).to(device)

                with self.maybe_autocast():
                    inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens.input_ids)
                    inputs_embeds = torch.cat([inputs_llm, inputs_embeds], dim=1)
                    attention_mask = torch.cat([atts_llm, llm_tokens.attention_mask], dim=1)

                    llm_kwargs = copy.deepcopy(self.kwargs)
                    llm_kwargs_keys = ["do_sample", "top_p", "temperature", "num_beams", "max_length",
                                       "min_length", "repetition_penalty", "length_penalty", "num_return_sequences"]
                    for key in self.kwargs.keys():
                        if key not in llm_kwargs_keys:
                            llm_kwargs.pop(key)

                    outputs = self.llm_model.generate(
                        inputs_embeds=inputs_embeds,
                        attention_mask=attention_mask,
                        output_scores=True,
                        return_dict_in_generate=True,
                        **llm_kwargs
                    )

                sequences = outputs.sequences
                scores = outputs.sequences_scores
                scores = torch.exp(scores).cpu().numpy().tolist()

                sequences[sequences == 0] = 2  # convert output id 0 to 2 (eos_token_id)
                captions = self.llm_tokenizer.batch_decode(sequences, skip_special_tokens=True)
                captions = [caption.strip() for caption in captions]

                return scores, captions

        def forward(self, samples):
            image = torch.cat([samples["image"], samples["region_images"]], 0)

            with self.maybe_autocast():
                cap_embeds = self.ln_vision(self.visual_encoder(image, samples))

            control_tags = [""] * len(cap_embeds)

            loss_llm = self.cap_forward(samples, cap_embeds, control_tags)

            return {"loss": loss_llm, "loss_llm": loss_llm.detach(), "loss_tag": 0}

        def predict_answers_memory_efficient(self, samples, **kwargs):
            split_size = int(self.kwargs.get("split_size", int(1e7)))
            l_samples = len(samples["ids"])
            chunk_samples = []
            if l_samples <= split_size:
                chunk_samples = [samples]
            else:
                idxs = torch.LongTensor(range(l_samples))
                chunk_idxs = torch.split(idxs, split_size_or_sections=split_size)
                for chunk_idx in chunk_idxs:
                    chunk_sample = dict()
                    for key, value in samples.items():
                        if len(value) != l_samples:
                            chunk_sample[key] = value
                        elif isinstance(value, list):
                            chunk_sample[key] = [value[idx] for idx in chunk_idx]
                        else:
                            chunk_sample[key] = value[chunk_idx]
                    chunk_samples.append(chunk_sample)

            output = []
            for chunk_sample in chunk_samples:
                result = self.predict_answers(chunk_sample, **kwargs)
                output.extend(result)

            return output

        def predict_answers(self, samples, **kwargs):
            split_size = int(self.kwargs.get("split_size", int(1e7)))
            l_samples = len(samples["ids"])
            if l_samples > split_size:
                return self.predict_answers_memory_efficient(samples, **kwargs)

            image = torch.cat([samples["image"], samples["region_images"]], 0)

            with self.maybe_autocast():
                cap_embeds = self.ln_vision(self.visual_encoder(image))

            control_tags = [""] * len(cap_embeds)

            scores, captions = self.cap_forward(samples, cap_embeds, control_tags)

            if self.kwargs.get("apply_lemmatizer", False):
                captions = self._lemmatize(captions)

            # format the predictions
            name = "reem_vicuna"
            output = []
            for id, caption, score in zip(samples["ids"], captions, scores):
                output.append(
                    {"id": id, "caption": {name: [{"caption": caption, "score": score}]}}
                )

            # # format the predictions
            # name = "reem_vicuna"
            # output = []
            # for id, stag, tag, caption, score in zip(samples["ids"], stags, tags, captions, scores):
            #     output.append(
            #         {"id": id, "caption": caption, "score": score, "tag": {"subj": stag, "full": tag}, "annotator": name}
            #     )

            return output

if version == "baseline with tagging":
    @registry.register_model("reem_vicuna")
    class ReemVicuna(Blip2VicunaInstruct):
        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs
            base_kwargs = copy.deepcopy(kwargs)
            base_kwargs_keys = ["vit_model", "img_size", "drop_path_rate", "use_grad_checkpoint", "freeze_vit",
                                "num_query_token", "llm_model", "prompt", "max_txt_len", "max_output_txt_len",
                                "qformer_text_input"]
            for key in kwargs.keys():
                if key not in base_kwargs_keys:
                    base_kwargs.pop(key)
            super().__init__(*args, **base_kwargs)


            # Contextual visual embedding module
            self.roi_size = (16, 16)
            self._roi_align = torchvision.ops.RoIAlign(output_size=self.roi_size, spatial_scale=1 / 14,
                                                       sampling_ratio=2)

            self.cve_cap_mlp = nn.Sequential(nn.Linear(self.visual_encoder.embed_dim*2, self.visual_encoder.embed_dim),
                                             nn.GELU(),
                                             nn.Linear(self.visual_encoder.embed_dim, self.visual_encoder.embed_dim))
            self.cve_tag_mlp = nn.Sequential(nn.Linear(self.visual_encoder.embed_dim*2, self.visual_encoder.embed_dim),
                                             nn.GELU(),
                                             nn.Linear(self.visual_encoder.embed_dim, self.visual_encoder.embed_dim))


            # Tag Head same as tag2text, query2label
            tag_bert_config = BertConfig.from_json_file(kwargs.get("tag_bert_config", "reem/models/tag_heads/tag_bert_config.json"))
            tag_bert_config.encoder_width = self.Qformer.config.encoder_width
            self.tag_head = BertModel(config=tag_bert_config, add_pooling_layer=False)
            del self.tag_head.embeddings
            for layer in self.tag_head.encoder.layer:
                del layer.attention
            tag_list = kwargs.get("tag_list", "reem/commom/tag_parser/ram_tag_list.txt")
            with open(tag_list, "r") as fr:
                self.tag_list = fr.readlines()
            self.tag_list = [tag.strip() for tag in self.tag_list]
            self.num_tags = len(self.tag_list)
            self.tag_labels = nn.Embedding(self.num_tags, tag_bert_config.hidden_size)
            self.tag_fc = nn.Linear(tag_bert_config.hidden_size, 1)
            self.tag_weight = 0.01
            self.tag_loss_function = AsymmetricLoss(gamma_neg=7, gamma_pos=0, clip=0.05)


            # Trainable parameters
            names = ["tag", "cve", "cee", "Qformer", "llm_proj"]
            trainable_params = 0
            all_params = 0
            for param_name, param in self.named_parameters():
                all_params += param.numel()
                param.requires_grad = False
                for name in names:
                    if name in param_name:
                        param.requires_grad = True
                        trainable_params += param.numel()
                        break
            print(f"[all params : {all_params}][trainable params : {trainable_params}][ratio : {trainable_params/all_params}]")

        @classmethod
        def from_config(cls, cfg):
            model = cls(**cfg)
            if cfg.pretrained is not None:
                model.load_checkpoint(url_or_filename=cfg.pretrained)
            return model

        def sequence2spatio(self, embeds):
            spatio_image_embeds = embeds[:, 1:]
            cls_embeds = embeds[:, 0][:, None]
            b, hw, c = spatio_image_embeds.shape
            h, w = int(math.sqrt(hw)), int(math.sqrt(hw))
            spatio_image_embeds = spatio_image_embeds.reshape(b, h, w, c).permute(0, 3, 1, 2)
            return spatio_image_embeds, cls_embeds

        def spatio2sequence(self, spatio_image_embeds, cls_embeds):
            b, c, h, w = spatio_image_embeds.shape
            spatio_image_embeds = spatio_image_embeds.permute(0, 2, 3, 1).reshape(b, -1, c)
            return torch.cat([cls_embeds, spatio_image_embeds], 1)

        def roi_align(self, image_embeds, samples):
            # prepare spatio image embeddings
            spatio_image_embeds, rois_cls_embeds = self.sequence2spatio(image_embeds)

            # mask the padding bbox of the recognition task to save memory
            bboxes = samples["bboxes"]
            ids = samples["batch_idx"].to(torch.int64)
            rois = torch.cat([ids[:, None], bboxes], -1)

            # instance level feature encoder
            rois_embeds = self._roi_align(spatio_image_embeds, rois)
            rois_cls_embeds = rois_cls_embeds[ids]

            # back to sequence
            rois_embeds = self.spatio2sequence(rois_embeds, rois_cls_embeds)
            return rois_embeds

        def prepare_control_tags(self, samples, tag_logits, drop_ratio=0.5, tag_thr=0.7):
            control_tags = []

            if self.training:
                tag_idxs = samples["tags"]
                stags = [[self.tag_list[tag_idx] for tag_idx in torch.nonzero(tag_idxs[bz_idx][:self.num_tags])]
                         for bz_idx in range(len(tag_idxs))]
                otags = [[self.tag_list[tag_idx] for tag_idx in torch.nonzero(tag_idxs[bz_idx][self.num_tags:])]
                         for bz_idx in range(len(tag_idxs))]
                tags = [stag + otag for stag, otag in zip(stags, otags)]

                for tag in tags:
                    l = len(tag)
                    if l==0:
                        control_tag = "|"
                    else:
                        sl = torch.from_numpy(np.random.uniform(0,1,l)>drop_ratio)
                        control_tag = [tag[tag_idx] for tag_idx in torch.nonzero(sl)]
                        random.shuffle(control_tag)
                        control_tag = ",".join(control_tag) + "|"
                    control_tags.append(control_tag)
            else:
                tag_scores = tag_logits.sigmoid()
                tag_idxs = (tag_scores > tag_thr).to(torch.long)
                tags = [[self.tag_list[tag_idx] for tag_idx in torch.nonzero(tag_idxs[bz_idx])]
                         for bz_idx in range(len(tag_idxs))]

                for tag in tags:
                    control_tag = ",".join(tag) + "|"
                    control_tags.append(control_tag)

            return control_tags, tags, tags

        def cve_forward(self, samples, embeds):
            bz = len(samples["image"])
            image_embeds = embeds[:bz]
            region_embeds = embeds[bz:]

            # local-roi fusion
            rois_embeds = self.roi_align(image_embeds, samples)
            object_embeds = torch.cat([rois_embeds, region_embeds], -1)
            cap_embeds = self.cve_cap_mlp(object_embeds)
            tag_embeds = self.cve_tag_mlp(object_embeds)

            return cap_embeds, tag_embeds

        def tag_forward(self, samples, tag_embeds):
            bs = len(tag_embeds)
            object_atts = torch.ones(tag_embeds.size()[:-1], dtype=torch.long).to(
                tag_embeds.device
            )
            label_embed = self.tag_labels.weight.unsqueeze(0).repeat(bs, 1, 1)

            tagging_embed = self.tag_head(
                encoder_embeds=label_embed,
                encoder_hidden_states=tag_embeds,
                encoder_attention_mask=object_atts,
                return_dict=False,
                mode='tagging',
            )
            tag_logits = self.tag_fc(tagging_embed[0]).squeeze(-1)
            return tag_logits

        def cap_forward(self, samples, cap_embeds, control_tags):
            if self.training:
                device = cap_embeds.device
                cap_atts = torch.ones(cap_embeds.size()[:-1], dtype=torch.long).to(device)
                query_tokens = self.query_tokens.expand(cap_embeds.shape[0], -1, -1)

                if self.qformer_text_input:
                    text_Qformer = self.tokenizer(
                        control_tags,
                        padding='longest',
                        truncation=True,
                        max_length=self.max_txt_len,
                        return_tensors="pt",
                    ).to(cap_embeds.device)
                    query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(device)
                    Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask], dim=1)

                    query_output = self.Qformer.bert(
                        text_Qformer.input_ids,
                        attention_mask=Qformer_atts,
                        query_embeds=query_tokens,
                        encoder_hidden_states=cap_embeds,
                        encoder_attention_mask=cap_atts,
                        return_dict=True,
                    )
                else:
                    query_output = self.Qformer.bert(
                        query_embeds=query_tokens,
                        encoder_hidden_states=cap_embeds,
                        encoder_attention_mask=cap_atts,
                        return_dict=True,
                    )

                inputs_llm = self.llm_proj(query_output.last_hidden_state[:, :query_tokens.size(1), :])
                atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(device)

                self.llm_tokenizer.padding_side = "right"
                self.llm_tokenizer.truncation_side = 'left'
                text_input_tokens = self.llm_tokenizer(
                    control_tags,
                    return_tensors="pt",
                    padding="longest",
                    truncation=True,
                    max_length=self.max_txt_len,
                ).to(device)

                self.llm_tokenizer.truncation_side = 'right'
                text_output_tokens = self.llm_tokenizer(
                    [t + self.llm_tokenizer.eos_token for t in samples['caps']],
                    return_tensors="pt",
                    padding="longest",
                    truncation=True,
                    max_length=self.max_output_txt_len,
                ).to(device)

                llm_tokens, input_part_targets_len = self.concat_text_input_output(
                    text_input_tokens.input_ids,
                    text_input_tokens.attention_mask,
                    text_output_tokens.input_ids,
                    text_output_tokens.attention_mask,
                )

                # do not apply loss to the padding
                targets = llm_tokens['input_ids'].masked_fill(
                    llm_tokens['input_ids'] == self.llm_tokenizer.pad_token_id, -100
                )

                # do not apply loss to the text input (i.e., instruction)
                for i, l in enumerate(input_part_targets_len):
                    targets[i][:l] = -100

                # do not apply loss to the query tokens
                empty_targets = (
                    torch.ones(atts_llm.size(), dtype=torch.long).to(device).fill_(-100)
                )
                targets = torch.cat([empty_targets, targets], dim=1)

                inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens['input_ids'])
                inputs_embeds = torch.cat([inputs_llm, inputs_embeds], dim=1)
                attention_mask = torch.cat([atts_llm, llm_tokens['attention_mask']], dim=1)

                with self.maybe_autocast():
                    outputs = self.llm_model(
                        inputs_embeds=inputs_embeds,
                        attention_mask=attention_mask,
                        return_dict=True,
                        labels=targets,
                    )

                loss = outputs.loss

                return loss
            else:
                device = cap_embeds.device
                self.llm_tokenizer.padding_side = "left"
                bs = cap_embeds.size(0)
                cap_atts = torch.ones(cap_embeds.size()[:-1], dtype=torch.long).to(device)
                query_tokens = self.query_tokens.expand(bs, -1, -1)
                if self.qformer_text_input:
                    text_Qformer = self.tokenizer(
                        control_tags,
                        padding='longest',
                        truncation=True,
                        max_length=self.max_txt_len,
                        return_tensors="pt",
                    ).to(cap_embeds.device)
                    query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(device)
                    Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask], dim=1)

                    query_output = self.Qformer.bert(
                        text_Qformer.input_ids,
                        attention_mask=Qformer_atts,
                        query_embeds=query_tokens,
                        encoder_hidden_states=cap_embeds,
                        encoder_attention_mask=cap_atts,
                        return_dict=True,
                    )
                else:
                    query_output = self.Qformer.bert(
                        query_embeds=query_tokens,
                        encoder_hidden_states=cap_embeds,
                        encoder_attention_mask=cap_atts,
                        return_dict=True,
                    )

                inputs_llm = self.llm_proj(query_output.last_hidden_state[:, :query_tokens.size(1), :])
                atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(device)

                llm_tokens = self.llm_tokenizer(
                    control_tags,
                    padding="longest",
                    return_tensors="pt"
                ).to(device)

                with self.maybe_autocast():
                    inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens.input_ids)
                    inputs_embeds = torch.cat([inputs_llm, inputs_embeds], dim=1)
                    attention_mask = torch.cat([atts_llm, llm_tokens.attention_mask], dim=1)

                    llm_kwargs = copy.deepcopy(self.kwargs)
                    llm_kwargs_keys = ["do_sample", "top_p", "temperature", "num_beams", "max_length",
                                       "min_length", "repetition_penalty", "length_penalty", "num_return_sequences"]
                    for key in self.kwargs.keys():
                        if key not in llm_kwargs_keys:
                            llm_kwargs.pop(key)

                    outputs = self.llm_model.generate(
                        inputs_embeds=inputs_embeds,
                        attention_mask=attention_mask,
                        output_scores=True,
                        return_dict_in_generate=True,
                        **llm_kwargs
                    )

                sequences = outputs.sequences
                scores = outputs.sequences_scores
                scores = torch.exp(scores).cpu().numpy().tolist()

                sequences[sequences == 0] = 2  # convert output id 0 to 2 (eos_token_id)
                captions = self.llm_tokenizer.batch_decode(sequences, skip_special_tokens=True)
                captions = [caption.strip() for caption in captions]

                return scores, captions

        def forward(self, samples):
            image = torch.cat([samples["image"], samples["region_images"]], 0)

            with self.maybe_autocast():
                embeds = self.ln_vision(self.visual_encoder(image))

            cap_embeds, tag_embeds = self.cve_forward(samples, embeds)
            tag_logits = self.tag_forward(samples, tag_embeds)
            control_tags, _, _ = self.prepare_control_tags(samples, tag_logits)

            tags = samples["tags"].to(torch.long)
            tags = tags[:, :self.num_tags] + tags[:, self.num_tags:]
            loss_tag = self.tag_loss_function(tag_logits, tags) * self.tag_weight

            loss_llm = self.cap_forward(samples, cap_embeds, control_tags)

            return {"loss": loss_llm + loss_tag, "loss_llm": loss_llm.detach(), "loss_tag": loss_tag.detach()}

        def predict_answers_memory_efficient(self, samples, **kwargs):
            split_size = int(self.kwargs.get("split_size", int(1e7)))
            l_samples = len(samples["ids"])
            chunk_samples = []
            if l_samples <= split_size:
                chunk_samples = [samples]
            else:
                idxs = torch.LongTensor(range(l_samples))
                chunk_idxs = torch.split(idxs, split_size_or_sections=split_size)
                for chunk_idx in chunk_idxs:
                    chunk_sample = dict()
                    for key, value in samples.items():
                        if len(value) != l_samples:
                            chunk_sample[key] = value
                        elif isinstance(value, list):
                            chunk_sample[key] = [value[idx] for idx in chunk_idx]
                        else:
                            chunk_sample[key] = value[chunk_idx]
                    chunk_samples.append(chunk_sample)

            output = []
            for chunk_sample in chunk_samples:
                result = self.predict_answers(chunk_sample, **kwargs)
                output.extend(result)

            return output

        def predict_answers(self, samples, **kwargs):
            split_size = int(self.kwargs.get("split_size", int(1e7)))
            l_samples = len(samples["ids"])
            if l_samples > split_size:
                return self.predict_answers_memory_efficient(samples, **kwargs)

            image = torch.cat([samples["image"], samples["region_images"]], 0)

            with self.maybe_autocast():
                embeds = self.ln_vision(self.visual_encoder(image))

            cap_embeds, tag_embeds = self.cve_forward(samples, embeds)
            tag_logits = self.tag_forward(samples, tag_embeds)
            control_tags, stags, tags = self.prepare_control_tags(samples, tag_logits)

            scores, captions = self.cap_forward(samples, cap_embeds, control_tags)

            if self.kwargs.get("apply_lemmatizer", False):
                captions = self._lemmatize(captions)

            # format the predictions
            name = "reem_vicuna"
            output = []
            for id, caption, score in zip(samples["ids"], captions, scores):
                output.append(
                    {"id": id, "caption": {name: [{"caption": caption, "score": score}]}}
                )

            # # format the predictions
            # name = "reem_vicuna"
            # output = []
            # for id, stag, tag, caption, score in zip(samples["ids"], stags, tags, captions, scores):
            #     output.append(
            #         {"id": id, "caption": caption, "score": score, "tag": {"subj": stag, "full": tag}, "annotator": name}
            #     )

            return output

if version == "v1":
    @registry.register_model("reem_vicuna")
    class ReemVicuna(Blip2VicunaInstruct):
        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs
            base_kwargs = copy.deepcopy(kwargs)
            base_kwargs_keys = ["vit_model", "img_size", "drop_path_rate", "use_grad_checkpoint", "freeze_vit",
                                "num_query_token", "llm_model", "prompt", "max_txt_len", "max_output_txt_len",
                                "qformer_text_input"]
            for key in kwargs.keys():
                if key not in base_kwargs_keys:
                    base_kwargs.pop(key)
            super().__init__(*args, **base_kwargs)

            # Contextual visual embedding module
            self.roi_size = (16, 16)
            self._roi_align = torchvision.ops.RoIAlign(output_size=self.roi_size, spatial_scale=1 / 14,
                                                       sampling_ratio=2)

            self.cve_cap_mlp = nn.Sequential(
                nn.Linear(self.visual_encoder.embed_dim * 2, self.visual_encoder.embed_dim),
                nn.GELU(),
                nn.Linear(self.visual_encoder.embed_dim, self.visual_encoder.embed_dim))
            self.cve_tag_mlp = nn.Sequential(
                nn.Linear(self.visual_encoder.embed_dim * 2, self.visual_encoder.embed_dim),
                nn.GELU(),
                nn.Linear(self.visual_encoder.embed_dim, self.visual_encoder.embed_dim))

            # Tag Head same as tag2text, query2label
            tag_bert_config = BertConfig.from_json_file(
                kwargs.get("tag_bert_config", "reem/models/tag_heads/tag_bert_config.json"))
            tag_bert_config.encoder_width = self.Qformer.config.encoder_width
            self.tag_head = BertModel(config=tag_bert_config, add_pooling_layer=False)
            del self.tag_head.embeddings
            for layer in self.tag_head.encoder.layer:
                del layer.attention
            tag_list = kwargs.get("tag_list", "reem/commom/tag_parser/ram_tag_list.txt")
            with open(tag_list, "r") as fr:
                self.tag_list = fr.readlines()
            self.tag_list = [tag.strip() for tag in self.tag_list]
            self.num_tags = len(self.tag_list)
            self.tag_labels = nn.Embedding(self.num_tags * 2, tag_bert_config.hidden_size)
            self.tag_fc = nn.Linear(tag_bert_config.hidden_size, 1)
            self.tag_weight = 0.01
            self.tag_loss_function = AsymmetricLoss(gamma_neg=7, gamma_pos=0, clip=0.05)

            # Trainable parameters
            names = ["tag", "cve", "cee", "Qformer", "llm_proj"]
            self.finetune_llm = kwargs.get("finetune_llm", False)
            if self.finetune_llm:
                lora_config = LoraConfig(
                    r=64, lora_alpha=128, lora_dropout=0.0,
                    target_modules=["embed_tokens", "lm_head", "q_proj", "v_proj"]
                )
                self.llm_model = get_peft_model(self.llm_model, lora_config)
                self.llm_model.to(torch.float32)
                names.append("lora")

            trainable_params = 0
            all_params = 0
            for param_name, param in self.named_parameters():
                all_params += param.numel()
                param.requires_grad = False
                for name in names:
                    if name in param_name:
                        param.requires_grad = True
                        trainable_params += param.numel()
                        break
            print(
                f"[all params : {all_params}][trainable params : {trainable_params}][ratio : {trainable_params / all_params}]")

        @classmethod
        def from_config(cls, cfg):
            model = cls(**cfg)
            if cfg.pretrained is not None:
                model.load_checkpoint(url_or_filename=cfg.pretrained)
            return model

        def sequence2spatio(self, embeds):
            spatio_image_embeds = embeds[:, 1:]
            cls_embeds = embeds[:, 0][:, None]
            b, hw, c = spatio_image_embeds.shape
            h, w = int(math.sqrt(hw)), int(math.sqrt(hw))
            spatio_image_embeds = spatio_image_embeds.reshape(b, h, w, c).permute(0, 3, 1, 2)
            return spatio_image_embeds, cls_embeds

        def spatio2sequence(self, spatio_image_embeds, cls_embeds):
            b, c, h, w = spatio_image_embeds.shape
            spatio_image_embeds = spatio_image_embeds.permute(0, 2, 3, 1).reshape(b, -1, c)
            return torch.cat([cls_embeds, spatio_image_embeds], 1)

        def roi_align(self, image_embeds, samples):
            # prepare spatio image embeddings
            spatio_image_embeds, rois_cls_embeds = self.sequence2spatio(image_embeds)

            # mask the padding bbox of the recognition task to save memory
            bboxes = samples["bboxes"]
            ids = samples["batch_idx"].to(torch.int64)
            rois = torch.cat([ids[:, None], bboxes], -1)

            # instance level feature encoder
            rois_embeds = self._roi_align(spatio_image_embeds, rois)
            rois_cls_embeds = rois_cls_embeds[ids]

            # back to sequence
            rois_embeds = self.spatio2sequence(rois_embeds, rois_cls_embeds)
            return rois_embeds

        def prepare_control_tags(self, samples, tag_logits, drop_ratio=0.5, tag_thr=0.75):
            control_tags = []

            if self.training:
                tag_idxs = samples["tags"]
                stags = [[self.tag_list[tag_idx] for tag_idx in torch.nonzero(tag_idxs[bz_idx][:self.num_tags])]
                         for bz_idx in range(len(tag_idxs))]
                otags = [[self.tag_list[tag_idx] for tag_idx in torch.nonzero(tag_idxs[bz_idx][self.num_tags:])]
                         for bz_idx in range(len(tag_idxs))]
                tags = [stag + otag for stag, otag in zip(stags, otags)]

                for tag in tags:
                    l = len(tag)
                    if l == 0:
                        control_tag = "|"
                    else:
                        sl = torch.from_numpy(np.random.uniform(0, 1, l) > drop_ratio)
                        control_tag = [tag[tag_idx] for tag_idx in torch.nonzero(sl)]
                        random.shuffle(control_tag)
                        control_tag = ",".join(control_tag) + "|"
                    # if l>0:
                    #     sl = random.randint(0, l)
                    #     random.shuffle(tag)
                    #     control_tag = tag[:sl]
                    #     control_tag = ",".join(control_tag)
                    # else:
                    #     control_tag = ""
                    control_tags.append(control_tag)
            else:
                tag_scores = tag_logits.sigmoid()
                tag_idxs = (tag_scores > tag_thr).to(torch.long)
                stags = [[self.tag_list[tag_idx] for tag_idx in torch.nonzero(tag_idxs[bz_idx][:self.num_tags])]
                         for bz_idx in range(len(tag_idxs))]
                otags = [[self.tag_list[tag_idx] for tag_idx in torch.nonzero(tag_idxs[bz_idx][self.num_tags:])]
                         for bz_idx in range(len(tag_idxs))]
                tags = [list(set(stag + otag)) for stag, otag in zip(stags, otags)]

                for tag in tags:
                    control_tag = ",".join(tag) + "|"
                    control_tags.append(control_tag)

            return control_tags, stags, tags

        def cve_forward(self, samples, embeds):
            bz = len(samples["image"])
            image_embeds = embeds[:bz]
            region_embeds = embeds[bz:]

            # local-roi fusion
            rois_embeds = self.roi_align(image_embeds, samples)
            object_embeds = torch.cat([rois_embeds, region_embeds], -1)
            cap_embeds = self.cve_cap_mlp(object_embeds)
            tag_embeds = self.cve_tag_mlp(object_embeds)

            return cap_embeds, tag_embeds

        def tag_forward(self, samples, tag_embeds):
            bs = len(tag_embeds)
            object_atts = torch.ones(tag_embeds.size()[:-1], dtype=torch.long).to(
                tag_embeds.device
            )
            label_embed = self.tag_labels.weight.unsqueeze(0).repeat(bs, 1, 1)

            tagging_embed = self.tag_head(
                encoder_embeds=label_embed,
                encoder_hidden_states=tag_embeds,
                encoder_attention_mask=object_atts,
                return_dict=False,
                mode='tagging',
            )
            tag_logits = self.tag_fc(tagging_embed[0]).squeeze(-1)
            return tag_logits

        def cap_forward(self, samples, cap_embeds, control_tags):
            if self.training:
                device = cap_embeds.device
                cap_atts = torch.ones(cap_embeds.size()[:-1], dtype=torch.long).to(device)
                query_tokens = self.query_tokens.expand(cap_embeds.shape[0], -1, -1)

                if self.qformer_text_input:
                    text_Qformer = self.tokenizer(
                        control_tags,
                        padding='longest',
                        truncation=True,
                        max_length=self.max_txt_len,
                        return_tensors="pt",
                    ).to(cap_embeds.device)
                    query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(device)
                    Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask], dim=1)

                    query_output = self.Qformer.bert(
                        text_Qformer.input_ids,
                        attention_mask=Qformer_atts,
                        query_embeds=query_tokens,
                        encoder_hidden_states=cap_embeds,
                        encoder_attention_mask=cap_atts,
                        return_dict=True,
                    )
                else:
                    query_output = self.Qformer.bert(
                        query_embeds=query_tokens,
                        encoder_hidden_states=cap_embeds,
                        encoder_attention_mask=cap_atts,
                        return_dict=True,
                    )

                inputs_llm = self.llm_proj(query_output.last_hidden_state[:, :query_tokens.size(1), :])
                atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(device)

                self.llm_tokenizer.padding_side = "right"
                self.llm_tokenizer.truncation_side = 'left'
                text_input_tokens = self.llm_tokenizer(
                    control_tags,
                    return_tensors="pt",
                    padding="longest",
                    truncation=True,
                    max_length=self.max_txt_len,
                ).to(device)

                self.llm_tokenizer.truncation_side = 'right'
                text_output_tokens = self.llm_tokenizer(
                    [t + self.llm_tokenizer.eos_token for t in samples['caps']],
                    return_tensors="pt",
                    padding="longest",
                    truncation=True,
                    max_length=self.max_output_txt_len,
                ).to(device)

                llm_tokens, input_part_targets_len = self.concat_text_input_output(
                    text_input_tokens.input_ids,
                    text_input_tokens.attention_mask,
                    text_output_tokens.input_ids,
                    text_output_tokens.attention_mask,
                )

                # do not apply loss to the padding
                targets = llm_tokens['input_ids'].masked_fill(
                    llm_tokens['input_ids'] == self.llm_tokenizer.pad_token_id, -100
                )

                # do not apply loss to the text input (i.e., instruction)
                for i, l in enumerate(input_part_targets_len):
                    targets[i][:l] = -100

                # do not apply loss to the query tokens
                empty_targets = (
                    torch.ones(atts_llm.size(), dtype=torch.long).to(device).fill_(-100)
                )
                targets = torch.cat([empty_targets, targets], dim=1)

                with self.maybe_autocast(torch.float16):
                    inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens['input_ids'])
                    inputs_embeds = torch.cat([inputs_llm, inputs_embeds], dim=1)
                    attention_mask = torch.cat([atts_llm, llm_tokens['attention_mask']], dim=1)
                    outputs = self.llm_model(
                        inputs_embeds=inputs_embeds,
                        attention_mask=attention_mask,
                        return_dict=True,
                        labels=targets,
                    )
                loss = outputs.loss

                return loss
            else:
                device = cap_embeds.device
                self.llm_tokenizer.padding_side = "left"
                bs = cap_embeds.size(0)
                cap_atts = torch.ones(cap_embeds.size()[:-1], dtype=torch.long).to(device)
                query_tokens = self.query_tokens.expand(bs, -1, -1)
                if self.qformer_text_input:
                    text_Qformer = self.tokenizer(
                        control_tags,
                        padding='longest',
                        truncation=True,
                        max_length=self.max_txt_len,
                        return_tensors="pt",
                    ).to(cap_embeds.device)
                    query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(device)
                    Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask], dim=1)

                    query_output = self.Qformer.bert(
                        text_Qformer.input_ids,
                        attention_mask=Qformer_atts,
                        query_embeds=query_tokens,
                        encoder_hidden_states=cap_embeds,
                        encoder_attention_mask=cap_atts,
                        return_dict=True,
                    )
                else:
                    query_output = self.Qformer.bert(
                        query_embeds=query_tokens,
                        encoder_hidden_states=cap_embeds,
                        encoder_attention_mask=cap_atts,
                        return_dict=True,
                    )

                inputs_llm = self.llm_proj(query_output.last_hidden_state[:, :query_tokens.size(1), :])
                atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(device)

                llm_tokens = self.llm_tokenizer(
                    control_tags,
                    padding="longest",
                    return_tensors="pt"
                ).to(device)

                with self.maybe_autocast(torch.float16):
                    inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens.input_ids)
                    inputs_embeds = torch.cat([inputs_llm, inputs_embeds], dim=1)
                    attention_mask = torch.cat([atts_llm, llm_tokens.attention_mask], dim=1)

                    llm_kwargs = copy.deepcopy(self.kwargs)
                    llm_kwargs_keys = ["do_sample", "top_p", "temperature", "num_beams", "max_length",
                                       "min_length", "repetition_penalty", "length_penalty", "num_return_sequences"]
                    for key in self.kwargs.keys():
                        if key not in llm_kwargs_keys:
                            llm_kwargs.pop(key)

                    outputs = self.llm_model.generate(
                        inputs_embeds=inputs_embeds,
                        attention_mask=attention_mask,
                        output_scores=True,
                        return_dict_in_generate=True,
                        **llm_kwargs
                    )

                sequences = outputs.sequences
                scores = outputs.sequences_scores
                scores = torch.exp(scores).cpu().numpy().tolist()

                sequences[sequences == 0] = 2  # convert output id 0 to 2 (eos_token_id)
                captions = self.llm_tokenizer.batch_decode(sequences, skip_special_tokens=True)
                captions = [caption.strip() for caption in captions]

                return scores, captions

        def forward(self, samples):
            image = torch.cat([samples["image"], samples["region_images"]], 0)

            with self.maybe_autocast():
                embeds = self.ln_vision(self.visual_encoder(image))

            cap_embeds, tag_embeds = self.cve_forward(samples, embeds)
            tag_logits = self.tag_forward(samples, tag_embeds)
            control_tags, _, _ = self.prepare_control_tags(samples, tag_logits)

            tags = samples["tags"].to(torch.long)
            loss_tag = self.tag_loss_function(tag_logits, tags) * self.tag_weight

            loss_llm = self.cap_forward(samples, cap_embeds, control_tags)

            return {"loss": loss_llm + loss_tag, "loss_llm": loss_llm.detach(), "loss_tag": loss_tag.detach()}

        def predict_answers_memory_efficient(self, samples, **kwargs):
            split_size = int(self.kwargs.get("split_size", int(1e7)))
            l_samples = len(samples["ids"])
            chunk_samples = []
            if l_samples <= split_size:
                chunk_samples = [samples]
            else:
                idxs = torch.LongTensor(range(l_samples))
                chunk_idxs = torch.split(idxs, split_size_or_sections=split_size)
                for chunk_idx in chunk_idxs:
                    chunk_sample = dict()
                    for key, value in samples.items():
                        if len(value) != l_samples:
                            chunk_sample[key] = value
                        elif isinstance(value, list):
                            chunk_sample[key] = [value[idx] for idx in chunk_idx]
                        else:
                            chunk_sample[key] = value[chunk_idx]
                    chunk_samples.append(chunk_sample)

            output = []
            for chunk_sample in chunk_samples:
                result = self.predict_answers(chunk_sample, **kwargs)
                output.extend(result)

            return output

        def predict_answers(self, samples, **kwargs):
            split_size = int(self.kwargs.get("split_size", int(1e7)))
            l_samples = len(samples["ids"])
            if l_samples > split_size:
                return self.predict_answers_memory_efficient(samples, **kwargs)

            image = torch.cat([samples["image"], samples["region_images"]], 0)

            with self.maybe_autocast():
                embeds = self.ln_vision(self.visual_encoder(image))

            cap_embeds, tag_embeds = self.cve_forward(samples, embeds)
            tag_logits = self.tag_forward(samples, tag_embeds)
            control_tags, stags, tags = self.prepare_control_tags(samples, tag_logits)

            scores, captions = self.cap_forward(samples, cap_embeds, control_tags)

            if self.kwargs.get("apply_lemmatizer", False):
                captions = self._lemmatize(captions)

            # format the predictions
            name = "reem_vicuna"
            output = []
            for id, caption, score in zip(samples["ids"], captions, scores):
                output.append(
                    {"id": id, "caption": {name: [{"caption": caption, "score": score}]}}
                )

            # # format the predictions
            # name = "reem_vicuna"
            # output = []
            # for id, stag, tag, caption, score in zip(samples["ids"], stags, tags, captions, scores):
            #     output.append(
            #         {"id": id, "caption": caption, "score": score, "tag": {"subj": stag, "full": tag}, "annotator": name}
            #     )

            return output

if version == "controlcap":
    class CrossAttnBlock(nn.Module):
        def __init__(self,
                     num_heads,
                     hidden_dim,
                     mlp_dim,
                     dropout=0,
                     attention_dropout=0,
                     ):
            super().__init__()
            self.num_heads = num_heads
            norm_layer = partial(nn.LayerNorm, eps=1e-6)

            self.ln_g = norm_layer(hidden_dim)
            self.cross_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=attention_dropout,
                                                         batch_first=True)
            self.dropout = nn.Dropout(dropout)

            self.ln_r = norm_layer(hidden_dim)
            self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)

        def forward(self, query_embeds, source_embeds):
            source_embeds = self.ln_g(source_embeds)
            x, attn = self.cross_attention(query_embeds, source_embeds, source_embeds)
            x = self.dropout(x)
            x = x + query_embeds
            y = self.ln_r(x)
            y = self.mlp(y)
            return x + y, attn

    @registry.register_model("reem_vicuna")
    class ReemVicuna(Blip2VicunaInstruct):
        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs
            base_kwargs = copy.deepcopy(kwargs)
            base_kwargs_keys = ["vit_model", "img_size", "drop_path_rate", "use_grad_checkpoint", "freeze_vit",
                                "num_query_token", "llm_model", "prompt", "max_txt_len", "max_output_txt_len",
                                "qformer_text_input"]
            for key in kwargs.keys():
                if key not in base_kwargs_keys:
                    base_kwargs.pop(key)
            super().__init__(*args, **base_kwargs)

            self.qformer_text_input = True

            # Contextual visual embedding module
            self.roi_size = (16, 16)
            self._roi_align = torchvision.ops.RoIAlign(output_size=self.roi_size, spatial_scale=1 / 14,
                                                       sampling_ratio=2)
            self.cve_cap_mlp = nn.Sequential(
                nn.Linear(self.visual_encoder.embed_dim * 2, self.visual_encoder.embed_dim),
                nn.GELU(),
                nn.Linear(self.visual_encoder.embed_dim, self.visual_encoder.embed_dim))
            self.cve_tag_mlp = nn.Sequential(
                nn.Linear(self.visual_encoder.embed_dim * 2, self.visual_encoder.embed_dim),
                nn.GELU(),
                nn.Linear(self.visual_encoder.embed_dim, self.visual_encoder.embed_dim))

            # Controllable entity embedding module
            llm_dim = self.llm_model.lm_head.in_features
            self.cee_memory = nn.Parameter(torch.zeros(llm_dim))

            # Bi-directional embedding bridging module
            beb_dim = 128
            self.beb_c2l_mlp = nn.Linear(llm_dim, beb_dim)
            self.beb_l2c_mlp = nn.Linear(beb_dim, llm_dim)
            self.beb_o2l_mlp = nn.Linear(self.visual_encoder.embed_dim, beb_dim)
            self.beb_l2o_mlp = nn.Linear(beb_dim, self.visual_encoder.embed_dim)
            self.beb_cl2ol_ca = CrossAttnBlock(num_heads=8, hidden_dim=beb_dim, mlp_dim=beb_dim)
            self.beb_ol2cl_ca = CrossAttnBlock(num_heads=8, hidden_dim=beb_dim, mlp_dim=beb_dim)

            # Tag Head same as tag2text, query2label
            tag_bert_config = BertConfig.from_json_file(
                kwargs.get("tag_bert_config", "reem/models/tag_heads/tag_bert_config.json"))
            tag_bert_config.encoder_width = self.Qformer.config.encoder_width
            self.tag_head = BertModel(config=tag_bert_config, add_pooling_layer=False)
            del self.tag_head.embeddings
            for layer in self.tag_head.encoder.layer:
                del layer.attention
            tag_list = kwargs.get("tag_list", "reem/commom/tag_parser/ram_tag_list.txt")
            with open(tag_list, "r") as fr:
                self.tag_list = fr.readlines()
            self.tag_list = [tag.strip() for tag in self.tag_list]
            self.num_tags = len(self.tag_list)
            self.tag_labels = nn.Embedding(self.num_tags, tag_bert_config.hidden_size)
            self.tag_fc = nn.Linear(tag_bert_config.hidden_size, 1)
            self.tag_weight = 0.01
            self.tag_loss_function = AsymmetricLoss(gamma_neg=7, gamma_pos=0, clip=0.05)

            # Trainable parameters
            names = ["tag", "cve", "cee", "beb", "Qformer", "llm_proj"]
            self.finetune_llm = kwargs.get("finetune_llm", False)
            if self.finetune_llm:
                lora_config = LoraConfig(
                    r=64, lora_alpha=128, lora_dropout=0.0,
                    target_modules=["embed_tokens", "lm_head", "q_proj", "v_proj"]
                )
                self.llm_model = get_peft_model(self.llm_model, lora_config)
                self.llm_model.to(torch.float32)
                names.append("lora")

            trainable_params = 0
            all_params = 0
            for param_name, param in self.named_parameters():
                all_params += param.numel()
                param.requires_grad = False
                for name in names:
                    if name in param_name:
                        param.requires_grad = True
                        trainable_params += param.numel()
                        break
            print(
                f"[all params : {all_params}][trainable params : {trainable_params}][ratio : {trainable_params / all_params}]")

        @classmethod
        def from_config(cls, cfg):
            model = cls(**cfg)
            if cfg.pretrained is not None:
                model.load_checkpoint(url_or_filename=cfg.pretrained)
            return model

        def sequence2spatio(self, embeds):
            spatio_image_embeds = embeds[:, 1:]
            cls_embeds = embeds[:, 0][:, None]
            b, hw, c = spatio_image_embeds.shape
            h, w = int(math.sqrt(hw)), int(math.sqrt(hw))
            spatio_image_embeds = spatio_image_embeds.reshape(b, h, w, c).permute(0, 3, 1, 2)
            return spatio_image_embeds, cls_embeds

        def spatio2sequence(self, spatio_image_embeds, cls_embeds):
            b, c, h, w = spatio_image_embeds.shape
            spatio_image_embeds = spatio_image_embeds.permute(0, 2, 3, 1).reshape(b, -1, c)
            return torch.cat([cls_embeds, spatio_image_embeds], 1)

        def roi_align(self, image_embeds, samples):
            # prepare spatio image embeddings
            spatio_image_embeds, rois_cls_embeds = self.sequence2spatio(image_embeds)

            # mask the padding bbox of the recognition task to save memory
            bboxes = samples["bboxes"]
            ids = samples["batch_idx"].to(torch.int64)
            rois = torch.cat([ids[:, None], bboxes], -1)

            # instance level feature encoder
            rois_embeds = self._roi_align(spatio_image_embeds, rois)
            rois_cls_embeds = rois_cls_embeds[ids]

            # back to sequence
            rois_embeds = self.spatio2sequence(rois_embeds, rois_cls_embeds)
            return rois_embeds

        def prepare_control_tags(self, samples, tag_logits, drop_ratio=0.5, tag_thr=0.9):
            control_tags = []

            if self.training:
                tag_idxs = samples["tags"]
                tags = [[self.tag_list[tag_idx] for tag_idx in torch.nonzero(tag_idxs[bz_idx][self.num_tags:])]
                        for bz_idx in range(len(tag_idxs))]

                for tag in tags:
                    l = len(tag)
                    if l == 0:
                        control_tag = "()"
                    else:
                        sl = torch.from_numpy(np.random.uniform(0, 1, l) > drop_ratio)
                        control_tag = [tag[tag_idx] for tag_idx in torch.nonzero(sl)]
                        random.shuffle(control_tag)
                        control_tag = "(" + ",".join(control_tag) + ")"
                    control_tags.append(control_tag)
            else:
                tag_scores = tag_logits.sigmoid()
                tag_idxs = (tag_scores > tag_thr).to(torch.long)
                tags = [[self.tag_list[tag_idx] for tag_idx in torch.nonzero(tag_idxs[bz_idx][self.num_tags:])]
                        for bz_idx in range(len(tag_idxs))]

                for tag in tags:
                    control_tag = "(" + ",".join(tag) + ")"
                    control_tags.append(control_tag)

            return control_tags

        def cve_forward(self, samples, embeds):
            bz = len(samples["image"])
            image_embeds = embeds[:bz]
            region_embeds = embeds[bz:]

            # local-roi fusion
            rois_embeds = self.roi_align(image_embeds, samples)
            object_embeds = torch.cat([rois_embeds, region_embeds], -1)
            cap_embeds = self.cve_cap_mlp(object_embeds)
            tag_embeds = self.cve_tag_mlp(object_embeds)

            return cap_embeds, tag_embeds

        def tag_forward(self, samples, tag_embeds):
            bs = len(tag_embeds)
            object_atts = torch.ones(tag_embeds.size()[:-1], dtype=torch.long).to(
                tag_embeds.device
            )
            label_embed = self.tag_labels.weight.unsqueeze(0).repeat(bs, 1, 1)

            tagging_embed = self.tag_head(
                encoder_embeds=label_embed,
                encoder_hidden_states=tag_embeds,
                encoder_attention_mask=object_atts,
                return_dict=False,
                mode='tagging',
            )
            tag_logits = self.tag_fc(tagging_embed[0]).squeeze(-1)
            return tag_logits

        def cap_forward(self, samples, cap_embeds, control_tags, crl_embeds):
            if self.training:
                device = cap_embeds.device
                cap_atts = torch.ones(cap_embeds.size()[:-1], dtype=torch.long).to(device)
                query_tokens = self.query_tokens.expand(cap_embeds.shape[0], -1, -1)

                if self.qformer_text_input:
                    text_Qformer = self.tokenizer(
                        control_tags,
                        padding='longest',
                        truncation=True,
                        max_length=self.max_txt_len,
                        return_tensors="pt",
                    ).to(cap_embeds.device)
                    query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(device)
                    Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask], dim=1)

                    query_output = self.Qformer.bert(
                        text_Qformer.input_ids,
                        attention_mask=Qformer_atts,
                        query_embeds=query_tokens,
                        encoder_hidden_states=cap_embeds,
                        encoder_attention_mask=cap_atts,
                        return_dict=True,
                    )
                else:
                    query_output = self.Qformer.bert(
                        query_embeds=query_tokens,
                        encoder_hidden_states=cap_embeds,
                        encoder_attention_mask=cap_atts,
                        return_dict=True,
                    )

                inputs_llm = self.llm_proj(query_output.last_hidden_state[:, :query_tokens.size(1), :])
                atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(device)

                self.llm_tokenizer.padding_side = "right"
                self.llm_tokenizer.truncation_side = 'left'
                text_input_tokens = self.llm_tokenizer(
                    control_tags,
                    return_tensors="pt",
                    padding="longest",
                    truncation=True,
                    max_length=self.max_txt_len,
                ).to(device)

                self.llm_tokenizer.truncation_side = 'right'
                text_output_tokens = self.llm_tokenizer(
                    [t + self.llm_tokenizer.eos_token for t in samples['caps']],
                    return_tensors="pt",
                    padding="longest",
                    truncation=True,
                    max_length=self.max_output_txt_len,
                ).to(device)

                llm_tokens, input_part_targets_len = self.concat_text_input_output(
                    text_input_tokens.input_ids,
                    text_input_tokens.attention_mask,
                    text_output_tokens.input_ids,
                    text_output_tokens.attention_mask,
                )

                # do not apply loss to the padding
                targets = llm_tokens['input_ids'].masked_fill(
                    llm_tokens['input_ids'] == self.llm_tokenizer.pad_token_id, -100
                )

                # do not apply loss to the text input (i.e., instruction)
                for i, l in enumerate(input_part_targets_len):
                    targets[i][:l] = -100

                # do not apply loss to the query tokens
                empty_targets = (
                    torch.ones(atts_llm.size(), dtype=torch.long).to(device).fill_(-100)
                )
                targets = torch.cat([empty_targets, targets], dim=1)

                with self.maybe_autocast(torch.float16):
                    inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens['input_ids'])
                    for idx, part in enumerate(input_part_targets_len):
                        inputs_embeds[idx, :part] = crl_embeds[idx, :part]
                    inputs_embeds = torch.cat([inputs_llm, inputs_embeds], dim=1)
                    attention_mask = torch.cat([atts_llm, llm_tokens['attention_mask']], dim=1)
                    outputs = self.llm_model(
                        inputs_embeds=inputs_embeds,
                        attention_mask=attention_mask,
                        return_dict=True,
                        labels=targets,
                    )
                loss = outputs.loss

                return loss
            else:
                device = cap_embeds.device
                self.llm_tokenizer.padding_side = "left"
                bs = cap_embeds.size(0)
                cap_atts = torch.ones(cap_embeds.size()[:-1], dtype=torch.long).to(device)
                query_tokens = self.query_tokens.expand(bs, -1, -1)
                if self.qformer_text_input:
                    text_Qformer = self.tokenizer(
                        control_tags,
                        padding='longest',
                        truncation=True,
                        max_length=self.max_txt_len,
                        return_tensors="pt",
                    ).to(cap_embeds.device)
                    query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(device)
                    Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask], dim=1)

                    query_output = self.Qformer.bert(
                        text_Qformer.input_ids,
                        attention_mask=Qformer_atts,
                        query_embeds=query_tokens,
                        encoder_hidden_states=cap_embeds,
                        encoder_attention_mask=cap_atts,
                        return_dict=True,
                    )
                else:
                    query_output = self.Qformer.bert(
                        query_embeds=query_tokens,
                        encoder_hidden_states=cap_embeds,
                        encoder_attention_mask=cap_atts,
                        return_dict=True,
                    )

                inputs_llm = self.llm_proj(query_output.last_hidden_state[:, :query_tokens.size(1), :])
                atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(device)

                llm_tokens = self.llm_tokenizer(
                    control_tags,
                    padding="longest",
                    return_tensors="pt"
                ).to(device)

                with self.maybe_autocast(torch.float16):
                    # inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens.input_ids)
                    inputs_embeds = crl_embeds
                    inputs_embeds = torch.cat([inputs_llm, inputs_embeds], dim=1)
                    attention_mask = torch.cat([atts_llm, llm_tokens.attention_mask], dim=1)

                    llm_kwargs = copy.deepcopy(self.kwargs)
                    llm_kwargs_keys = ["do_sample", "top_p", "temperature", "num_beams", "max_length",
                                       "min_length", "repetition_penalty", "length_penalty", "num_return_sequences"]
                    for key in self.kwargs.keys():
                        if key not in llm_kwargs_keys:
                            llm_kwargs.pop(key)

                    outputs = self.llm_model.generate(
                        inputs_embeds=inputs_embeds,
                        attention_mask=attention_mask,
                        output_scores=True,
                        return_dict_in_generate=True,
                        **llm_kwargs
                    )

                sequences = outputs.sequences
                scores = outputs.sequences_scores
                scores = torch.exp(scores).cpu().numpy().tolist()

                sequences[sequences == 0] = 2  # convert output id 0 to 2 (eos_token_id)
                captions = self.llm_tokenizer.batch_decode(sequences, skip_special_tokens=True)
                captions = [caption.strip() for caption in captions]

                return scores, captions

        def cee_forward(self, control_tags, embeds):
            if self.training:
                crl_tokens = self.llm_tokenizer(
                    control_tags,
                    return_tensors="pt",
                    padding="longest",
                    truncation=True,
                    max_length=self.max_txt_len,
                ).to(embeds.device)
            else:
                crl_tokens = self.llm_tokenizer(
                    control_tags,
                    padding="longest",
                    return_tensors="pt"
                ).to(embeds.device)
            crl_embeds = self.llm_model.get_input_embeddings()(crl_tokens['input_ids'])
            return crl_embeds + self.cee_memory, crl_tokens

        def beb_forward(self, o_embeds, c_embeds):
            ol_embeds = self.beb_o2l_mlp(o_embeds)
            cl_embeds = self.beb_c2l_mlp(c_embeds)
            ol_embeds, attn_c2o = self.beb_cl2ol_ca(ol_embeds, cl_embeds)
            cl_embeds, attn_o2c = self.beb_ol2cl_ca(cl_embeds, ol_embeds)
            o_embeds = o_embeds + self.beb_l2o_mlp(ol_embeds)
            c_embeds = c_embeds + self.beb_l2c_mlp(cl_embeds)
            return o_embeds, c_embeds, (attn_c2o, attn_o2c)

        def forward(self, samples):
            image = torch.cat([samples["image"], samples["region_images"]], 0)

            with self.maybe_autocast():
                embeds = self.ln_vision(self.visual_encoder(image))

            cap_embeds, tag_embeds = self.cve_forward(samples, embeds)
            tag_logits = self.tag_forward(samples, tag_embeds)
            control_tags = self.prepare_control_tags(samples, tag_logits)
            crl_embeds, _ = self.cee_forward(control_tags, embeds)
            cap_embeds, crl_embeds, _ = self.beb_forward(cap_embeds, crl_embeds)

            tags = samples["tags"].to(torch.long)[:, self.num_tags:]
            loss_tag = self.tag_loss_function(tag_logits, tags) * self.tag_weight

            loss_llm = self.cap_forward(samples, cap_embeds, control_tags, crl_embeds)

            return {"loss": loss_llm + loss_tag, "loss_llm": loss_llm.detach(), "loss_tag": loss_tag.detach()}

        def predict_answers_memory_efficient(self, samples, **kwargs):
            split_size = int(self.kwargs.get("split_size", int(1e7)))
            l_samples = len(samples["ids"])
            chunk_samples = []
            if l_samples <= split_size:
                chunk_samples = [samples]
            else:
                idxs = torch.LongTensor(range(l_samples))
                chunk_idxs = torch.split(idxs, split_size_or_sections=split_size)
                for chunk_idx in chunk_idxs:
                    chunk_sample = dict()
                    for key, value in samples.items():
                        if len(value) != l_samples:
                            chunk_sample[key] = value
                        elif isinstance(value, list):
                            chunk_sample[key] = [value[idx] for idx in chunk_idx]
                        else:
                            chunk_sample[key] = value[chunk_idx]
                    chunk_samples.append(chunk_sample)

            output = []
            for chunk_sample in chunk_samples:
                result = self.predict_answers(chunk_sample, **kwargs)
                output.extend(result)

            return output

        def predict_answers(self, samples, **kwargs):
            split_size = int(self.kwargs.get("split_size", int(1e7)))
            l_samples = len(samples["ids"])
            if l_samples > split_size:
                return self.predict_answers_memory_efficient(samples, **kwargs)

            image = torch.cat([samples["image"], samples["region_images"]], 0)

            with self.maybe_autocast():
                embeds = self.ln_vision(self.visual_encoder(image))

            cap_embeds, tag_embeds = self.cve_forward(samples, embeds)
            tag_logits = self.tag_forward(samples, tag_embeds)
            control_tags = self.prepare_control_tags(samples, tag_logits)
            crl_embeds, _ = self.cee_forward(control_tags, embeds)
            cap_embeds, crl_embeds, _ = self.beb_forward(cap_embeds, crl_embeds)

            scores, captions = self.cap_forward(samples, cap_embeds, control_tags, crl_embeds)

            if self.kwargs.get("apply_lemmatizer", False):
                captions = self._lemmatize(captions)

            # format the predictions
            name = "reem_vicuna"
            output = []
            for id, caption, score in zip(samples["ids"], captions, scores):
                output.append(
                    {"id": id, "caption": {name: [{"caption": caption, "score": score}]}}
                )

            # # format the predictions
            # name = "reem_vicuna"
            # output = []
            # for id, stag, tag, caption, score in zip(samples["ids"], stags, tags, captions, scores):
            #     output.append(
            #         {"id": id, "caption": caption, "score": score, "tag": {"subj": stag, "full": tag}, "annotator": name}
            #     )

            return output





