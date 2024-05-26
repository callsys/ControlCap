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


@registry.register_model("controlcap_t5")
class ControlCapT5(Blip2T5):
    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs
        base_kwargs = copy.deepcopy(kwargs)
        base_kwargs_keys = ["vit_model", "img_size", "drop_path_rate", "use_grad_checkpoint", "vit_precision",
                            "freeze_vit", "num_query_token", "t5_model", "prompt", "max_txt_len", "apply_lemmatizer"]
        for key in kwargs.keys():
            if key not in base_kwargs_keys:
                base_kwargs.pop(key)
        super().__init__(*args, **base_kwargs)

        # contextual visual embedding module
        input_image_size = self.visual_encoder.image_size
        patch_size = self.visual_encoder.patch_embed.patch_size[0]
        self._roi_align = torchvision.ops.RoIAlign(output_size=input_image_size//patch_size, spatial_scale=1 / patch_size,
                                                   sampling_ratio=2)

        self.cvem_mlp = nn.Sequential(
            nn.Linear(self.visual_encoder.embed_dim * 2, self.visual_encoder.embed_dim),
            nn.ReLU(),
            nn.Linear(self.visual_encoder.embed_dim, self.visual_encoder.embed_dim))

        # control embedding module
        self.cem_memory = nn.Parameter(torch.zeros(self.t5_model.model_dim))

        # embedding bridging module
        ebm_dim = 128
        ebm_num_heads = 8
        self.ebm_c2l_mlp = nn.Linear(self.t5_model.model_dim, ebm_dim)
        self.ebm_l2c_mlp = nn.Linear(ebm_dim, self.t5_model.model_dim)
        self.ebm_v2l_mlp = nn.Linear(self.visual_encoder.embed_dim, ebm_dim)
        self.ebm_l2v_mlp = nn.Linear(ebm_dim, self.visual_encoder.embed_dim)
        self.ebm_cl2vl_ca = CrossAttnBlock(num_heads=ebm_num_heads, hidden_dim=ebm_dim, mlp_dim=ebm_dim)
        self.ebm_vl2cl_ca = CrossAttnBlock(num_heads=ebm_num_heads, hidden_dim=ebm_dim, mlp_dim=ebm_dim)

        # region tagging head
        embed_dim = 256
        self.cvem_tag_mlp = nn.Sequential(
            nn.Linear(self.visual_encoder.embed_dim * 2, self.visual_encoder.embed_dim),
            nn.ReLU(),
            nn.Linear(self.visual_encoder.embed_dim, embed_dim))
        tag_bert_config = BertConfig.from_json_file(
            kwargs.get("tag_bert_config", "controlcap/models/tagging_heads/tag_bert_config.json"))
        # tag_bert_config.encoder_width = self.Qformer.config.encoder_width
        tag_bert_config.encoder_width = embed_dim
        tag_bert_config.hidden_size = embed_dim
        tag_bert_config.intermediate_size = embed_dim * 4
        self.tag_head = BertModel(config=tag_bert_config, add_pooling_layer=False)
        del self.tag_head.embeddings
        for layer in self.tag_head.encoder.layer:
            del layer.attention
        tag_list = kwargs.get("tag_list", "controlcap/common/tagging/ram_tag_list.txt")
        with open(tag_list, "r") as fr:
            self.tag_list = fr.readlines()
        self.tag_list = [tag.strip() for tag in self.tag_list]
        self.num_tags = len(self.tag_list)
        self.tag_labels = nn.Embedding(self.num_tags * 2, tag_bert_config.hidden_size)
        self.tag_fc = nn.Linear(tag_bert_config.hidden_size, 1)
        self.tag_weight = 0.01
        self.tag_loss_function = AsymmetricLoss(gamma_neg=7, gamma_pos=0, clip=0.05)

        # Trainable parameters
        names = ["cvem", "cem", "tag", "ebm", "Qformer", "t5_proj"]
        self.finetune_llm = kwargs.get("finetune_llm", False)
        if self.finetune_llm:
            lora_config = LoraConfig(
                r=64, lora_alpha=128, lora_dropout=0.0,
                target_modules=["embed_tokens", "lm_head", "q", "v"]
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

    def roi_align(self, image_embeds, samples):
        # prepare cls image embeds and spatio image embeddings
        spatio_image_embeds = image_embeds[:, 1:]
        cls_image_embeds = image_embeds[:, 0][:, None]
        b, hw, c = spatio_image_embeds.shape
        h, w = int(math.sqrt(hw)), int(math.sqrt(hw))
        spatio_image_embeds = spatio_image_embeds.reshape(b, h, w, c).permute(0, 3, 1, 2)

        # extract roi features
        bboxes = samples["bboxes"]
        ids = samples["batch_idx"].to(torch.int64)
        rois = torch.cat([ids[:, None], bboxes], -1)
        spatio_rois_embeds = self._roi_align(spatio_image_embeds, rois)
        cls_image_embeds = cls_image_embeds[ids]

        # back to sequence
        bv = spatio_rois_embeds.shape[0]
        spatio_rois_embeds = spatio_rois_embeds.permute(0, 2, 3, 1).reshape(bv, -1, c)
        rois_embeds = torch.cat([cls_image_embeds, spatio_rois_embeds], 1)
        return rois_embeds

    def cvem_forward(self, samples, embeds):
        bz = len(samples["image"])
        image_embeds = embeds[:bz]
        region_embeds = embeds[bz:]
        rois_embeds = self.roi_align(image_embeds, samples)
        visual_embeds = torch.cat([rois_embeds, region_embeds], -1)
        visual_tag_embeds = self.cvem_tag_mlp(visual_embeds)
        visual_embeds = self.cvem_mlp(visual_embeds)
        return visual_embeds, visual_tag_embeds

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

    def prepare_control_words(self, samples, tag_logits):
        control_words = []
        full_drop_ratio = self.kwargs.get("full_drop_ratio", 0.5)
        drop_ratio = self.kwargs.get("drop_ratio", 0.5)
        tag_thr = self.kwargs.get("tag_thr", 0.7)

        if self.training:
            for bz_idx, cap in enumerate(samples["caps"]):
                try:
                    s2 = TextBlob(cap).tags
                    tokens = [el[0] for el in s2]
                    infowords = [name for name, value in s2 if ("NN" in value) or ("JJ" in value)]
                    nouns = [name for name, value in s2 if ("NN" in value)]
                    if len(infowords) > 0:
                        words = []
                        for word in infowords:
                            st_idx = tokens.index(word)
                            ed_idx = st_idx + 1
                            while (ed_idx < len(tokens)) and (tokens[ed_idx] in nouns):
                                ed_idx = ed_idx + 1
                            word = " ".join(tokens[st_idx:ed_idx])
                            words.append(word)
                    else:
                        words = [""]
                except:
                    words = [""]
                tag_idxs = samples["tags"]
                stags = [self.tag_list[tag_idx] for tag_idx in torch.nonzero(tag_idxs[bz_idx][:self.num_tags])]
                otags = [self.tag_list[tag_idx] for tag_idx in torch.nonzero(tag_idxs[bz_idx][self.num_tags:])]
                tags = stags + otags + words
                tags = list(set(tags))
                l = len(tags)
                if np.random.uniform(0, 1) < full_drop_ratio:
                    control_word = ""
                else:
                    if l == 0:
                        control_word = ""
                    else:
                        sl = torch.from_numpy(np.random.uniform(0, 1, l) > drop_ratio)
                        control_word = [tags[tag_idx] for tag_idx in torch.nonzero(sl)]
                        random.shuffle(control_word)
                        control_word = ",".join(control_word)
                control_words.append(control_word + "|")
            return control_words
        else:
            tag_scores = tag_logits.sigmoid()
            tag_idxs = (tag_scores > tag_thr).to(torch.long)
            stags = [[self.tag_list[tag_idx] for tag_idx in torch.nonzero(tag_idxs[bz_idx][:self.num_tags])]
                     for bz_idx in range(len(tag_idxs))]
            otags = [[self.tag_list[tag_idx] for tag_idx in torch.nonzero(tag_idxs[bz_idx][self.num_tags:])]
                     for bz_idx in range(len(tag_idxs))]
            tags = [stag + otag for stag, otag in zip(stags, otags)]

            first_word_control = self.kwargs.get("first_word_control", False)
            if first_word_control:
                first_words = []
                for bz_idx, cap in enumerate(samples["caps"]):
                    try:
                        s2 = TextBlob(cap).tags
                        tokens = [el[0] for el in s2]
                        infowords = [name for name, value in s2 if ("NN" in value) or ("JJ" in value)]
                        nouns = [name for name, value in s2 if ("NN" in value)]
                        if len(infowords) > 0:
                            words = []
                            for word in infowords:
                                st_idx = tokens.index(word)
                                ed_idx = st_idx + 1
                                while (ed_idx < len(tokens)) and (tokens[ed_idx] in nouns):
                                    ed_idx = ed_idx + 1
                                word = " ".join(tokens[st_idx:ed_idx])
                                words.append(word)
                        else:
                            words = []
                    except:
                        words = []
                    if len(words) > 0:
                        first_word = [words[0]]
                    else:
                        first_word = []
                    first_words.append(first_word)
                tags = [fword + tag for fword, tag in zip(first_words, tags)]

            for control_tag in tags:
                control_tag = list(set(control_tag))
                # control_tag.sort()
                control_word = ",".join(control_tag)
                control_words.append(control_word + "|")

            return control_words, stags, otags

    def cem_forward(self, tags, embeds):
        control_tokens = self.t5_tokenizer(
            tags,
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(embeds.device)
        control_embeds = self.t5_model.encoder.embed_tokens(control_tokens.input_ids) + self.cem_memory
        return control_embeds, control_tokens

    def ebm_forward(self, v_embeds, c_embeds):
        vl_embeds = self.ebm_v2l_mlp(v_embeds)
        cl_embeds = self.ebm_c2l_mlp(c_embeds)
        vl_embeds, _ = self.ebm_cl2vl_ca(vl_embeds, cl_embeds)
        cl_embeds, _ = self.ebm_vl2cl_ca(cl_embeds, vl_embeds)
        v_embeds = v_embeds + self.ebm_l2v_mlp(vl_embeds)
        c_embeds = c_embeds + self.ebm_l2c_mlp(cl_embeds)
        return v_embeds, c_embeds

    def forward(self, samples):
        image = torch.cat([samples["image"], samples["region_images"]], 0)

        with self.maybe_autocast(dtype=torch.float16):
            embeds = self.ln_vision(self.visual_encoder(image))
            visual_embeds, visual_tag_embeds = self.cvem_forward(samples, embeds)
            tag_logits = self.tag_forward(samples, visual_tag_embeds)
            control_words = self.prepare_control_words(samples, tag_logits)
            control_embeds, control_tokens = self.cem_forward(control_words, visual_embeds)
            visual_embeds, control_embeds = self.ebm_forward(visual_embeds, control_embeds)

        with self.maybe_autocast(dtype=torch.bfloat16):
            object_atts = torch.ones(visual_embeds.size()[:-1], dtype=torch.long).to(
                image.device
            )
            query_tokens = self.query_tokens.expand(visual_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=visual_embeds,
                encoder_attention_mask=object_atts,
                return_dict=True,
            )
            inputs_t5 = self.t5_proj(query_output.last_hidden_state)
            atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(image.device)
            encoder_atts = torch.cat([atts_t5, control_tokens.attention_mask], dim=1)
            inputs_embeds = torch.cat([inputs_t5, control_embeds], dim=1)

            tags = samples["tags"].to(torch.long)
            loss_tag = self.tag_loss_function(tag_logits, tags) * self.tag_weight

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

            return {"loss": loss_llm + loss_tag, "loss_llm": loss_llm.detach(), "loss_tag": loss_tag.detach()}

    def predict_answers(
            self,
            samples,
            *args,
            **kwargs,
    ):
        image = torch.cat([samples["image"], samples["region_images"]], 0)

        with self.maybe_autocast(dtype=torch.float16):
            embeds = self.ln_vision(self.visual_encoder(image))
            visual_embeds, visual_tag_embeds = self.cvem_forward(samples, embeds)
            tag_logits = self.tag_forward(samples, visual_tag_embeds)
            control_words, stags, otags = self.prepare_control_words(samples, tag_logits)
            control_embeds, control_tokens = self.cem_forward(control_words, visual_embeds)
            visual_embeds, control_embeds = self.ebm_forward(visual_embeds, control_embeds)

        with self.maybe_autocast(dtype=torch.bfloat16):
            object_atts = torch.ones(visual_embeds.size()[:-1], dtype=torch.long).to(
                image.device
            )
            query_tokens = self.query_tokens.expand(visual_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=visual_embeds,
                encoder_attention_mask=object_atts,
                return_dict=True,
            )
            inputs_t5 = self.t5_proj(query_output.last_hidden_state)
            atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(image.device)
            encoder_atts = torch.cat([atts_t5, control_tokens.attention_mask], dim=1)
            inputs_embeds = torch.cat([inputs_t5, control_embeds], dim=1)

            llm_kwargs = {
                "do_sample": False,
                "num_beams": self.kwargs.get("num_beams", 5),
                "max_new_tokens": self.kwargs.get("max_new_tokens", 10),
                "min_length": self.kwargs.get("min_length", 1),
                "length_penalty": self.kwargs.get("length_penalty", -1),
                "repetition_penalty": self.kwargs.get("repetition_penalty", None),
                "num_return_sequences": self.kwargs.get("num_return_sequences", 1),
                "top_p": self.kwargs.get("top_p", None),
                "temperature": self.kwargs.get("temperature", None)}
            keys_to_pop = [key for key, value in llm_kwargs.items() if value is None]
            for key in keys_to_pop:
                llm_kwargs.pop(key)

            outputs = self.t5_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
                output_scores=True,
                return_dict_in_generate=True,
                **llm_kwargs
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
        for id, caption, score, stag, otag in zip(samples["ids"], captions, scores, stags, otags):
            output.append(
                {"id": id, "caption": caption, "score": score, "tag_set1": stag, "tag_set2": otag}
            )

        return output

    @classmethod
    def from_config(cls, cfg):
        model = cls(**cfg)
        if cfg.pretrained is not None:
            model.load_checkpoint(url_or_filename=cfg.pretrained)
        return model





