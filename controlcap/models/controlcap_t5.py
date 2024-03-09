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
from controlcap.models.tag_heads.bert import BertConfig, BertModel
from controlcap.models.tag_heads.asymmetric_loss import AsymmetricLoss


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
        base_kwargs_keys = ["vit_model", "img_size", "drop_path_rate", "use_grad_checkpoint", "vit_precision", "freeze_vit",
                            "num_query_token", "t5_model", "prompt", "max_txt_len", "apply_lemmatizer"]
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


        # Controllable entity embedding module
        self.cee_memory = nn.Parameter(torch.zeros(self.t5_model.model_dim))


        # Bi-directional embedding bridging module
        beb_dim = 128
        self.beb_c2l_mlp = nn.Linear(self.t5_model.model_dim, beb_dim)
        self.beb_l2c_mlp = nn.Linear(beb_dim, self.t5_model.model_dim)
        self.beb_o2l_mlp = nn.Linear(self.visual_encoder.embed_dim, beb_dim)
        self.beb_l2o_mlp = nn.Linear(beb_dim, self.visual_encoder.embed_dim)
        self.beb_cl2ol_ca = CrossAttnBlock(num_heads=8, hidden_dim=beb_dim, mlp_dim=beb_dim)
        self.beb_ol2cl_ca = CrossAttnBlock(num_heads=8, hidden_dim=beb_dim, mlp_dim=beb_dim)


        # Tag Head same as tag2text, query2label
        tag_bert_config = BertConfig.from_json_file(
            kwargs.get("tag_bert_config", "controlcap/models/tag_heads/tag_bert_config.json"))
        tag_bert_config.encoder_width = self.Qformer.config.encoder_width
        self.tag_head = BertModel(config=tag_bert_config, add_pooling_layer=False)
        del self.tag_head.embeddings
        for layer in self.tag_head.encoder.layer:
            del layer.attention
        tag_list = kwargs.get("tag_list", "controlcap/commom/tag_parser/ram_tag_list.txt")
        with open(tag_list, "r") as fr:
            self.tag_list = fr.readlines()
        self.tag_list = [tag.strip() for tag in self.tag_list]
        self.num_tags = len(self.tag_list)
        self.tag_labels = nn.Embedding(self.num_tags * 2, tag_bert_config.hidden_size)
        self.tag_fc = nn.Linear(tag_bert_config.hidden_size, 1)
        self.tag_weight = 0.01
        self.tag_loss_function = AsymmetricLoss(gamma_neg=7, gamma_pos=0, clip=0.05)


        # Trainable parameters
        names = ["cve", "cee", "tag", "beb", "Qformer", "t5_proj"]
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

    def prepare_control_tags(self, samples, tag_logits, drop_ratio=0.5, tag_thr=(0.7, 0.7)):
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
                    if len(words)>0:
                        first_word = [words[0]]
                    else:
                        first_word = []
                    first_words.append(first_word)

            tag_thr = self.kwargs.get("tag_thr", tag_thr)
            if not isinstance(tag_thr, tuple):
                tag_thr = (tag_thr, tag_thr)
            tag_scores = tag_logits.sigmoid()
            tag_thr_mat = torch.zeros_like(tag_scores).to(tag_scores)
            tag_thr_mat[:, :self.num_tags] = tag_thr[0]
            tag_thr_mat[:, self.num_tags:] = tag_thr[1]
            tag_idxs = (tag_scores > tag_thr_mat).to(torch.long)
            stags = [[self.tag_list[tag_idx] for tag_idx in torch.nonzero(tag_idxs[bz_idx][:self.num_tags])]
                     for bz_idx in range(len(tag_idxs))]
            otags = [[self.tag_list[tag_idx] for tag_idx in torch.nonzero(tag_idxs[bz_idx][self.num_tags:])]
                     for bz_idx in range(len(tag_idxs))]
            tags = [stag + otag for stag, otag in zip(stags, otags)]

            if first_word_control:
                tags = [fword + tag for fword, tag in zip(first_words, tags)]

            for tag in tags:
                tag = set(tag)
                if len(tag)==0:
                    control_tag = "|"
                else:
                    control_tag = ",".join(set(tag)) + "|"
                control_tags.append(control_tag)

        if self.training:
            return control_tags
        else:
            return control_tags, stags, otags

    def cee_forward(self, tags, embeds):
        crl_tokens = self.t5_tokenizer(
            tags,
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(embeds.device)
        crl_embeds = self.t5_model.encoder.embed_tokens(crl_tokens.input_ids)
        return crl_embeds + self.cee_memory, crl_tokens

    def beb_forward(self, o_embeds, c_embeds):
        ol_embeds = self.beb_o2l_mlp(o_embeds)
        cl_embeds = self.beb_c2l_mlp(c_embeds)
        ol_embeds, attn_c2o = self.beb_cl2ol_ca(ol_embeds, cl_embeds)
        cl_embeds, attn_o2c = self.beb_ol2cl_ca(cl_embeds, ol_embeds)
        o_embeds = o_embeds + self.beb_l2o_mlp(ol_embeds)
        c_embeds = c_embeds + self.beb_l2c_mlp(cl_embeds)
        return o_embeds, c_embeds, (attn_c2o, attn_o2c)

    def loss(self, tag_logits, inputs_embeds, encoder_atts, samples):
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

    def forward(self, samples):
        image = torch.cat([samples["image"], samples["region_images"]], 0)

        with self.maybe_autocast(dtype=torch.float16):
            embeds = self.ln_vision(self.visual_encoder(image))
            embeds, tag_embeds = self.cve_forward(samples, embeds)
            tag_logits = self.tag_forward(samples, tag_embeds)
            control_tags = self.prepare_control_tags(samples, tag_logits)
            crl_embeds, crl_tokens = self.cee_forward(control_tags, embeds)
            cap_embeds, crl_embeds, _ = self.beb_forward(embeds, crl_embeds)

        with self.maybe_autocast(dtype=torch.bfloat16):
            object_atts = torch.ones(cap_embeds.size()[:-1], dtype=torch.long).to(
                image.device
            )
            query_tokens = self.query_tokens.expand(cap_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=cap_embeds,
                encoder_attention_mask=object_atts,
                return_dict=True,
            )
            inputs_t5 = self.t5_proj(query_output.last_hidden_state)
            atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(image.device)
            encoder_atts = torch.cat([atts_t5, crl_tokens.attention_mask], dim=1)
            inputs_embeds = torch.cat([inputs_t5, crl_embeds], dim=1)

            return self.loss(tag_logits, inputs_embeds, encoder_atts, samples)

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
        image = torch.cat([samples["image"], samples["region_images"]], 0)

        with self.maybe_autocast():
            embeds = self.ln_vision(self.visual_encoder(image))
            embeds, tag_embeds = self.cve_forward(samples, embeds)
            tag_logits = self.tag_forward(samples, tag_embeds)
            control_tags, stags, otags = self.prepare_control_tags(samples, tag_logits)
            crl_embeds, crl_tokens = self.cee_forward(control_tags, embeds)
            cap_embeds, crl_embeds, _ = self.beb_forward(embeds, crl_embeds)

        with self.maybe_autocast(dtype=torch.bfloat16):
            object_atts = torch.ones(cap_embeds.size()[:-1], dtype=torch.long).to(
                image.device
            )
            query_tokens = self.query_tokens.expand(cap_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=cap_embeds,
                encoder_attention_mask=object_atts,
                return_dict=True,
            )
            inputs_t5 = self.t5_proj(query_output.last_hidden_state)
            atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(image.device)
            encoder_atts = torch.cat([atts_t5, crl_tokens.attention_mask], dim=1)
            inputs_embeds = torch.cat([inputs_t5, crl_embeds], dim=1)

            llm_kwargs = {"top_p": 0.9,
                          "repetition_penalty": 1.5,
                          "length_penalty": 0,
                          "temperature": 1}

            outputs = self.t5_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
                do_sample=False,
                num_beams=3,
                max_new_tokens=20,
                min_length=min_len,
                # length_penalty=length_penalty,
                num_return_sequences=1,
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

        name = "controlcap_t5"
        output = []
        for id, caption, score, stag, otag in zip(samples["ids"], captions, scores, stags, otags):
            output.append(
                {"id": id, "caption": {name: [{"caption": caption, "score": score}]},
                 # "extra_info": {"pred_subj_tags": [self.tag_list.index(el) for el in stag], "pred_obj_tags": [self.tag_list.index(el) for el in otag]}}
                 "extra_info": {"pred_subj_tags": stag,
                                "pred_obj_tags": otag}}
            )

        return output

    @classmethod
    def from_config(cls, cfg):
        cfg["max_txt_len"] = cfg.get("max_txt_len", 128)
        model = cls(**cfg)
        if cfg.pretrained is not None:
            model.load_checkpoint(url_or_filename=cfg.pretrained)
        return model





