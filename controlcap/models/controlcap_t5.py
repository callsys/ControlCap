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
                            "freeze_vit",
                            "num_query_token", "t5_model", "prompt", "max_txt_len", "apply_lemmatizer"]
        for key in kwargs.keys():
            if key not in base_kwargs_keys:
                base_kwargs.pop(key)
        super().__init__(*args, **base_kwargs)

        # Contextual object embedding module
        self.roi_size = (16, 16)
        self._roi_align = torchvision.ops.RoIAlign(output_size=self.roi_size, spatial_scale=1 / 14,
                                                   sampling_ratio=2)

        self.coe_cap_mlp = nn.Sequential(nn.Linear(2816, 1408), nn.ReLU(), nn.Linear(1408, 1408))
        self.coe_tag_mlp = nn.Sequential(nn.Linear(2816, 1408), nn.ReLU(), nn.Linear(1408, 1408))

        # Controllable entity embedding module
        self.cee_memory = nn.Parameter(torch.zeros(2048))

        # Tag Head same as tag2text, query2label
        q2l_config = BertConfig.from_json_file(f'controlcap/models/tag_heads/tag_bert_config.json')
        q2l_config.encoder_width = self.Qformer.config.encoder_width
        self.tag_head = BertModel(config=q2l_config, add_pooling_layer=False)
        del self.tag_head.embeddings
        for layer in self.tag_head.encoder.layer:
            del layer.attention
        tag_file = "controlcap/common/tagging/ram_tag_list.txt"
        with open(tag_file, "r") as fr:
            tag_list = fr.readlines()
        self.tag_list = [tag.strip() for tag in tag_list]
        self.num_tags = len(self.tag_list)
        self.tag_labels = nn.Embedding(self.num_tags * 2, q2l_config.hidden_size)
        self.tag_fc = nn.Linear(q2l_config.hidden_size, 1)
        self.tag_weight = 0.01
        self.tag_loss_function = AsymmetricLoss(gamma_neg=7, gamma_pos=0, clip=0.05)

        # Bridger module
        bridge_dim = 128
        self.bridger_c2l_mlp = nn.Linear(2048, bridge_dim)
        self.bridger_l2c_mlp = nn.Linear(bridge_dim, 2048)
        self.bridger_o2l_mlp = nn.Linear(1408, bridge_dim)
        self.bridger_l2o_mlp = nn.Linear(bridge_dim, 1408)
        self.bridger_cl2ol_ca = CrossAttnBlock(num_heads=8, hidden_dim=bridge_dim, mlp_dim=bridge_dim)
        self.bridger_ol2cl_ca = CrossAttnBlock(num_heads=8, hidden_dim=bridge_dim, mlp_dim=bridge_dim)

        # Trainable parameters
        names = ["tag", "coe", "cee", "Qformer", "t5_proj", "bridger"]
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

    def coe_forward(self, samples, embeds):
        bz = len(samples["image"])
        image_embeds = embeds[:bz]
        region_embeds = embeds[bz:]

        # local-roi fusion
        rois_embeds = self.roi_align(image_embeds, samples)
        object_embeds = torch.cat([rois_embeds, region_embeds], -1)
        cap_embeds = self.coe_cap_mlp(object_embeds)
        tag_embeds = self.coe_tag_mlp(object_embeds)

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

    def prepare_control_tags(self, samples, tag_logits, full_drop_ratio=0.5, drop_ratio=0.5, tag_thr=0.7):
        control_tags = []

        if self.training:
            for bz_idx, cap in enumerate(samples["answers"]):
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
                    control_tag = ""
                else:
                    if l == 0:
                        control_tag = ""
                    else:
                        sl = torch.from_numpy(np.random.uniform(0, 1, l) > drop_ratio)
                        control_tag = [tags[tag_idx] for tag_idx in torch.nonzero(sl)]
                        # control_tag.sort()
                        random.shuffle(control_tag)
                        control_tag = ",".join(control_tag)
                control_tags.append(control_tag + "|")
        else:
            first_word_control = False
            if first_word_control:
                first_words = []
                for bz_idx, cap in enumerate(samples["answers"]):
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

            tag_scores = tag_logits.sigmoid()
            tag_idxs = (tag_scores > tag_thr).to(torch.long)
            stags = [[self.tag_list[tag_idx] for tag_idx in torch.nonzero(tag_idxs[bz_idx][:self.num_tags])]
                     for bz_idx in range(len(tag_idxs))]
            # tags = stags
            otags = [[self.tag_list[tag_idx] for tag_idx in torch.nonzero(tag_idxs[bz_idx][self.num_tags:])]
                     for bz_idx in range(len(tag_idxs))]
            tags = [stag + otag for stag, otag in zip(stags, otags)]

            if first_word_control:
                tags = [fword + tag for fword, tag in zip(first_words, tags)]
                tags = [set(tag) for tag in tags]

            for control_tag in tags:
                control_tag = list(set(control_tag))
                # control_tag.sort()
                control_tag = ",".join(control_tag)
                control_tags.append(control_tag + "|")

        return control_tags

    def cee_forward(self, tags, embeds):
        c_tokens = self.t5_tokenizer(
            tags,
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(embeds.device)
        c_embeds = self.t5_model.encoder.embed_tokens(c_tokens.input_ids)
        return c_embeds + self.cee_memory, c_tokens

    def bridge_forward(self, o_embeds, c_embeds):
        ol_embeds = self.bridger_o2l_mlp(o_embeds)
        cl_embeds = self.bridger_c2l_mlp(c_embeds)
        ol_embeds, attn_c2o = self.bridger_cl2ol_ca(ol_embeds, cl_embeds)
        cl_embeds, attn_o2c = self.bridger_ol2cl_ca(cl_embeds, ol_embeds)
        o_embeds = o_embeds + self.bridger_l2o_mlp(ol_embeds)
        c_embeds = c_embeds + self.bridger_l2c_mlp(cl_embeds)
        return o_embeds, c_embeds, (attn_c2o, attn_o2c)

    def loss(self, tag_logits, inputs_embeds, encoder_atts, samples):
        tags = samples["tags"].to(torch.long)
        loss_tag = self.tag_loss_function(tag_logits, tags) * self.tag_weight

        output_tokens = self.t5_tokenizer(
            samples["answers"],
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

        with self.maybe_autocast():
            embeds = self.ln_vision(self.visual_encoder(image))
            cap_embeds, tag_embeds = self.coe_forward(samples, embeds)
            tag_logits = self.tag_forward(samples, tag_embeds)
            control_tags = self.prepare_control_tags(samples, tag_logits)
            c_embeds, c_tokens = self.cee_forward(control_tags, embeds)
            cap_embeds, c_embeds, attn = self.bridge_forward(cap_embeds, c_embeds)

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
            encoder_atts = torch.cat([atts_t5, c_tokens.attention_mask], dim=1)
            inputs_embeds = torch.cat([inputs_t5, c_embeds], dim=1)

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
            cap_embeds, tag_embeds = self.coe_forward(samples, embeds)
            tag_logits = self.tag_forward(samples, tag_embeds)
            control_tags = self.prepare_control_tags(samples, tag_logits)
            c_embeds, c_tokens = self.cee_forward(control_tags, embeds)
            cap_embeds, c_embeds, attn = self.bridge_forward(cap_embeds, c_embeds)

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
            encoder_atts = torch.cat([atts_t5, c_tokens.attention_mask], dim=1)
            inputs_embeds = torch.cat([inputs_t5, c_embeds], dim=1)

            outputs = self.t5_model.generate(
                prefix_allowed_tokens_fn=None,
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

            output_scores = scores.cpu().numpy().tolist()
            output_captions = self.t5_tokenizer.batch_decode(
                sequences, skip_special_tokens=True
            )

        if self._apply_lemmatizer:
            output_captions = self._lemmatize(output_captions)

        output = []
        for id, output_caption, output_score in zip(samples["ids"], output_captions, output_scores):
            output.append(
                {"id": id, "caption": output_caption, "score": output_score, "tag_set1": [], "tag_set2": []}
            )

        return output

    @classmethod
    def from_config(cls, cfg):
        model = cls(**cfg)
        if cfg.pretrained is not None:
            model.load_checkpoint(url_or_filename=cfg.pretrained)
        return model





