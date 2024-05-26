### 4.3 Training

Training ControlCap on VG V1.2 and RefCOCOg.
```
bash scripts/train train_vg1.2_refcocog.sh
```
Finetuning ControlCap on RefCOCOg (Pretrained by VG V1.2 and RefCOCOg).
```
bash scripts/train finetune_refcocog.sh ckpts/vg1.2_refcocog_5e.pth
```
Training ControlCap on VG V1.2.
```
bash scripts/train train_vg1.2.sh
```
Training ControlCap on VGCOCO.
```
bash scripts/train train_vgcoco.sh
```

### 4.4 Evaluation
Evaluating the dense captioning performance of ControlCap on VG V1.2 : (`mAP 44.7`).
```
bash scripts/eval/eval_vg1.2_densecap.sh ckpts/vg1.2_refcocog_5e.pth
```
Evaluating the referring expression generation performance of ControlCap on VG : (`METEOR 20.5, CIDEr 183.3`).
```
bash scripts/eval/eval_vg_reg.sh ckpts/vg1.2_refcocog_5e.pth
```
Evaluating the referring expression generation performance of ControlCap on RefCOCOg : (`METEOR 17.8, CIDEr 112.8`).
```
bash scripts/eval/eval_refcocog_reg.sh ckpts/refcocog_ft.pth
```


