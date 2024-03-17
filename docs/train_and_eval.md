### 4.3 Training

Training ControlCap on VG V1.2 and RefCOCOg.
```
bash scripts/train train_vg1.2_refcocog.sh
```
Finetuning ControlCap on RefCOCOg (Pretrained by VG V1.2 and RefCOCOg).
```
bash scripts/train finetune_refcocog.sh
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
Evaluate the dense captioning performance of ControlCap on VG V1.2.
```
bash scripts/eval/eval_vg1.2_densecap.sh ckpts/vg1.2_refcocog_5e.pth
```
Evaluate the referring expression generation performance of ControlCap on VG V1.2.
```
bash scripts/eval/eval_vg_reg.sh ckpts/vg1.2_refcocog_5e.pth
```
Evaluate the referring expression generation performance of ControlCap on RefCOCOg.
```
bash scripts/eval/eval_refcocog_reg.sh ckpts/refcocog_ft.pth
```


