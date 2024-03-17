 | Dataset & Files                        | Description                                                               | Download                                                                |
  | -------------------------------------- | ---------------------------------------------------------------------- | --------------------------------------------------------------------- |
  | Visual Genome  | Visual Genome dataset  | [Official](https://homes.cs.washington.edu/~ranjay/visualgenome/api.html)   
| MSCOCO 2014  | MSCOCO 2014 dataset        | [Official](https://cocodataset.org/#home) 
  | data/          | Converted annotations for VG and RefCOCOg | [OneDrive](https://mailsucasaccn-my.sharepoint.com/personal/zhaoyuzhong20_mails_ucas_ac_cn/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fzhaoyuzhong20%5Fmails%5Fucas%5Fac%5Fcn%2FDocuments%2FControlCap&view=0)                 |

  | Models       | Description       | VG V1.2 (Dense captioning) | VG (Referring expression generation)  | RefCOCOg (Referring expression generation) |
  | -----------  | ----------------- | -- | --------- | -------------------------- |
  | [ControlCap](https://mailsucasaccn-my.sharepoint.com/:u:/g/personal/zhaoyuzhong20_mails_ucas_ac_cn/ESHZeEf0p-5FmH6esN9XZ8MBv_xalVZm4NhPabgg4Wgvvg?e=G8iJ26) | Trained on VG and RefCOCOg for 5 epochs | mAP (44.7) | METEOR (20.5) CIDEr (183.3) | - |
  | [ControlCap](https://mailsucasaccn-my.sharepoint.com/:u:/g/personal/zhaoyuzhong20_mails_ucas_ac_cn/EQf7kx1AgSBHoLWu1BgWpWoB7JKy-I98Om9EMM1VNFhEkg?e=nauMkF) | Finetuned on RefCOCOg for 3 epochs | - | - | METEOR (17.8) CIDEr (112.8) |
 

To train and evaluate ControlCap, download the files in the table and arrange the files according to the file tree below. (Uploading)

```text
    |--ControlCap/
      |--data/
        |--vg/
           |--controlcap/
           |--images/
              |--1000.jpg
              |--1001.jpg
              ...
        |--refcoco
           |--controlcap/
           |--images/
              |--COCO_train2014_000000000009.jpg
              |--COCO_train2014_000000000025.jpg
              ...
           ...
      |--ckpts/
         |--vg1.2_refcocog_5e.pth
         |--refcocog_gt.pth
      |--configs/
      |--controlcap/
      |--docs/
      |--scripts/
      |--train.py
      |--eval.py
```
