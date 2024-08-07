 | Data                        | Description                                                               | Download                                                                |
  | -------------------------------------- | ---------------------------------------------------------------------- | --------------------------------------------------------------------- |
  | Visual Genome  | `ln -s VG/images data/vg/images`  | [Official](https://homes.cs.washington.edu/~ranjay/visualgenome/api.html)   
  | MSCOCO 2014 | `ln -s coco2014/images data/refcoco/images`       | [Official](https://cocodataset.org/#home) |
  | Converted annotations | `unzip data.zip` | [OneDrive](https://mailsucasaccn-my.sharepoint.com/:f:/g/personal/zhaoyuzhong20_mails_ucas_ac_cn/Es5tiSmgeyBEtPAFBwJN8RAB9ZeP3oi21JA2YIETXZtupA?e=R4ev2J) |
  | Meteor package | `unzip meteor.zip -d controlcap/common/evaluation/` | [OneDrive]([https://mailsucasaccn-my.sharepoint.com/personal/zhaoyuzhong20_mails_ucas_ac_cn/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fzhaoyuzhong20%5Fmails%5Fucas%5Fac%5Fcn%2FDocuments%2FControlCap&view=0](https://mailsucasaccn-my.sharepoint.com/:f:/g/personal/zhaoyuzhong20_mails_ucas_ac_cn/Es5tiSmgeyBEtPAFBwJN8RAB9ZeP3oi21JA2YIETXZtupA?e=R4ev2J)) |
| Pre-trained ControlCap weights and logs (Optional) | `mv <your_path>/ckpts/* ckpts/` | [OneDrive](https://mailsucasaccn-my.sharepoint.com/:f:/g/personal/zhaoyuzhong20_mails_ucas_ac_cn/Es5tiSmgeyBEtPAFBwJN8RAB9ZeP3oi21JA2YIETXZtupA?e=R4ev2J) |

 P.S. Files in [BaiduDrive](https://pan.baidu.com/s/1SRdtx4NdOlFBmspXPUIAjg), the passpord is (3g1k).

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
P.S. The converted annotations are generated using `data.sh`, the original annotations are as follows:

1. [annotations](https://homes.cs.washington.edu/~ranjay/visualgenome/api.html) of Visual Genome for dense captioning.

2. [test_caption.json](https://drive.google.com/file/d/1zF3UGHU1rvgTujinqJ-hZtrCBVsfsuel/view?usp=sharing) and [mdetr_annotations](https://drive.google.com/file/d/1gvH5ToNtmIr3qz7C9lNi_fDmElwAANsI/view) of GlaMM for evaluating referring expression generation.
