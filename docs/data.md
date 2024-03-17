 | Dataset & Files                        | Download                                                               | Description                                                                 |
  | -------------------------------------- | ---------------------------------------------------------------------- | --------------------------------------------------------------------- |
  | Visual Genome        | [Official](https://homes.cs.washington.edu/~ranjay/visualgenome/api.html)                        | Visual Genome dataset  
| MSCOCO 2014          | [Official](https://cocodataset.org/#home)                        | MSCOCO 2014 dataset  
  | data/          | [OneDrive](https://mailsucasaccn-my.sharepoint.com/personal/zhaoyuzhong20_mails_ucas_ac_cn/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fzhaoyuzhong20%5Fmails%5Fucas%5Fac%5Fcn%2FDocuments%2FControlCap&view=0)                        | Converted annotations for VG and RefCOCOg                                                     |

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
