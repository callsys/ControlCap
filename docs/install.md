To setup the environment of GenPromp, we use `conda` to manage our dependencies. Our developers use `CUDA 11.7` to do experiments. Run the following commands to install GenPromp:
 ```
conda create -n controlcap python=3.8 -y && conda activate controlcap
pip install --upgrade pip
pip install salesforce-lavis
pip install SceneGraphParser
python -m spacy download en
pip install textblob
 ```
Download [meteor](https://mailsucasaccn-my.sharepoint.com/:f:/g/personal/zhaoyuzhong20_mails_ucas_ac_cn/Es5tiSmgeyBEtPAFBwJN8RABZTkcA0LlymyURt4lsR4lKg?e=QaSVvu) and place it in `controlcap/common/evaluation/meteor`.
