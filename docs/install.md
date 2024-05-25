We use `conda` to manage our dependencies. Our developers use `CUDA 12.1` to do experiments. Run the following commands to to setup the environment of ControlCap:
 ```
git clone https://github.com/callsys/ControlCap
cd ControlCap

conda create -n controlcap python=3.8 -y
conda activate controlcap

bash scripts/setup.sh
 ```
