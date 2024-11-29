# change to bash
bash

# create and activate environment
python3 -m venv ~/npbrs-venv
source ~/npbrs-venv/bin/activate

# install packages using requirements.txt
pip install --index-url http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com -r requirements.txt
