# create environment
python3 -m venv venv
source venv/bin/activate
python3 -m pip install -r requirements.txt

# obtain repo dependencies
git clone --recurse-submodules git@github.com:hahnec/pala_dataset datasets/pala_dataset
python3 -m pip install -r datasets/pala_dataset/requirements.txt