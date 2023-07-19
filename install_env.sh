# create environment
python3 -m venv venv
source venv/bin/activate
python3 -m pip install -r requirements.txt

# obtain repo dependency
git clone --recurse-submodules https://github.com/hahnec/pala_dataset datasets/pala_dataset
