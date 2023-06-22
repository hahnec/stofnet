import zipfile

def zip_extract(data_path):
    if not data_path.exists():
        with zipfile.ZipFile(str(data_path)+'.zip', 'r') as zip_ref:
            zip_ref.extractall(data_path.parent)