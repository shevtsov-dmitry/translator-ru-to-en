# translator-ru-to-en

## Run guide

```bash
python -m venv . 
source ./bin/activate
./bin/pip install -U $(cat packages.txt)
./bin/python main.py
```

## Use as script

```bash
chmod +x ./main.py
sudo ln -s $(realpath main.py) /usr/local/bin/translator-ru-to-en
```

_terminal restart required_
