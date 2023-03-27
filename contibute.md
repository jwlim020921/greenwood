# Contributing

## Intalling Requirments
```bash
pip install -r requirements.txt
```

## Updating Requirments
```bash
python -m pip freeze > requirments.txt
```

## Using Database
Place .env file in root directory <br>
Import database_helper.py, create a datbase object and use its methods. <br>

## Generate autodocumentaion
```bash
sphinx-apidoc -o docs greenwood
make clean html
```
