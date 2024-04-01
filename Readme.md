- Create the env:
```
python3 -m venv mistralpizza
```

- Activate the env:
```
source mistralpizza/bin/activate
```
- Install the requirements:

```
pip install -r requirements.txt
```

- Create a folder called model
```
mkdir folder_name
```
- Download the model from huggingface
```
wget https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q8_0.gguf?download=true
```

```chainlit run main.py -w```