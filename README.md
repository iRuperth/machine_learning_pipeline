# Basic Machine Learning Pipeline and the Utilization of Pickle.

Educative project about:
- Machine Learning Pipeline
- Utilization of Pickle

## Project installation

93.33% accuracy

94.12% accuracy because we added more data to the dataset.
```bash
source .venv/bin/activate
```

```bash
git clone https://github.com/iRuperth/machine_learning_pipeline.git
cd ai-vs-human-detector
uv venv
uv pip install -r requirements.txt
source .venv/Scripts/activate 
```

To train the model, run:
```bash
cd model
uv run python train_model.py

```

in windows.
```bash
uv run train_model.py
```

To run the API, run:
```bash
uv run uvicorn api.main:app --reload
```

To test the API, run:
```bash
cd test
uv run python test/prediction_test.py
```


Open Navigator
```bash
http://127.0.0.1:8000/docs
```

Request Example:
```bash
{
    "text": "This is a structured generated explanation with formal tone"
}
```