# Basic Machine Learning Pipeline and the Utilization of Pickle.

Educative project about:
- Machine Learning Pipeline
- Utilization of Pickle

## Project installation

93.33% accuracy

94.12% accuracy because we added more data to the dataset.

95.88% accuracy because we added more data to the dataset.


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
uv run python model_training.py

```

in windows.
```bash
uv run model_training.py
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

Build docker image:
```bash
docker build -t ai-human-classifier:v1 .
```

```bash
docker run -p 8000:8000 ai-human-classifier:v1 
```


if you need to remove all containers:
```bash
docker rm -f $(docker ps -aq)
```

if you need to remove all images:
```bash
docker rmi -f $(docker images -aq)
```

if you need to remove all volumes:
```bash
docker volume rm $(docker volume ls -q)
```




Pushing images
You can push a new image to this repository using the CLI:

docker tag ai-human-classifier:v1 iRuperth/ai-human-classifier:latest
docker push iRuperth/ai-human-classifier:latest


docker push iruperth/pipeline-ml-docker
