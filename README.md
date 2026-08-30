# MLOps PyTorch Pipeline

An end-to-end MLOps pipeline for training and serving a PyTorch image classification model using Docker and Kubernetes.

The project uses CIFAR-10 with a simple convolutional neural network (CNN). Training and inference are containerized separately, while Kubernetes is used for training jobs, model serving, persistent storage and autoscaling.

## Architecture

```text
CIFAR-10 Dataset
       |
       v
PyTorch CNN Training
       |
       v
Model Checkpoint (.pt)
       |
       v
FastAPI Model Server
       |
       v
Docker Container
       |
       v
Kubernetes Deployment
       |
       v
Kubernetes Service
```

## Project Structure

```text
mlops-pytorch-pipeline/
├── .github/
│   └── workflows/
│       └── ci.yml
├── configs/
│   └── training_config.yaml
├── docker/
│   ├── Dockerfile.train
│   └── Dockerfile.serve
├── k8s/
│   ├── namespace.yaml
│   ├── configmap.yaml
│   ├── training-job.yaml
│   ├── serving-deployment.yaml
│   ├── serving-service.yaml
│   └── hpa.yaml
├── requirements/
│   ├── train.txt
│   └── serve.txt
├── src/
│   ├── model.py
│   ├── dataset.py
│   ├── train.py
│   └── serve.py
├── tests/
│   └── test_model.py
├── .gitignore
└── README.md
```

## Model

The project uses a simple CNN for CIFAR-10 image classification.

The model contains three convolutional blocks followed by fully connected layers. The output layer contains 10 units corresponding to the 10 CIFAR-10 classes.

## Local Training

Install the training dependencies:

```bash
pip install -r requirements/train.txt
```

Run training:

```bash
python src/train.py
```

Training parameters are defined in:

```text
configs/training_config.yaml
```

The best model checkpoint is saved to:

```text
checkpoints/classifier_v1.pt
```

## Docker Training

Build the training image:

```bash
docker build -f docker/Dockerfile.train -t mlops-train:v1 .
```

Run training with mounted data and checkpoint directories:

```bash
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/checkpoints:/app/checkpoints \
  mlops-train:v1
```

## Docker Model Serving

Build the serving image:

```bash
docker build -f docker/Dockerfile.serve -t mlops-serve:v1 .
```

Run the serving container:

```bash
docker run --rm -p 8080:8080 \
  -v $(pwd)/checkpoints:/app/checkpoints \
  mlops-serve:v1
```

## API Endpoints

Health check:

```bash
curl http://localhost:8080/health
```

Prediction:

```bash
curl -X POST http://localhost:8080/predict \
  -F "image=@test_image.png"
```

The prediction endpoint returns the predicted class ID and probabilities for all classes.

## Kubernetes Deployment

Create the namespace and training resources:

```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/training-job.yaml
```

Check the training workload:

```bash
kubectl get pods -n ml-training
```

After training completes, deploy the model serving components:

```bash
kubectl apply -f k8s/serving-deployment.yaml
kubectl apply -f k8s/serving-service.yaml
kubectl apply -f k8s/hpa.yaml
```

Check the deployment:

```bash
kubectl get pods -n ml-training
kubectl describe deployment model-serving -n ml-training
```

For local access to the Kubernetes service:

```bash
kubectl port-forward svc/model-serving 8080:80 -n ml-training
```

Then test the prediction endpoint:

```bash
curl -X POST http://localhost:8080/predict \
  -F "image=@test_image.png"
```

## Kubernetes Features

The Kubernetes configuration includes:

- Dedicated `ml-training` namespace
- ConfigMap for training configuration
- Kubernetes Job for model training
- Persistent storage for datasets and checkpoints
- Two model-serving replicas
- Liveness and readiness probes
- CPU and memory requests and limits
- Rolling update deployment strategy
- ClusterIP service
- Horizontal Pod Autoscaler

## Testing

Run the unit tests with:

```bash
pytest tests/ -v
```

The model test verifies that the CNN produces the expected output shape for CIFAR-10 classification.

## Continuous Integration

GitHub Actions is configured to run automated tests for pushes and pull requests to the `main` and `develop` branches.

The CI workflow:

1. Checks out the repository
2. Sets up Python 3.11
3. Installs project dependencies
4. Runs the pytest test suite

## Git Workflow

Development follows a feature branch workflow.

```text
main
  |
develop
  |
  +-- feature/project-setup
  +-- feature/docker-training
  +-- feature/k8s-deployment
  +-- feature/ci-testing
```

Changes are merged through Pull Requests with Conventional Commit messages.
