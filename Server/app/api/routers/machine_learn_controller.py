from fastapi import APIRouter
from app.services.machine_learn_service import MachineLearningService
from app.api.deps import get_machine_learn_service
from app.model.flight import FlightClassifier, FlightRegression

service: MachineLearningService = get_machine_learn_service()

router = APIRouter(prefix="/v1/machineLearn", tags=["Machine Learn"])

@router.get("/analysis")
async def analysis():
    await service.analysis()
    return "Análise em andamento"

@router.get("/metrics")
async def metrics():
    metrics = await service.get_metrics()
    return metrics

@router.post("/regression/predict")
async def regression_predict(data: FlightRegression):
    return await service.regression_predict(data)

@router.post("/classifier/predict")
async def classifier_predict(data: FlightClassifier):
    return await service.classifier_predict(data)