"""
Health Check Endpoints
System status, GPU info, model status
"""

from fastapi import APIRouter, Request
from app.core.environment import get_system_info
from app.schemas.health_schema import HealthResponse, SystemInfoResponse
from app.monitoring.gpu_monitor import gpu_monitor
from app.monitoring.metrics import global_metrics

router = APIRouter()


@router.get("/", response_model=HealthResponse)
async def health_check():
    """Basic health check"""
    return HealthResponse(
        status="healthy",
        message="CSLR Backend is running"
    )


@router.get("/system", response_model=SystemInfoResponse)
async def system_info():
    """Detailed system information"""
    info = get_system_info()
    return SystemInfoResponse(**info)


@router.get("/models")
async def model_status(request: Request):
    """Check model loading status"""

    # Check if InferenceService is initialized
    service_loaded = False
    module_status = {}
    
    if request and hasattr(request, 'app') and hasattr(request.app.state, 'inference_service'):
        service = request.app.state.inference_service
        if service:
            service_loaded = True
            # Check individual modules
            module_status = {
                "rgb_stream": "loaded" if hasattr(service, 'rgb_stream') and service.rgb_stream else "not_loaded",
                "pose_stream": "loaded" if hasattr(service, 'pose_stream') and service.pose_stream else "not_loaded",
                "fusion": "loaded" if hasattr(service, 'fusion') and service.fusion else "not_loaded",
                "temporal_model": "loaded" if hasattr(service, 'temporal_model') and service.temporal_model else "not_loaded",
                "ctc_decoder": "loaded" if hasattr(service, 'ctc_decoder') and service.ctc_decoder else "not_loaded",
                "grammar_corrector": "loaded" if hasattr(service, 'grammar_corrector') else "not_loaded"
            }
    
    return {
        "inference_service": "loaded" if service_loaded else "not_loaded",
        "modules": module_status,
        "status": "operational" if service_loaded else "degraded"
    }


@router.get("/gpu")
async def gpu_status():
    """GPU and system resource status"""
    return gpu_monitor.get_full_status()


@router.get("/metrics")
async def metrics_summary():
    """Inference performance metrics summary"""
    return global_metrics.get_summary()


@router.get("/cache")
async def cache_status(request: Request):
    """Cache availability status"""
    cache_service = getattr(request.app.state, "cache_service", None)
    return {"enabled": bool(cache_service and getattr(cache_service, "enabled", False))}
