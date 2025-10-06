from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field, ConfigDict
from typing import Dict, Optional, List
import asyncio
import base64
from io import BytesIO
from PIL import Image
from datetime import datetime
from enum import Enum
import logging
import sys
from pathlib import Path
import torch
from contextlib import asynccontextmanager
import threading

# Import inference module
import inference

# CRITICAL: Make inference classes available in __main__ for unpickling
sys.modules['__main__'].LabelEncoder = inference.LabelEncoder
sys.modules['__main__'].TrainingConfig = inference.TrainingConfig
sys.modules['__main__'].MultiLabelViT = inference.MultiLabelViT

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# MODEL CONFIGURATION
# ============================================================
CHECKPOINT_PATH = r"C:\Users\Crown Tech\jupyter\raresenc\microservice1\best_model.pth"
THRESHOLD = 0.55
DEVICE = "cpu"
MAX_WORKERS = 7
# ============================================================

# Enums for status
class TaskStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    DONE = "done"
    ERROR = "error"

# Pydantic models
class PredictRequest(BaseModel):
    id: int = Field(..., description="Unique integer identifier for the request", gt=0)
    image: str = Field(..., description="Base64 encoded image")

class PredictResponse(BaseModel):
    id: int
    status: TaskStatus
    message: str

class StatusResponse(BaseModel):
    id: int
    status: TaskStatus
    classes: Optional[List[str]] = None
    error: Optional[str] = None
    created_at: Optional[str] = None
    completed_at: Optional[str] = None

class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    status: str
    timestamp: str
    active_tasks: int
    queue_size: int
    model_loaded: bool

# In-memory storage
tasks: Dict[int, Dict] = {}

# Thread-safe semaphore for limiting workers
worker_semaphore = threading.Semaphore(MAX_WORKERS)

# Thread-safe counters
counter_lock = threading.Lock()
active_count = 0
queued_count = 0

# Global model objects
model = None
label_encoder = None
transform = None
device = None

# Helper functions
def validate_base64_image(base64_string: str) -> bool:
    """Validate if the base64 string is a valid image"""
    try:
        if "," in base64_string:
            base64_string = base64_string.split(",")[1]
        
        image_data = base64.b64decode(base64_string)
        image = Image.open(BytesIO(image_data))
        image.verify()
        
        return True
    except Exception as e:
        logger.error(f"Invalid image: {str(e)}")
        return False

def base64_to_pil_image(base64_string: str) -> Image.Image:
    """Convert base64 string to PIL Image"""
    if "," in base64_string:
        base64_string = base64_string.split(",")[1]
    
    image_data = base64.b64decode(base64_string)
    image = Image.open(BytesIO(image_data)).convert('RGB')
    return image

def increment_queued():
    """Thread-safe increment of queued counter"""
    global queued_count
    with counter_lock:
        queued_count += 1

def decrement_queued():
    """Thread-safe decrement of queued counter"""
    global queued_count
    with counter_lock:
        queued_count -= 1

def increment_active():
    """Thread-safe increment of active counter"""
    global active_count
    with counter_lock:
        active_count += 1

def decrement_active():
    """Thread-safe decrement of active counter"""
    global active_count
    with counter_lock:
        active_count -= 1

def process_image_sync(task_id: int, base64_image: str):
    """Synchronous background task to process the image"""
    global queued_count, active_count
    
    # Task is queued initially
    increment_queued()
    logger.info(f"Task {task_id}: In queue (queued={queued_count}, active={active_count})")
    
    # Acquire semaphore (blocks if 3 workers are busy)
    worker_semaphore.acquire()
    
    try:
        # Move from queued to active
        decrement_queued()
        increment_active()
        
        tasks[task_id]["status"] = TaskStatus.PROCESSING
        logger.info(f"Task {task_id}: STARTED PROCESSING (queued={queued_count}, active={active_count})")
        
        # Convert base64 to PIL Image
        pil_image = base64_to_pil_image(base64_image)
        logger.info(f"Task {task_id}: Image converted. Size: {pil_image.size}")
        
        # Save temp image
        temp_image_path = Path(f"temp_image_{task_id}.jpg")
        pil_image.save(temp_image_path)
        
        try:
            # Run inference
            logger.info(f"Task {task_id}: Running inference...")
            predictions, all_probs = inference.predict_image(
                str(temp_image_path), 
                model, 
                label_encoder, 
                transform, 
                device, 
                THRESHOLD
            )
            
            # Extract class names
            classes = [p['class'] for p in predictions]
            
            logger.info(f"Task {task_id}: Found {len(classes)} classes: {classes}")
            
            # Update task with result
            tasks[task_id]["status"] = TaskStatus.DONE
            tasks[task_id]["classes"] = classes if classes else []
            tasks[task_id]["completed_at"] = datetime.utcnow().isoformat()
            
            logger.info(f"Task {task_id}: COMPLETED")
            
        finally:
            # Clean up temp file
            if temp_image_path.exists():
                temp_image_path.unlink()
        
    except Exception as e:
        logger.error(f"Task {task_id}: ERROR - {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        tasks[task_id]["status"] = TaskStatus.ERROR
        tasks[task_id]["error"] = str(e)
        tasks[task_id]["completed_at"] = datetime.utcnow().isoformat()
    
    finally:
        # Release semaphore and decrement active counter
        decrement_active()
        worker_semaphore.release()
        logger.info(f"Task {task_id}: FINISHED (queued={queued_count}, active={active_count})")

# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global model, label_encoder, transform, device
    
    logger.info(f"Server starting with {MAX_WORKERS} concurrent workers")
    
    try:
        if not Path(CHECKPOINT_PATH).exists():
            logger.error(f"Checkpoint not found: {CHECKPOINT_PATH}")
        else:
            device = torch.device(DEVICE if torch.cuda.is_available() else 'cpu')
            logger.info(f"Using device: {device}")
            
            logger.info("Loading jewelry classification model...")
            model, label_encoder = inference.load_model(CHECKPOINT_PATH, device)
            transform = inference.get_transforms()
            
            logger.info(f"Model loaded successfully!")
            logger.info(f"Classes: {label_encoder.classes}")
            logger.info(f"Number of classes: {label_encoder.num_classes}")
            
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        model = None
        label_encoder = None
        transform = None
    
    yield
    
    # Shutdown
    logger.info("Server shutting down")
    
    for temp_file in Path(".").glob("temp_image_*.jpg"):
        try:
            temp_file.unlink()
        except:
            pass

# Initialize FastAPI app
app = FastAPI(title="Jewelry Classification API", version="1.0.0", lifespan=lifespan)

# API Endpoints

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest, background_tasks: BackgroundTasks):
    """Submit an image for jewelry classification"""
    try:
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded. Check server logs.")
        
        if not validate_base64_image(request.image):
            raise HTTPException(status_code=400, detail="Invalid base64 image format")
        
        if request.id in tasks:
            raise HTTPException(status_code=409, detail=f"Task with ID {request.id} already exists")
        
        # Initialize task
        tasks[request.id] = {
            "id": request.id,
            "status": TaskStatus.QUEUED,
            "classes": None,
            "error": None,
            "created_at": datetime.utcnow().isoformat(),
            "completed_at": None
        }
        
        # Add processing task to background (runs in thread pool)
        background_tasks.add_task(process_image_sync, request.id, request.image)
        
        logger.info(f"Task {request.id}: Added to queue")
        
        return PredictResponse(
            id=request.id,
            status=TaskStatus.QUEUED,
            message="Task submitted successfully and will be processed"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in predict endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/status/{task_id}", response_model=StatusResponse)
async def status(task_id: int):
    """Check the status of a processing task by ID"""
    try:
        if task_id not in tasks:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
        
        task = tasks[task_id]
        
        return StatusResponse(
            id=task["id"],
            status=task["status"],
            classes=task.get("classes"),
            error=task.get("error"),
            created_at=task.get("created_at"),
            completed_at=task.get("completed_at")
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in status endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    try:
        with counter_lock:
            current_active = active_count
            current_queued = queued_count
        
        return HealthResponse(
            status="healthy" if model is not None else "degraded",
            timestamp=datetime.utcnow().isoformat(),
            active_tasks=current_active,
            queue_size=current_queued,
            model_loaded=model is not None
        )
    
    except Exception as e:
        logger.error(f"Error in health endpoint: {str(e)}")
        return HealthResponse(
            status="unhealthy",
            timestamp=datetime.utcnow().isoformat(),
            active_tasks=0,
            queue_size=0,
            model_loaded=False
        )

@app.delete("/tasks/{task_id}")
async def delete_task(task_id: int):
    """Delete a completed or errored task"""
    try:
        if task_id not in tasks:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
        
        task_status = tasks[task_id]["status"]
        if task_status in [TaskStatus.QUEUED, TaskStatus.PROCESSING]:
            raise HTTPException(status_code=400, detail="Cannot delete task that is still processing")
        
        del tasks[task_id]
        return {"message": f"Task {task_id} deleted successfully"}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting task: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)