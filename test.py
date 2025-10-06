import requests
import base64
from pathlib import Path
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import threading

# API base URL
BASE_URL = "http://localhost:8000"

# Lock for thread-safe printing
print_lock = threading.Lock()

def safe_print(*args, **kwargs):
    """Thread-safe print"""
    with print_lock:
        print(*args, **kwargs)

def load_image_base64(image_path: str) -> str:
    """Load and encode image to base64"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def submit_prediction(task_id: int, base64_image: str):
    """Submit a prediction request"""
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json={
                "id": task_id,
                "image": base64_image
            },
            timeout=30
        )
        return {
            "task_id": task_id,
            "success": response.status_code == 200,
            "response": response.json(),
            "status_code": response.status_code,
            "submit_time": datetime.now()
        }
    except Exception as e:
        return {
            "task_id": task_id,
            "success": False,
            "error": str(e),
            "submit_time": datetime.now()
        }

def check_status(task_id: int):
    """Check status of a task"""
    try:
        response = requests.get(f"{BASE_URL}/status/{task_id}", timeout=10)
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def check_health():
    """Check API health"""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def monitor_task(task_id: int, max_wait: int = 180):
    """Monitor a single task until completion"""
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        status = check_status(task_id)
        
        if status.get("status") == "done":
            elapsed = time.time() - start_time
            return {
                "task_id": task_id,
                "status": "done",
                "classes": status.get("classes", []),
                "elapsed_time": elapsed,
                "completed_at": status.get("completed_at")
            }
        elif status.get("status") == "error":
            elapsed = time.time() - start_time
            return {
                "task_id": task_id,
                "status": "error",
                "error": status.get("error"),
                "elapsed_time": elapsed
            }
        
        time.sleep(1)
    
    return {
        "task_id": task_id,
        "status": "timeout",
        "elapsed_time": max_wait
    }

def stress_test_burst(num_requests: int = 40, image_path: str = None):
    """Send all requests simultaneously and monitor health"""
    
    if not image_path or not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        return
    
    print("=" * 80)
    print(f"BURST STRESS TEST: {num_requests} SIMULTANEOUS REQUESTS")
    print("=" * 80)
    print(f"Image: {image_path}")
    print(f"Target: {BASE_URL}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Check initial health
    print("\n📊 Initial Health Check:")
    health = check_health()
    print(f"   Status: {health.get('status', 'unknown')}")
    print(f"   Model Loaded: {health.get('model_loaded', False)}")
    print(f"   Active Tasks: {health.get('active_tasks', 0)}")
    print(f"   Queue Size: {health.get('queue_size', 0)}")
    
    if not health.get("model_loaded"):
        print("\n❌ Model not loaded! Aborting test.")
        return
    
    # Load image once
    print("\n📁 Loading image...")
    base64_image = load_image_base64(image_path)
    print(f"   Image size: {len(base64_image)} characters")
    
    # Submit ALL requests simultaneously
    print(f"\n🚀 Submitting {num_requests} requests SIMULTANEOUSLY...")
    submission_results = []
    base_task_id = int(time.time() * 1000)
    
    submit_start = time.time()
    
    with ThreadPoolExecutor(max_workers=num_requests) as executor:
        futures = []
        for i in range(num_requests):
            task_id = base_task_id + i
            future = executor.submit(submit_prediction, task_id, base64_image)
            futures.append(future)
        
        # Collect all results
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            submission_results.append(result)
            
            if result["success"]:
                safe_print(f"   [{i}/{num_requests}] ✓ Task {result['task_id']} submitted")
            else:
                safe_print(f"   [{i}/{num_requests}] ✗ Task {result['task_id']} failed: {result.get('error', 'Unknown')}")
    
    submit_elapsed = time.time() - submit_start
    successful_submissions = sum(1 for r in submission_results if r["success"])
    
    print(f"\n📈 Submission Summary:")
    print(f"   Total Requests: {num_requests}")
    print(f"   Successful: {successful_submissions}")
    print(f"   Failed: {num_requests - successful_submissions}")
    print(f"   Submission Time: {submit_elapsed:.2f}s")
    print(f"   Submission Rate: {num_requests/submit_elapsed:.2f} req/s")
    
    # Monitor health immediately after submission
    print("\n📊 Health Check (Immediately After Submission):")
    health = check_health()
    print(f"   Status: {health.get('status', 'unknown')}")
    print(f"   Active Tasks: {health.get('active_tasks', 0)} 🔴")
    print(f"   Queue Size: {health.get('queue_size', 0)} 🔴")
    
    # Monitor health every 2 seconds for 20 seconds
    print("\n📊 Real-time Health Monitoring (20 seconds):")
    for i in range(10):
        time.sleep(2)
        health = check_health()
        active = health.get('active_tasks', 0)
        queue = health.get('queue_size', 0)
        
        indicator = "🔴" if (active > 0 or queue > 0) else "🟢"
        print(f"   [{i*2+2}s] {indicator} Active: {active:2d} | Queue: {queue:3d}")
    
    # Monitor all tasks
    print(f"\n⏳ Monitoring {successful_submissions} tasks for completion...")
    
    task_ids = [r["task_id"] for r in submission_results if r["success"]]
    
    completed_results = []
    completed_count = 0
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(monitor_task, task_id, 180): task_id for task_id in task_ids}
        
        for future in as_completed(futures):
            result = future.result()
            completed_results.append(result)
            completed_count += 1
            
            if result["status"] == "done":
                safe_print(f"   [{completed_count}/{successful_submissions}] ✓ Task {result['task_id']} completed in {result['elapsed_time']:.2f}s - Classes: {result['classes']}")
            elif result["status"] == "error":
                safe_print(f"   [{completed_count}/{successful_submissions}] ✗ Task {result['task_id']} error: {result.get('error')}")
            else:
                safe_print(f"   [{completed_count}/{successful_submissions}] ⏱ Task {result['task_id']} timeout")
    
    # Final Analysis
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    
    done_tasks = [r for r in completed_results if r["status"] == "done"]
    error_tasks = [r for r in completed_results if r["status"] == "error"]
    timeout_tasks = [r for r in completed_results if r["status"] == "timeout"]
    
    print(f"\n📊 Completion Status:")
    print(f"   Submitted Successfully: {successful_submissions}/{num_requests}")
    print(f"   Completed: {len(done_tasks)}")
    print(f"   Errors: {len(error_tasks)}")
    print(f"   Timeouts: {len(timeout_tasks)}")
    
    if done_tasks:
        processing_times = [r["elapsed_time"] for r in done_tasks]
        avg_time = sum(processing_times) / len(processing_times)
        min_time = min(processing_times)
        max_time = max(processing_times)
        
        print(f"\n⏱ Processing Times:")
        print(f"   Average: {avg_time:.2f}s")
        print(f"   Minimum: {min_time:.2f}s")
        print(f"   Maximum: {max_time:.2f}s")
        
        # Throughput calculation
        total_time = max_time
        throughput = len(done_tasks) / total_time if total_time > 0 else 0
        
        print(f"\n🚀 Performance:")
        print(f"   Total Time: {max_time:.2f}s")
        print(f"   Throughput: {throughput:.2f} tasks/second")
        print(f"   Expected with 3 workers: ~{3/avg_time:.2f} tasks/second")
        
        # Analyze detected classes
        all_classes = []
        for r in done_tasks:
            all_classes.extend(r.get("classes", []))
        
        if all_classes:
            from collections import Counter
            class_counts = Counter(all_classes)
            
            print(f"\n🔍 Detected Classes Distribution:")
            for cls, count in class_counts.most_common():
                percentage = (count / len(done_tasks)) * 100
                print(f"   {cls}: {count} detections ({percentage:.1f}% of tasks)")
        
        # Sample results
        print(f"\n📋 Sample Results (First 10):")
        for i, result in enumerate(done_tasks[:10], 1):
            classes = result.get("classes", [])
            print(f"   {i}. Task {result['task_id']}: {classes if classes else 'No jewelry'}")
    
    # Final health check
    print("\n📊 Final Health Check:")
    time.sleep(1)
    health = check_health()
    print(f"   Status: {health.get('status', 'unknown')}")
    print(f"   Active Tasks: {health.get('active_tasks', 0)}")
    print(f"   Queue Size: {health.get('queue_size', 0)}")
    
    print("\n" + "=" * 80)
    print("TEST COMPLETED")
    print("=" * 80)
    
    return {
        "total_requests": num_requests,
        "successful_submissions": successful_submissions,
        "completed": len(done_tasks),
        "errors": len(error_tasks),
        "timeouts": len(timeout_tasks),
        "avg_processing_time": avg_time if done_tasks else 0,
        "results": completed_results
    }

if __name__ == "__main__":
    import sys
    
    image_path = r"C:\Users\Crown Tech\jupyter\raresenc\microservice1\human_image_35.jpg"
    
    if len(sys.argv) > 1:
        num = int(sys.argv[1])
        stress_test_burst(num, image_path)
    else:
        # Default 40 requests
        stress_test_burst(50, image_path)