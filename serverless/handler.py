"""RunPod serverless handler for ComfyUI.

Event input:
  {
    "workflow":  <ComfyUI API-format workflow dict>,   # required
    "images":    [{"name": "...", "image": "<b64>"}],  # optional inputs
  }

Returns:
  {"images": [{"filename": "...", "type": "base64", "data": "<b64>"}]}
  or {"error": "..."}  on failure.
"""

import base64
import json
import os
import time
import urllib.parse
import urllib.request
import uuid

import requests
import runpod

COMFY_HOST = os.environ.get("COMFY_HOST", "127.0.0.1:8188")
COMFY_URL = f"http://{COMFY_HOST}"
COMFY_BOOT_TIMEOUT_S = int(os.environ.get("COMFY_BOOT_TIMEOUT_S", 300))
COMFY_JOB_TIMEOUT_S = int(os.environ.get("COMFY_JOB_TIMEOUT_S", 600))
REFRESH_WORKER = os.environ.get("REFRESH_WORKER", "false").lower() == "true"

CLIENT_ID = str(uuid.uuid4())


def wait_for_comfy(timeout_s: int) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            r = requests.get(f"{COMFY_URL}/system_stats", timeout=3)
            if r.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(0.5)
    raise RuntimeError(f"ComfyUI did not become ready within {timeout_s}s")


def upload_input_images(images):
    for img in images or []:
        name = img.get("name")
        data = img.get("image")
        if not name or not data:
            continue
        raw = base64.b64decode(data)
        files = {"image": (name, raw, "application/octet-stream")}
        data_fields = {"overwrite": "true", "type": "input"}
        r = requests.post(
            f"{COMFY_URL}/upload/image",
            files=files,
            data=data_fields,
            timeout=60,
        )
        r.raise_for_status()


def queue_prompt(workflow):
    payload = json.dumps({"prompt": workflow, "client_id": CLIENT_ID}).encode("utf-8")
    req = urllib.request.Request(
        f"{COMFY_URL}/prompt",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def wait_for_completion(prompt_id: str, timeout_s: int):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        r = requests.get(f"{COMFY_URL}/history/{prompt_id}", timeout=10)
        r.raise_for_status()
        history = r.json()
        if prompt_id in history:
            return history[prompt_id]
        time.sleep(0.5)
    raise TimeoutError(f"Job {prompt_id} did not complete within {timeout_s}s")


def collect_outputs(prompt_history):
    outputs = prompt_history.get("outputs", {})
    results = []
    for _node_id, node_output in outputs.items():
        for image in node_output.get("images", []) or []:
            qs = urllib.parse.urlencode(
                {
                    "filename": image["filename"],
                    "subfolder": image.get("subfolder", ""),
                    "type": image.get("type", "output"),
                }
            )
            r = requests.get(f"{COMFY_URL}/view?{qs}", timeout=60)
            r.raise_for_status()
            results.append(
                {
                    "filename": image["filename"],
                    "type": "base64",
                    "data": base64.b64encode(r.content).decode("utf-8"),
                }
            )
    return results


def handler(event):
    job_input = event.get("input") or {}
    workflow = job_input.get("workflow")
    if not workflow:
        return {"error": "Missing required 'workflow' field in input"}

    try:
        wait_for_comfy(COMFY_BOOT_TIMEOUT_S)
        upload_input_images(job_input.get("images"))
        queued = queue_prompt(workflow)
        prompt_id = queued.get("prompt_id")
        if not prompt_id:
            return {"error": f"ComfyUI rejected prompt: {queued}"}
        history = wait_for_completion(prompt_id, COMFY_JOB_TIMEOUT_S)
        images = collect_outputs(history)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}

    result = {"images": images}
    if REFRESH_WORKER:
        result["refresh_worker"] = True
    return result


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
