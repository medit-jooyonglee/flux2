"""Flask server exposing FLUX.2 [klein] 4B inference (scripts/bench_klein4b.py) over HTTP.

Endpoint: POST /generate
  Why POST (not GET/PUT): this creates a new generated image from submitted input each
  call -- it's not a safe/cacheable lookup (GET) and not an idempotent replace of a
  resource at a fixed URL (PUT). POST is the right verb for "run this job with this
  input".

  Headers (all optional except X-Prompt):
    X-Prompt             (required) text prompt
    X-Width              default 1024
    X-Height             default 1024
    X-Num-Steps          default: model default (4 for klein 4B)
    X-Guidance           default: model default (1.0 for klein 4B)
    X-Seed               default 0
    X-Match-Image-Size   index into reference images to match output size to, optional
    X-Cpu-Offloading     "true"/"false", default "true"
    X-Image-Urls         comma-separated URLs the server should download as reference
                          image(s) for editing/multi-reference generation, optional

  Body (optional, pick one):
    - multipart/form-data with one or more "images" file parts -> multiple reference
      images, in upload order (use this for several local files at once)
    - raw image bytes (Content-Type: application/octet-stream) -> a single reference
      image
    Either is combined with any images fetched from X-Image-Urls (those come last).
    Omit all three for plain text-to-image.

  Response: image/png bytes. Header X-Elapsed-Ms carries the generation time.

Run:
  PYTHONPATH=src python scripts/server.py --host=0.0.0.0 --port=8000

Note: this is a single-worker dev server on purpose -- GPU inference is inherently
serialized anyway (one generation at a time given VRAM limits), so concurrency
beyond the Flask dev server isn't useful here without a request queue.
"""

import argparse
import io
import os
import shutil
import sys
import tempfile
import threading
import traceback
from pathlib import Path

import requests
import torch
from flask import Flask, Response, request
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import bench_klein4b as bk  # noqa: E402

app = Flask(__name__)


def _parse_bool(s: str | None, default: bool) -> bool:
    if s is None:
        return default
    return s.strip().lower() in ("1", "true", "yes", "on")


def _download_image(url: str) -> Image.Image:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content)).convert("RGB")


_FATAL_CUDA_MARKERS = ("out of memory", "cuda error", "device-side assert")


def _is_fatal_cuda_error(e: Exception) -> bool:
    msg = str(e).lower()
    return any(marker in msg for marker in _FATAL_CUDA_MARKERS)


@app.route("/health", methods=["GET"])
def health():
    return {"status": "ok", "models_loaded": bk._MODELS is not None}


@app.route("/generate", methods=["POST"])
def generate():
    prompt = request.headers.get("X-Prompt")
    if not prompt:
        return Response("Missing required header: X-Prompt", status=400)

    try:
        width = int(request.headers.get("X-Width", 1024))
        height = int(request.headers.get("X-Height", 1024))
        num_steps = request.headers.get("X-Num-Steps")
        num_steps = int(num_steps) if num_steps else None
        guidance = request.headers.get("X-Guidance")
        guidance = float(guidance) if guidance else None
        seed = int(request.headers.get("X-Seed", 0))
        match_image_size = request.headers.get("X-Match-Image-Size")
        match_image_size = int(match_image_size) if match_image_size else None
        cpu_offloading = _parse_bool(request.headers.get("X-Cpu-Offloading"), True)
        image_urls = request.headers.get("X-Image-Urls", "")
    except ValueError as e:
        return Response(f"Invalid header value: {e}", status=400)

    images: list[Image.Image] = []

    upload_files = request.files.getlist("images") if request.files else []
    if upload_files:
        for f in upload_files:
            try:
                images.append(Image.open(f.stream).convert("RGB"))
            except Exception as e:
                return Response(f"Invalid image upload ({f.filename}): {e}", status=400)
    elif request.data:
        try:
            images.append(Image.open(io.BytesIO(request.data)).convert("RGB"))
        except Exception as e:
            return Response(f"Invalid image body: {e}", status=400)

    for url in [u.strip() for u in image_urls.split(",") if u.strip()]:
        try:
            images.append(_download_image(url))
        except Exception as e:
            return Response(f"Failed to fetch {url}: {e}", status=400)

    # infer() takes local file paths (comma-separated), so persist any in-memory
    # images to a scratch dir for the duration of this request.
    tmp_dir = None
    input_images = ""
    if images:
        tmp_dir = tempfile.mkdtemp(prefix="flux2_req_")
        paths = []
        for i, img in enumerate(images):
            p = os.path.join(tmp_dir, f"ref_{i}.png")
            img.save(p)
            paths.append(p)
        input_images = ",".join(paths)

    try:
        result_img, elapsed = bk.infer(
            prompt=prompt,
            width=width,
            height=height,
            match_image_size=match_image_size,
            num_steps=num_steps,
            guidance=guidance,
            seed=seed,
            input_images=input_images,
            cpu_offloading=cpu_offloading,
        )
    except Exception as e:
        traceback.print_exc()
        fatal = _is_fatal_cuda_error(e)

        try:
            torch.cuda.empty_cache()
        except Exception:
            # If this itself throws, the CUDA context is definitely poisoned.
            fatal = True

        if fatal:
            print(
                "FATAL: CUDA context is likely corrupted after this error. "
                "Exiting so the process can be restarted (see scripts/run_server_forever.ps1).",
                file=sys.stderr,
            )
            # Give Flask a moment to flush this response before killing the process.
            threading.Timer(1.0, lambda: os._exit(1)).start()

        return Response(f"Generation failed: {e}", status=500)
    finally:
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    buf = io.BytesIO()
    result_img.save(buf, format="PNG")
    resp = Response(buf.getvalue(), mimetype="image/png")
    resp.headers["X-Elapsed-Ms"] = f"{elapsed * 1000:.1f}"
    return resp


if __name__ == "__main__":
    import torch
    torch.cuda.set_device('cuda:5')
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8602)
    parser.add_argument("--preload", action="store_true", default=True)
    parser.add_argument("--no-preload", dest="preload", action="store_false")
    args = parser.parse_args()

    if args.preload:
        # Load models at startup rather than on the first request.
        bk.get_models(cpu_offloading=False)

    app.run(host=args.host, port=args.port, threaded=False)
