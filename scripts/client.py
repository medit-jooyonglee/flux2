"""Client for scripts/server.py.

Usage:
  # plain text-to-image
  python scripts/client.py --prompt="a cat in a hat" --output=out.png

  # editing with a local reference image (sent as raw binary body)
  python scripts/client.py --prompt="add a wizard hat" --image=ref.png --output=out.png

  # editing with reference image(s) by URL (server downloads them)
  python scripts/client.py --prompt="add a wizard hat" --image_url="https://.../cat.png" --output=out.png

  # multiple reference images by URL
  python scripts/client.py --prompt="combine these" --image_url="https://a.png,https://b.png" --output=out.png
"""

import argparse
import sys
from pathlib import Path

import requests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="http://127.0.0.1:8000")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--num_steps", type=int, default=None)
    parser.add_argument("--guidance", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--match_image_size", type=int, default=None)
    parser.add_argument("--cpu_offloading", default="true", choices=["true", "false"])
    parser.add_argument("--image", default=None, help="local image file, sent as raw binary body")
    parser.add_argument("--image_url", default=None, help="comma-separated URL(s) for the server to fetch")
    parser.add_argument("--output", default="output/client_result.png")
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()

    headers = {
        "X-Prompt": args.prompt,
        "X-Width": str(args.width),
        "X-Height": str(args.height),
        "X-Seed": str(args.seed),
        "X-Cpu-Offloading": args.cpu_offloading,
    }
    if args.num_steps is not None:
        headers["X-Num-Steps"] = str(args.num_steps)
    if args.guidance is not None:
        headers["X-Guidance"] = str(args.guidance)
    if args.match_image_size is not None:
        headers["X-Match-Image-Size"] = str(args.match_image_size)
    if args.image_url:
        headers["X-Image-Urls"] = args.image_url

    body = b""
    if args.image:
        with open(args.image, "rb") as f:
            body = f.read()
        headers["Content-Type"] = "application/octet-stream"

    resp = requests.post(f"{args.server}/generate", headers=headers, data=body, timeout=args.timeout)
    if resp.status_code != 200:
        print(f"Error {resp.status_code}: {resp.text}", file=sys.stderr)
        sys.exit(1)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        f.write(resp.content)
    print(f"Saved {args.output}  (elapsed: {resp.headers.get('X-Elapsed-Ms', '?')} ms)")


if __name__ == "__main__":
    main()
