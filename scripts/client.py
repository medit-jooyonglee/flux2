"""Client for scripts/server.py.

Usage:
  # plain text-to-image
  python scripts/client.py --prompt="a cat in a hat" --output=out.png

  # editing with a single local reference image (sent as raw binary body)
  python scripts/client.py --prompt="add a wizard hat" --image=ref.png --output=out.png

  # multiple local reference images (sent as multipart/form-data)
  python scripts/client.py --prompt="put the shirt from image 2 on the person in image 1" \
      --images="person.png,shirt.png" --output=out.png

  # editing with reference image(s) by URL (server downloads them)
  python scripts/client.py --prompt="add a wizard hat" --image_url="https://.../cat.png" --output=out.png

  # multiple reference images by URL
  python scripts/client.py --prompt="combine these" --image_url="https://a.png,https://b.png" --output=out.png
"""

import argparse
import sys
from pathlib import Path
import time
import requests

def time_strftime(under_sec=1000):
    t0 = time.time()
    ms = str(int((t0 - int(t0)) * under_sec))
    return time.strftime('%Y%m%d%H%M%S') + ms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="http://10.100.1.45:8602/")
    parser.add_argument("--prompt", default="a pretty sexy bikiny koregirl")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--num_steps", type=int, default=None)
    parser.add_argument("--guidance", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--match_image_size", type=int, default=None)
    parser.add_argument("--cpu_offloading", default="false", choices=["true", "false"])
    parser.add_argument("--image", default=None, help="single local image file, sent as raw binary body")
    parser.add_argument(
        "--images", default=None, help="comma-separated local image files, sent as multipart/form-data"
    )
    parser.add_argument("--image_url", default=None, help="comma-separated URL(s) for the server to fetch")
    parser.add_argument("--output", default="output/client_result")
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()

    # args.prompt = 'pretty korean baby with yewllow big smile teeth'
    headers = {
        "X-Prompt": args.prompt,
        "X-Width": str(args.width),
        "X-Height": str(args.height),
        "X-Seed": str(args.seed),
        "X-Cpu-Offloading": args.cpu_offloading,
    }
    print(f"Prompt: {args.prompt}")
    if args.num_steps is not None:
        headers["X-Num-Steps"] = str(args.num_steps)
    if args.guidance is not None:
        headers["X-Guidance"] = str(args.guidance)
    if args.match_image_size is not None:
        headers["X-Match-Image-Size"] = str(args.match_image_size)
    if args.image_url:
        headers["X-Image-Urls"] = args.image_url

    assert not (args.image and args.images), "use either --image or --images, not both"

    body = b""
    files = None
    open_handles = []
    if args.images:
        paths = [p.strip() for p in args.images.split(",") if p.strip()]
        open_handles = [open(p, "rb") for p in paths]
        files = [("images", (Path(p).name, fh, "application/octet-stream")) for p, fh in zip(paths, open_handles)]
    elif args.image:
        with open(args.image, "rb") as f:
            body = f.read()
        headers["Content-Type"] = "application/octet-stream"

    try:
        resp = requests.post(
            f"{args.server}/generate", headers=headers, data=body, files=files, timeout=args.timeout
        )
    finally:
        for fh in open_handles:
            fh.close()

    if resp.status_code != 200:
        print(f"Error {resp.status_code}: {resp.text}", file=sys.stderr)
        sys.exit(1)

    save_name = f'{args.output}{time_strftime()}.jpg'
    Path(save_name).parent.mkdir(parents=True, exist_ok=True)
    with open(save_name, "wb") as f:
        f.write(resp.content)
    print(f"Saved {save_name}  (elapsed: {resp.headers.get('X-Elapsed-Ms', '?')} ms)")


if __name__ == "__main__":
    main()
