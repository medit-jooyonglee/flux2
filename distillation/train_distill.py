"""Phase 1 capacity distillation: FLUX.2 Klein 4B Base (Teacher, frozen) -> ~1B Student
(trained), per docs/flux2_distillation_05b_1b.md section 32's "1차 실험 Recipe".

Per training step: load a cached (image latent, text embedding) pair, sample noise and a
random timestep t, build the noisy latent x_t via flow-matching interpolation, run BOTH
Teacher (frozen, no_grad) and Student on the exact same (x_t, t, text embedding), and
match the Student's prediction to the Teacher's (knowledge-distillation loss) plus the
real flow-matching target (ground-truth loss) -- both objectives combined
(L_total = lambda_gt * L_gt + lambda_kd * L_kd), as in section 13.2/31.

Explicitly NOT done yet (section 32 "초기에는 하지 않는 것"): Teacher-prediction caching
(Teacher runs online every step, per section 30), feature/attention distillation, step
distillation (still uses the full continuous timestep range, no step count reduction),
quantization.

Uses only officially-supported libraries -- PyTorch + Accelerate for the training loop,
plus this repo's own frozen VAE/Qwen3-4B/Flux2 implementations (already offline-cached by
prepare_cache.py, so neither the VAE nor the text encoder is loaded here at all). No
custom distillation library, no diffusers, no PEFT -- see the module docstring in
../docs/flux2_lora_training.md and distillation/README.md for why this is a genuinely
different situation from the LoRA case (full-parameter training of a plain nn.Module
needs no adapter-injection machinery, so the fused-layer-naming problem that blocked
PEFT-based LoRA here just doesn't apply).

Usage (smoke test):
  python train_distill.py --gpu=2 \
    --cache_dir=/data1/jooyonglee/smiledesign_cache_smoke \
    --output_dir=./runs/smoke \
    --max_steps=20 --batch_size=2 --log_every=1
"""

import argparse
import os
import re
import sys
from collections import deque
from pathlib import Path

# This machine's GPUs are shared with other jobs (see README.md) -- --gpu picks which
# physical device (0-7) to train on. Must be handled *before* importing torch/accelerate
# and constructing the Accelerator: CUDA_VISIBLE_DEVICES only takes effect if set before
# anything touches CUDA, so a plain post-parse `os.environ[...] = ...` later would be too
# late. Check `nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu
# --format=csv` first to pick one that's actually free.
_gpu_pre_parser = argparse.ArgumentParser(add_help=False)
_gpu_pre_parser.add_argument("--gpu", type=int, default=None, choices=range(8))
_gpu_pre_args, _ = _gpu_pre_parser.parse_known_args()
if _gpu_pre_args.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_gpu_pre_args.gpu)

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cache_dataset import CachedLatentDataset  # noqa: E402
from student import build_student, build_teacher  # noqa: E402

from flux2.sampling import batched_prc_img, batched_prc_txt, denoise, get_schedule, scatter_ids  # noqa: E402
from flux2.util import load_ae  # noqa: E402

VALIDATION_MODEL_NAME = "flux.2-klein-base-4b"  # only used to load the (frozen) VAE for decoding


def cycle(loader: DataLoader):
    while True:
        yield from loader


_CKPT_STEP_RE = re.compile(r"student_step(\d+)\.pt$")


def find_latest_checkpoint(output_dir: Path) -> Path | None:
    candidates = [p for p in output_dir.glob("student_step*.pt") if _CKPT_STEP_RE.search(p.name)]
    if not candidates:
        return None
    return max(candidates, key=lambda p: int(_CKPT_STEP_RE.search(p.name).group(1)))


def load_checkpoint(path: Path, resume_step_override: int | None) -> tuple[dict, dict | None, int]:
    """Returns (model_state_dict, optimizer_state_dict_or_None, resume_step).

    Current checkpoints intentionally contain only {"model", "step"}: AdamW's two FP32
    moment tensors would roughly triple each checkpoint from 4.2 GB to 12.6 GB. Loading
    the older {"model", "optimizer", "step"} format remains supported. Bare state-dict
    checkpoints are also supported; their step is parsed from "student_step<N>.pt" (or
    must be passed via --resume_step for names such as student_final.pt).
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt and "step" in ckpt:
        return ckpt["model"], ckpt.get("optimizer"), ckpt["step"]

    print(f"  ! {path.name} is an old-format checkpoint (bare state_dict): optimizer state will NOT be restored.")
    match = _CKPT_STEP_RE.search(path.name)
    if match:
        step = int(match.group(1))
    elif resume_step_override is not None:
        step = resume_step_override
    else:
        raise ValueError(
            f"Can't infer the training step {path.name} was saved at (it's not named "
            "student_step<N>.pt). Pass --resume_step=<N> explicitly."
        )
    return ckpt, None, step


@torch.no_grad()
def generate_validation_images(model, ae, samples: list[dict], device, num_steps: int, seed: int) -> torch.Tensor:
    """Run the full multi-step denoise loop (like actual inference) for each cached
    validation sample's text embedding, decode through the VAE, and return a
    (N, 3, H, W) tensor in [0, 1] suitable for TensorBoard's add_images.
    """
    was_training = model.training
    model.eval()
    images = []
    for i, sample in enumerate(samples):
        embedding = sample["embedding"].unsqueeze(0).to(device)
        ctx, ctx_ids = batched_prc_txt(embedding)

        latent_shape = (1, *sample["latent"].shape)
        generator = torch.Generator(device=device).manual_seed(seed + i)
        randn = torch.randn(latent_shape, generator=generator, dtype=ctx.dtype, device=device)
        x, x_ids = batched_prc_img(randn)

        timesteps = get_schedule(num_steps, x.shape[1])
        # guidance is unused: Klein4BParams.use_guidance_embed is False, so the model never
        # reads this value -- denoise() just requires something to build guidance_vec from.
        x = denoise(model, x, x_ids, ctx, ctx_ids, timesteps=timesteps, guidance=1.0)
        x = torch.cat(scatter_ids(x, x_ids)).squeeze(2)

        img = ae.decode(x).float()[0]
        images.append(((img.clamp(-1, 1) + 1) / 2).cpu())

    model.train(was_training)
    return torch.stack(images)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        choices=range(8),
        help="Physical GPU index (0-7) to train on. Already consumed pre-import to set "
        "CUDA_VISIBLE_DEVICES -- redeclared here only so --help/argparse validation see it.",
    )
    parser.add_argument("--cache_dir", required=True, help="Output of prepare_cache.py.")
    parser.add_argument("--output_dir", required=True, help="Where to save Student checkpoints.")
    parser.add_argument("--max_steps", type=int, default=20, help="20 for a smoke test.")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lambda_gt", type=float, default=1.0, help="Weight on the real flow-matching loss.")
    parser.add_argument("--lambda_kd", type=float, default=1.0, help="Weight on the teacher-matching (KD) loss.")
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--save_every", type=int, default=None, help="Default: only save at the end.")
    parser.add_argument(
        "--validation_every_epochs",
        type=int,
        default=1,
        help="Run validation-image generation every N epochs (epoch = len(loader) steps). 0 disables it.",
    )
    parser.add_argument("--num_validation_images", type=int, default=4)
    parser.add_argument(
        "--validation_num_steps",
        type=int,
        default=30,
        help="Denoise steps for validation generation (doc's Phase 1 target: 30-50-step quality).",
    )
    parser.add_argument("--validation_seed", type=int, default=0)
    parser.add_argument(
        "--random_init",
        action="store_true",
        help="Skip teacher-layer-mapping init and start the Student from scratch (random). "
        "Only useful to isolate init quality from everything else; the default "
        "(teacher-initialized) is what section 6 of the distillation doc recommends.",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Resume from a checkpoint: a path to a student_step<N>.pt/student_final.pt file, "
        "or 'auto' to pick the highest-step student_step<N>.pt in --output_dir. Skips "
        "teacher-layer-mapping init (the checkpoint's weights are used instead) and "
        "continues the step counter from where that checkpoint left off.",
    )
    parser.add_argument(
        "--resume_step",
        type=int,
        default=None,
        help="Step to resume from, only needed if --resume points at a checkpoint whose "
        "filename doesn't encode it (e.g. student_final.pt) and isn't the new "
        "dict-format checkpoint (which records its own step).",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    accelerator = Accelerator(mixed_precision="bf16", log_with="tensorboard", project_dir=str(output_dir))
    accelerator.init_trackers("distill", config=vars(args))
    device = accelerator.device

    print(f"[{accelerator.process_index}] Loading cached dataset from {args.cache_dir} ...")
    dataset = CachedLatentDataset(args.cache_dir)
    print(f"{len(dataset)} cached samples")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    print("Loading frozen Teacher (FLUX.2 Klein 4B Base)...")
    teacher = build_teacher(device)

    resume_path = None
    if args.resume == "auto":
        resume_path = find_latest_checkpoint(output_dir)
        if resume_path is None:
            print(f"--resume=auto but no student_step*.pt found in {output_dir}; starting fresh.")
    elif args.resume:
        resume_path = Path(args.resume)

    resume_model_state = resume_optimizer_state = None
    start_step = 0
    if resume_path is not None:
        print(f"Resuming from {resume_path} ...")
        resume_model_state, resume_optimizer_state, start_step = load_checkpoint(resume_path, args.resume_step)
        print(f"  will continue from step {start_step}")

    print("Building Student (~1B) ...")
    # teacher-layer-mapping init is pointless when we're about to overwrite every weight
    # with the resumed checkpoint anyway.
    student = build_student(device, teacher=None if (args.random_init or resume_model_state) else teacher)
    if resume_model_state is not None:
        student.load_state_dict(resume_model_state)

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr)
    student, optimizer, loader = accelerator.prepare(student, optimizer, loader)
    if resume_optimizer_state is not None:
        optimizer.load_state_dict(resume_optimizer_state)
        print("  restored optimizer state from legacy full checkpoint")
    elif resume_path is not None:
        print("  checkpoint has no optimizer state; AdamW starts with fresh moments")

    steps_per_epoch = len(loader)
    validate_every_steps = steps_per_epoch * args.validation_every_epochs if args.validation_every_epochs else None

    val_samples = None
    ae = None
    if validate_every_steps and accelerator.is_main_process:
        val_samples = [dataset[i] for i in range(min(args.num_validation_images, len(dataset)))]
        print(
            f"Validation: {len(val_samples)} images every {args.validation_every_epochs} epoch(s) "
            f"({validate_every_steps} steps), {args.validation_num_steps} denoise steps each. "
            "NOTE: these samples come from the training cache itself (no held-out set yet) -- "
            "treat this as a qualitative sanity check, not a held-out quality metric."
        )
        print("Loading frozen VAE for validation decoding...")
        ae = load_ae(VALIDATION_MODEL_NAME, device=device).eval()

        print("Generating one-time Teacher reference images for comparison...")
        teacher_images = generate_validation_images(
            teacher, ae, val_samples, device, args.validation_num_steps, args.validation_seed
        )
        tracker = next((t for t in accelerator.trackers if t.name == "tensorboard"), None)
        if tracker is not None:
            tracker.writer.add_images("validation/teacher_reference", teacher_images, global_step=0, dataformats="NCHW")
            for i, sample in enumerate(val_samples):
                tracker.writer.add_text(f"validation/prompt_{i}", sample["prompt"], global_step=0)

    # Rolling window for smoothed loss printouts (single-step values at this dataset size
    # are noisy -- e.g. a step that happens to sample t near 0 or 1 has a very different
    # loss scale than one near 0.5 -- so the raw per-step number alone doesn't show
    # whether training is actually progressing).
    window = 20
    recent = {"loss": deque(maxlen=window), "gt": deque(maxlen=window), "kd": deque(maxlen=window)}
    epoch_acc = {"loss": 0.0, "gt": 0.0, "kd": 0.0, "teacher_gt": 0.0, "n": 0}

    step = start_step
    for batch in cycle(loader):
        if step >= args.max_steps:
            break

        x0 = batch["latent"].to(device)  # (B, 128, H/16, W/16)
        text_embedding = batch["embedding"].to(device)  # (B, 512, 7680)
        batch_size = x0.shape[0]

        noise = torch.randn_like(x0)
        t = torch.rand(batch_size, device=device, dtype=x0.dtype)
        t_broadcast = t.view(-1, 1, 1, 1)
        x_t = (1 - t_broadcast) * x0 + t_broadcast * noise
        target_velocity = noise - x0

        x_t_tok, x_ids = batched_prc_img(x_t)
        target_velocity_tok, _ = batched_prc_img(target_velocity)
        ctx, ctx_ids = batched_prc_txt(text_embedding)

        with torch.no_grad():
            teacher_pred = teacher(x=x_t_tok, x_ids=x_ids, timesteps=t, ctx=ctx, ctx_ids=ctx_ids, guidance=None)
            # Reference point, not used in the loss: how far even the (frozen, much
            # bigger) Teacher's own prediction is from the real flow-matching target.
            # The Student's loss_gt approaching this number is the actual sign that
            # distillation is working -- it can't realistically beat the Teacher.
            teacher_gt_loss = F.mse_loss(teacher_pred.float(), target_velocity_tok.float())

        student_pred = student(x=x_t_tok, x_ids=x_ids, timesteps=t, ctx=ctx, ctx_ids=ctx_ids, guidance=None)

        loss_gt = F.mse_loss(student_pred.float(), target_velocity_tok.float())
        loss_kd = F.mse_loss(student_pred.float(), teacher_pred.float())
        loss = args.lambda_gt * loss_gt + args.lambda_kd * loss_kd

        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}: {loss.item()} (gt={loss_gt.item()}, kd={loss_kd.item()})")

        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()

        recent["loss"].append(loss.item())
        recent["gt"].append(loss_gt.item())
        recent["kd"].append(loss_kd.item())
        epoch_acc["loss"] += loss.item()
        epoch_acc["gt"] += loss_gt.item()
        epoch_acc["kd"] += loss_kd.item()
        epoch_acc["teacher_gt"] += teacher_gt_loss.item()
        epoch_acc["n"] += 1

        if step % args.log_every == 0:
            avg_loss = sum(recent["loss"]) / len(recent["loss"])
            avg_gt = sum(recent["gt"]) / len(recent["gt"])
            avg_kd = sum(recent["kd"]) / len(recent["kd"])
            print(
                f"step {step:4d}/{args.max_steps}  loss={loss.item():.4f} (avg{window}={avg_loss:.4f})  "
                f"gt={loss_gt.item():.4f} (avg{window}={avg_gt:.4f})  "
                f"kd={loss_kd.item():.4f} (avg{window}={avg_kd:.4f})  "
                f"teacher_gt={teacher_gt_loss.item():.4f}"
            )
        accelerator.log(
            {
                "train/loss": loss.item(),
                "train/loss_gt": loss_gt.item(),
                "train/loss_kd": loss_kd.item(),
                "train/teacher_gt_loss": teacher_gt_loss.item(),
            },
            step=step,
        )

        if args.save_every and (step + 1) % args.save_every == 0:
            ckpt_path = output_dir / f"student_step{step + 1}.pt"
            # Weights + step only by design. Saving AdamW's FP32 moments adds about 8.4 GB
            # for this 1.05B model; a resumed run therefore starts with fresh moments.
            accelerator.save(
                {
                    "model": accelerator.unwrap_model(student).state_dict(),
                    "step": step + 1,
                },
                ckpt_path,
            )
            print(f"  saved checkpoint -> {ckpt_path} (resume with --resume={ckpt_path})")

        if (step + 1) % steps_per_epoch == 0:
            epoch = (step + 1) // steps_per_epoch
            n = epoch_acc["n"]
            print(
                f"=== epoch {epoch} done ({n} steps): "
                f"avg loss={epoch_acc['loss'] / n:.4f}  avg gt={epoch_acc['gt'] / n:.4f}  "
                f"avg kd={epoch_acc['kd'] / n:.4f}  avg teacher_gt={epoch_acc['teacher_gt'] / n:.4f} ==="
            )
            epoch_acc = {"loss": 0.0, "gt": 0.0, "kd": 0.0, "teacher_gt": 0.0, "n": 0}

        if validate_every_steps and (step + 1) % validate_every_steps == 0 and accelerator.is_main_process:
            epoch = (step + 1) // steps_per_epoch
            print(f"  running validation (epoch {epoch})...")
            student_images = generate_validation_images(
                accelerator.unwrap_model(student), ae, val_samples, device, args.validation_num_steps,
                args.validation_seed,
            )
            tracker = next((t for t in accelerator.trackers if t.name == "tensorboard"), None)
            if tracker is not None:
                tracker.writer.add_images(
                    "validation/student", student_images, global_step=step + 1, dataformats="NCHW"
                )

        step += 1

    final_ckpt = output_dir / "student_final.pt"
    accelerator.save(
        {"model": accelerator.unwrap_model(student).state_dict(), "step": step},
        final_ckpt,
    )
    print(f"\nDone: {step} steps. Final checkpoint -> {final_ckpt}")
    accelerator.end_training()


if __name__ == "__main__":
    main()
