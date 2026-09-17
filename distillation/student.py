"""Teacher/Student construction for Phase 1 capacity distillation.

Per docs/flux2_distillation_05b_1b.md section 5/6: for the first 1B experiment, keep the
Student's hidden width, head count, and I/O interface IDENTICAL to the Teacher's and only
reduce depth (number of transformer blocks). This means every submodule except
double_blocks/single_blocks has an *exact* shape match between Teacher and Student, so
they can be copied 1:1 (not just "layer mapped") -- only the transformer blocks
themselves need the sparser teacher-block-subsampling init described in section 6-B.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flux2.model import Flux2, Klein4BParams  # noqa: E402
from flux2.util import load_flow_model  # noqa: E402

TEACHER_MODEL_NAME = "flux.2-klein-base-4b"

# depth=2, depth_single_blocks=3 -> ~1.05B params (see distillation/README.md for how this
# was chosen) while keeping hidden_size/num_heads/context_in_dim identical to the teacher.
STUDENT_PARAMS = Klein4BParams(depth=2, depth_single_blocks=3)


def build_teacher(device: torch.device) -> Flux2:
    teacher = load_flow_model(TEACHER_MODEL_NAME, device=device)
    teacher.eval().requires_grad_(False)
    return teacher


def build_student(device: torch.device, teacher: Flux2 | None = None) -> Flux2:
    """Build the Student. If `teacher` is given, initialize shared-shape submodules and a
    subsampled selection of transformer blocks from it (recommended); otherwise the
    Student is randomly initialized (only useful for isolating "does the training loop
    run at all" from "does the initialization matter").
    """
    with torch.device("meta"):
        student = Flux2(STUDENT_PARAMS)
    student = student.to_empty(device=device)
    for p in student.parameters():
        torch.nn.init.normal_(p, mean=0.0, std=0.02)

    if teacher is not None:
        _init_from_teacher(student, teacher)

    return student


def _evenly_spaced_indices(n_teacher: int, n_student: int) -> list[int]:
    if n_student == 1:
        return [0]
    return [round(i * (n_teacher - 1) / (n_student - 1)) for i in range(n_student)]


def _init_from_teacher(student: Flux2, teacher: Flux2) -> None:
    # Everything except the transformer blocks has an identical shape (same hidden_size,
    # num_heads, context_in_dim, in_channels) -- copy it wholesale.
    shared_modules = [
        "pe_embedder",
        "img_in",
        "time_in",
        "txt_in",
        "double_stream_modulation_img",
        "double_stream_modulation_txt",
        "single_stream_modulation",
        "final_layer",
    ]
    if student.use_guidance_embed and teacher.use_guidance_embed:
        shared_modules.append("guidance_in")
    for name in shared_modules:
        getattr(student, name).load_state_dict(getattr(teacher, name).state_dict())

    double_idx = _evenly_spaced_indices(len(teacher.double_blocks), len(student.double_blocks))
    for student_i, teacher_i in enumerate(double_idx):
        student.double_blocks[student_i].load_state_dict(teacher.double_blocks[teacher_i].state_dict())

    single_idx = _evenly_spaced_indices(len(teacher.single_blocks), len(student.single_blocks))
    for student_i, teacher_i in enumerate(single_idx):
        student.single_blocks[student_i].load_state_dict(teacher.single_blocks[teacher_i].state_dict())

    print(
        f"Initialized student from teacher: double_blocks {double_idx}, single_blocks {single_idx} "
        f"(out of {len(teacher.double_blocks)}/{len(teacher.single_blocks)} teacher blocks)"
    )


if __name__ == "__main__":
    # Quick standalone check: param counts + that layer-mapping init actually runs.
    with torch.device("meta"):
        teacher_meta = Flux2(Klein4BParams())
        student_meta = Flux2(STUDENT_PARAMS)
    n_teacher = sum(p.numel() for p in teacher_meta.parameters())
    n_student = sum(p.numel() for p in student_meta.parameters())
    print(f"Teacher: {n_teacher:,} ({n_teacher / 1e9:.3f}B)")
    print(f"Student: {n_student:,} ({n_student / 1e9:.3f}B)")
