# 단일 GPU 학습
GPU_IDS=3 ./lora/run_train.sh t2i /data/my_t2i

# 다중 GPU 학습
GPU_IDS=2,3 ./lora/run_train.sh edit /data/my_edit

# GPU 3에서 추론
GPU_ID=3 ./lora/inference.sh t2i ./lora/output/t2i-lora \
  "TOK. A cinematic portrait" ./output/result.png



python ./lora/inference.py --mode t2i \
--lora "lora/output/t2i-lora/checkpoint-2000" \
--prompt "mysexy, 1girl, high heels, nipples, breasts, realistic, nude, photorealistic, solo, on back, lying, blonde hair, looking at viewer, medium breasts, long hair, strappy heels, black footwear, on floor, shoes, blue eyes, hand on own head, (not worst quality, not low quality:1.4), not poorly drawn, no bad anatomy, not wrong anatomy,not  deformed, not disfigured, not ugly" \
--output "./output/lora_result2" \
--dtype bf16 \
--num-images 30 \
--device cuda:3






## trouble shoot
## weights server.py 같은 경우는 
# bench_klein4b.py를 통해 필요한 파일만 개별 다운로드합니다.- flux-2-klein-4b.safetensors
# - FLUX.2-dev/ae.safetensors
# - 별도 Qwen3-4B-FP8

# lora/inference.py 관련 주석
#  [lora/inference.py (line 90)](/workspace/repo/flux2/lora/inference.py:90)는 아래처럼 전체 Diffusers 파이프라인을 요청합니다.