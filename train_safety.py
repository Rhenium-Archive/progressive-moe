"""
안전수칙 위반 AI 식별 모델 학습 스크립트

SAFETY_AI_ROADMAP.md Phase A 구현:
  Phase 1: Master 모델 — 안전 도메인 언어 패턴 학습 (SIF 재해개요 + 법령 텍스트)
  Phase 2: MoE 모델   — 위반유형별 Expert 특화 (3단계 학습)

사용법:
  # Phase 1만 (Master 학습)
  python train_safety.py --phase 1

  # Phase 2만 (MoE 학습, Phase 1 체크포인트 필요)
  python train_safety.py --phase 2

  # Phase 1 + 2 연속
  python train_safety.py --phase all
"""

import argparse
import json
import os
import gc
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from model import ModelConfig, MoEBlock, precompute_freqs_cis, RMSNorm, convert_legacy_state_dict
from train_master import (
    MasterModel,
    train_master_model,
    build_qlora_master_model,
    _qlora_to_full_state_dict,
    load_master_for_moe,
)
from train_moe import MoEModel, convert_master_to_moe, train_moe_stage
from utils import TextDataset, KoreanBPETokenizer, get_gpu_memory, print_memory_stats


# ==========================================
# 1. 안전 도메인 데이터 로더
# ==========================================

def load_sif_data(sif_path: str = "data/processed/sif_train.json") -> list[dict]:
    """SIF 학습 데이터 로드."""
    with open(sif_path, encoding="utf-8") as f:
        data = json.load(f)
    print(f"   -> SIF 데이터: {len(data)}건")
    return data


def load_law_texts(law_path: str = "data/law_texts/all_laws.json") -> list[dict]:
    """안전보건법령 텍스트 로드."""
    with open(law_path, encoding="utf-8") as f:
        data = json.load(f)
    print(f"   -> 법령 조문: {len(data)}건")
    return data


def build_phase1_texts(sif_data: list[dict], law_data: list[dict]) -> list[str]:
    """Phase 1 학습용 텍스트 구성.

    안전 도메인 언어 패턴 학습을 위해:
    - SIF 재해개요 (사고 경위 서술)
    - SIF 재해유발요인 + 위험성감소대책 (위반-대책 쌍)
    - 안전보건법령 조문 (법적 기준)
    """
    texts = []

    # SIF 재해개요 — 사고 서술 패턴 학습
    for r in sif_data:
        texts.append(r["input"])

    # SIF 위반-대책 쌍 — 구조화된 안전 지식
    for r in sif_data:
        cause = r.get("cause", "")
        countermeasure = r.get("countermeasure", "")
        if cause and countermeasure:
            pair_text = f"[위반사항] {cause}\n[대책] {countermeasure}"
            texts.append(pair_text)

    # 법령 조문 — 법적 기준 텍스트
    for art in law_data:
        texts.append(art["text"])

    random.shuffle(texts)
    print(f"   -> Phase 1 학습 텍스트: {len(texts)}건")
    return texts


def build_phase2_texts(sif_data: list[dict]) -> list[str]:
    """Phase 2 학습용 텍스트 구성.

    위반유형 라벨을 포함한 구조화된 텍스트:
    입력 → 위반유형/사고유형/위험등급/원인/대책 형태로 모델이 학습
    """
    texts = []

    for r in sif_data:
        # 위반유형 라벨이 있는 건만 사용 (미분류 제외)
        if r.get("violation_type", "미분류") == "미분류":
            continue

        # 구조화된 학습 텍스트 구성
        parts = [
            f"[재해개요] {r['input']}",
            f"[위반유형] {r['violation_type']}",
            f"[세부유형] {r['sub_type']}",
            f"[사고유형] {r['accident_type']}",
            f"[위험등급] {r['risk_level']}",
            f"[기인물] {r['equipment']}",
        ]

        cause = r.get("cause", "")
        if cause:
            parts.append(f"[원인] {cause}")

        countermeasure = r.get("countermeasure", "")
        if countermeasure:
            parts.append(f"[대책] {countermeasure}")

        texts.append("\n".join(parts))

    random.shuffle(texts)
    print(f"   -> Phase 2 학습 텍스트: {len(texts)}건 (미분류 제외)")
    return texts


# ==========================================
# 2. Phase 1: Master 모델 학습
# ==========================================

def run_phase1(args):
    """Phase 1: 안전 도메인 Master 모델 학습."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print("Phase 1: 안전 도메인 Master 모델 학습")
    print(f"{'='*60}")
    print(f"디바이스: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

    save_dir = args.save_dir or "safety_checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    # 모델 설정
    use_qlora = args.qlora
    if use_qlora:
        config = ModelConfig(
            vocab_size=32000,
            dim=2560,
            n_layers=32,
            n_heads=20,
            head_dim=128,
            max_seq_len=2048,
            num_experts=1,
            num_experts_per_tok=1,
            dropout=0.1,
        )
        batch_size = 4
        gradient_accumulation_steps = 2
        print(f"모델: QLoRA (dim={config.dim}, layers={config.n_layers})")
    else:
        config = ModelConfig(
            vocab_size=32000,
            dim=2304,
            n_layers=26,
            n_heads=18,
            head_dim=128,
            max_seq_len=1920,
            num_experts=1,
            num_experts_per_tok=1,
            dropout=0.1,
        )
        batch_size = 2
        gradient_accumulation_steps = 2
        print(f"모델: Full FT (dim={config.dim}, layers={config.n_layers})")

    # 데이터 로드
    print("\n>>> 안전 도메인 데이터 로드 중...")
    sif_data = load_sif_data()
    law_data = load_law_texts()
    texts = build_phase1_texts(sif_data, law_data)
    del sif_data, law_data
    gc.collect()

    # 토크나이저
    print("\n>>> 토크나이저 구축 중...")
    tokenizer = KoreanBPETokenizer(vocab_size=config.vocab_size)
    # 안전 도메인 텍스트로 어휘 구축 (법령 + SIF 혼합)
    vocab_sample = texts[:min(5000, len(texts))]
    tokenizer.build_vocab(vocab_sample)
    del vocab_sample

    # 데이터셋
    dataset = TextDataset(texts, tokenizer, max_length=config.max_seq_len)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
    )
    print(f"배치 크기: {batch_size}, GA: {gradient_accumulation_steps}")
    print(f"배치 수: {len(dataloader)}")
    del texts

    # 모델 생성
    print("\n>>> Master 모델 생성 중...")
    if use_qlora:
        try:
            model = build_qlora_master_model(config, device=device)
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model.parameters())
            print(f"QLoRA 적용: {trainable/1e6:.1f}M / {total/1e6:.1f}M 학습 가능")
        except Exception as e:
            print(f"QLoRA 실패: {e}")
            print("Full FT로 전환...")
            use_qlora = False
            model = MasterModel(config, use_flash_attention=False).to(device)
    else:
        model = MasterModel(config, use_flash_attention=False).to(device)

    # Gradient checkpointing
    base = model.base_model.model if hasattr(model, "base_model") else model
    for layer in getattr(base, "layers", []):
        if hasattr(layer, "use_gradient_checkpointing"):
            layer.use_gradient_checkpointing = True

    # 옵티마이저
    lr = 2e-4 if use_qlora else 3e-4
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(
            model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95),
        )
    except ImportError:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95),
        )

    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # 학습
    losses = train_master_model(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        num_epochs=args.epochs,
        save_dir=save_dir,
        save_every_n_epochs=1,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )

    print_memory_stats("Phase 1 완료")

    # 저장
    if use_qlora:
        try:
            full_state = _qlora_to_full_state_dict(model, device="cpu")
            moe_path = os.path.join(save_dir, "safety_master_for_moe.pt")
            torch.save({"model_state_dict": full_state, "config": config}, moe_path)
            print(f"MoE용 체크포인트 저장: {moe_path}")
        except Exception as e:
            print(f"MoE용 저장 실패: {e}")
    else:
        final_path = os.path.join(save_dir, "safety_master_for_moe.pt")
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": config,
        }, final_path)
        print(f"Master 체크포인트 저장: {final_path}")

    # 토크나이저 저장
    tokenizer.save(os.path.join(save_dir, "tokenizer.json"))
    print(f"토크나이저 저장: {save_dir}/tokenizer.json")

    print(f"\n Phase 1 완료! 평균 Loss: {sum(losses)/len(losses):.4f}")
    return losses


# ==========================================
# 3. Phase 2: MoE 모델 학습
# ==========================================

def run_phase2(args):
    """Phase 2: 안전 도메인 MoE 모델 학습 (위반유형별 Expert 특화)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print("Phase 2: 안전 도메인 MoE 모델 학습")
    print(f"{'='*60}")
    print(f"디바이스: {device}")

    save_dir = args.save_dir or "safety_checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    # Master 모델 로드
    master_path = os.path.join(save_dir, "safety_master_for_moe.pt")
    epoch_path = os.path.join(save_dir, "master_model_epoch_1.pt")

    if os.path.exists(master_path):
        print(f"\n>>> Master 모델 로드: {master_path}")
        ckpt = torch.load(master_path, map_location="cpu", weights_only=False)
        master_model = MasterModel(ckpt["config"], use_flash_attention=False)
        state = {k: v for k, v in ckpt["model_state_dict"].items()
                 if "lora_" not in k and "base_layer" not in k}
        state = convert_legacy_state_dict(state)
        master_model.load_state_dict(state, strict=True)
        del ckpt, state
        gc.collect()
    elif os.path.exists(epoch_path):
        print(f"\n>>> Master 모델 로드: {epoch_path}")
        master_model = load_master_for_moe(epoch_path, device="cpu")
    else:
        print(f"Master 체크포인트 없음: {master_path}")
        print("Phase 1을 먼저 실행하세요: python train_safety.py --phase 1")
        return
    print("Master 모델 로드 완료")

    # MoE 변환
    num_experts = args.num_experts
    print(f"\n>>> MoE 변환 (Expert {num_experts}개, copy_with_noise)...")
    moe_model = convert_master_to_moe(
        master_model,
        num_experts=num_experts,
        expert_init_method="copy_with_noise",
        new_expert_noise=0.01,
    )
    del master_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # bf16 + GPU
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    moe_model = moe_model.to(device=device, dtype=compute_dtype)
    print(f"모델 dtype: {compute_dtype}")

    # Gradient checkpointing
    for layer in moe_model.layers:
        if hasattr(layer, "use_gradient_checkpointing"):
            layer.use_gradient_checkpointing = True
    moe_model.print_param_stats()

    # 데이터 로드 (Phase 2: 위반유형 라벨 포함)
    print("\n>>> 안전 도메인 데이터 로드 중...")
    sif_data = load_sif_data()
    texts = build_phase2_texts(sif_data)
    del sif_data
    gc.collect()

    # 토크나이저 — Phase 1과 동일한 어휘 사전 사용
    print("\n>>> 토크나이저 로드 중...")
    tokenizer_path = os.path.join(save_dir, "tokenizer.json")
    if os.path.exists(tokenizer_path):
        tokenizer = KoreanBPETokenizer.load(tokenizer_path)
        print(f"Phase 1 토크나이저 로드: {tokenizer_path}")
    else:
        print(f"경고: {tokenizer_path} 없음 — 새 어휘 구축")
        tokenizer = KoreanBPETokenizer(vocab_size=moe_model.config.vocab_size)
        tokenizer.build_vocab(texts[:min(5000, len(texts))])

    # 데이터로더
    dataset = TextDataset(texts, tokenizer, max_length=moe_model.config.max_seq_len)
    batch_size = args.batch_size
    gradient_accumulation_steps = 4
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
    )
    print(f"배치 크기: {batch_size}, GA: {gradient_accumulation_steps}")
    print(f"배치 수: {len(dataloader)}")
    del texts

    # 3단계 학습
    moe_save_dir = os.path.join(save_dir, "moe")
    os.makedirs(moe_save_dir, exist_ok=True)

    # Stage 1: Router만 학습 — 위반유형별 라우팅 학습
    stage1_ckpt = os.path.join(moe_save_dir, "moe_stage1_router.pt")
    if os.path.exists(stage1_ckpt):
        print(f"\n>>> Stage 1 체크포인트 발견, 로드 중: {stage1_ckpt}")
        ckpt = torch.load(stage1_ckpt, map_location="cpu", weights_only=False)
        moe_model.load_state_dict(ckpt["model_state_dict"], strict=False)
        moe_model = moe_model.to(device=device, dtype=compute_dtype)
        del ckpt
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        stage1_losses = None
    else:
        print("\n>>> Stage 1: Router 학습 (위반유형별 라우팅)")
        stage1_losses, _ = train_moe_stage(
            stage_name="Stage1_Router",
            moe_model=moe_model,
            dataloader=dataloader,
            device=device,
            num_epochs=args.epochs,
            learning_rate=1e-4,
            train_router_only=True,
            save_dir=moe_save_dir,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )

    # Stage 2: Expert만 학습 — 각 Expert가 특정 위반유형에 특화
    print("\n>>> Stage 2: Expert 학습 (위반유형별 특화)")
    stage2_losses, _ = train_moe_stage(
        stage_name="Stage2_Expert",
        moe_model=moe_model,
        dataloader=dataloader,
        device=device,
        num_epochs=args.epochs,
        learning_rate=2e-4,
        train_experts_only=True,
        save_dir=moe_save_dir,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )

    # Stage 3: Router + Expert 전체 최적화
    print("\n>>> Stage 3: 전체 미세조정 (Router + Expert)")
    stage3_losses, _ = train_moe_stage(
        stage_name="Stage3_Full",
        moe_model=moe_model,
        dataloader=dataloader,
        device=device,
        num_epochs=args.epochs,
        learning_rate=1e-4,
        train_router_only=False,
        train_experts_only=False,
        save_dir=moe_save_dir,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )

    # 최종 모델 저장
    final_path = os.path.join(save_dir, "safety_moe_final.pt")
    torch.save({
        "model_state_dict": moe_model.state_dict(),
        "config": moe_model.config,
        "stage1_losses": stage1_losses or [],
        "stage2_losses": stage2_losses,
        "stage3_losses": stage3_losses,
        "num_experts": num_experts,
    }, final_path)
    tokenizer.save(os.path.join(save_dir, "tokenizer.json"))

    # 결과 출력
    print(f"\n{'='*60}")
    print("Phase 2 완료!")
    print(f"{'='*60}")
    if stage1_losses:
        print(f"  Stage 1 (Router): {sum(stage1_losses)/len(stage1_losses):.4f}")
    else:
        print(f"  Stage 1 (Router): 체크포인트 로드")
    print(f"  Stage 2 (Expert): {sum(stage2_losses)/len(stage2_losses):.4f}")
    print(f"  Stage 3 (Full):   {sum(stage3_losses)/len(stage3_losses):.4f}")
    print(f"\n최종 모델: {final_path}")


# ==========================================
# 4. 메인
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="안전수칙 위반 AI 식별 모델 학습")
    parser.add_argument(
        "--phase", type=str, default="all", choices=["1", "2", "all"],
        help="학습 단계: 1=Master, 2=MoE, all=전체 (기본: all)",
    )
    parser.add_argument("--epochs", type=int, default=1, help="에폭 수 (기본: 1)")
    parser.add_argument("--qlora", action="store_true", help="QLoRA 사용 (Phase 1)")
    parser.add_argument("--num-experts", type=int, default=2, help="Expert 수 (기본: 2)")
    parser.add_argument("--batch-size", type=int, default=1, help="배치 크기 (기본: 1)")
    parser.add_argument("--save-dir", type=str, default="safety_checkpoints", help="체크포인트 저장 경로")
    args = parser.parse_args()

    print(">>> 안전수칙 위반 AI 식별 모델 학습")
    print(f"   Phase: {args.phase}")
    print(f"   Epochs: {args.epochs}")
    print(f"   QLoRA: {args.qlora}")
    print(f"   Experts: {args.num_experts}")
    print(f"   저장 경로: {args.save_dir}")

    if args.phase in ("1", "all"):
        run_phase1(args)

    if args.phase in ("2", "all"):
        run_phase2(args)

    print("\n>>> 학습 완료!")


if __name__ == "__main__":
    main()
