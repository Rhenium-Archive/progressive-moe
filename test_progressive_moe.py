#!/usr/bin/env python3
"""
Progressive MoE 간단한 테스트 스크립트
"""
import sys
import os

# 현재 디렉토리를 Python path에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from progressive_moe_layer import ProgressiveMoELayer, ProgressiveMoEConfig
    from model import ModelConfig
    import torch

    print("=== Progressive MoE 테스트 ===")

    # 간단한 테스트 설정
    config = ModelConfig(
        vocab_size=1000,
        dim=256,
        n_layers=2,
        n_heads=4,
        head_dim=64,
        num_experts=2,
        num_experts_per_tok=2,
    )

    progressive_config = ProgressiveMoEConfig(
        base_config=config,
        initial_num_experts=2,
        max_num_experts=6,
        experts_per_stage=2,
    )

    print(f"설정: dim={config.dim}, 초기 experts={progressive_config.initial_num_experts}")

    # Progressive MoE 레이어 생성
    moe_layer = ProgressiveMoELayer(config, progressive_config)
    print(f"✅ MoE 레이어 생성 완료")
    print(f"   -> 초기 experts: {moe_layer.current_num_experts}")

    # [P-2] Router 초기화 충돌 검증
    assert moe_layer.router.current_num_experts == progressive_config.initial_num_experts, \
        f"Router current_num_experts 초기화 불일치: {moe_layer.router.current_num_experts} != {progressive_config.initial_num_experts}"
    print(f"   -> ✅ Router 초기화 정상 (current={moe_layer.router.current_num_experts})")

    # 테스트 입력
    batch_size, seq_len = 1, 16
    x = torch.randn(batch_size, seq_len, config.dim)
    print(f"   -> 테스트 입력: {x.shape}")

    # Forward pass 테스트
    output1, aux_loss1 = moe_layer(x)
    assert aux_loss1.item() >= 0.0, f"aux_loss가 음수: {aux_loss1.item()}"
    assert moe_layer.routing_counts.sum() > 0, "routing_counts가 0 (라우팅 카운트 미누적)"
    print(f"   -> 초기 출력: {output1.shape}, aux_loss: {aux_loss1.item():.6f}")

    # Expert 추가 테스트
    print("\n>>> Expert 추가 테스트")
    moe_layer.add_experts(2, init_method="random")

    # 확장 후 Forward pass
    output2, aux_loss2 = moe_layer(x)
    print(f"   -> 확장 후 출력: {output2.shape}, aux_loss: {aux_loss2.item():.6f}")
    print(f"   -> 최종 experts: {moe_layer.current_num_experts}")

    # Expert 사용량 통계
    usage_stats = moe_layer.get_expert_usage_stats()
    print(f"   -> Expert 사용량: {usage_stats}")

    # Expert 동결/해제 검증
    moe_layer.freeze_experts([0])
    assert not moe_layer.experts[0].w1.weight.requires_grad, "freeze_experts 실패"
    moe_layer.unfreeze_experts([0])
    assert moe_layer.experts[0].w1.weight.requires_grad, "unfreeze_experts 실패"
    print("   -> ✅ freeze/unfreeze 동작 확인")

    # aux_loss gradient 흐름 확인
    output, aux_loss = moe_layer(x)
    loss = output.sum() + 0.01 * aux_loss
    loss.backward()
    for p in moe_layer.router.parameters():
        assert p.grad is not None, "Router gradient 없음"
    print("   -> ✅ gradient 흐름 확인")

    # max_num_experts 초과 방지 (별도 레이어로 테스트)
    small_config = ProgressiveMoEConfig(
        base_config=config,
        initial_num_experts=2,
        max_num_experts=4,
        experts_per_stage=2,
    )
    small_layer = ProgressiveMoELayer(config, small_config)
    try:
        small_layer.add_experts(999)
        assert False, "ValueError 미발생"
    except ValueError:
        print("   -> ✅ max_num_experts 초과 방지 확인")

    print("\n✅ 모든 테스트 통과!")

except ImportError as e:
    print(f"❌ Import 에러: {e}")
    print("필요한 모듈을 찾을 수 없습니다.")

except Exception as e:
    print(f"❌ 실행 에러: {e}")
    import traceback
    traceback.print_exc()

print("\n=== 테스트 완료 ===")
