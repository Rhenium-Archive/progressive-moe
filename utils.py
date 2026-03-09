"""
공용 유틸리티 모듈
- TextDataset: 텍스트 → 토큰 Dataset
- KoreanBPETokenizer: 한국어 BPE 토크나이저
- GPU 메모리 모니터링
"""

import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors, decoders


# ==========================================
# GPU 메모리 유틸리티
# ==========================================
def get_gpu_memory():
    """현재 GPU 메모리 사용량 반환 (dict 또는 None)"""
    if not torch.cuda.is_available():
        return None
    return {
        'allocated': torch.cuda.memory_allocated() / 1e9,
        'reserved': torch.cuda.memory_reserved() / 1e9,
        'max_allocated': torch.cuda.max_memory_allocated() / 1e9,
    }


def print_memory_stats(tag: str = ""):
    """GPU 메모리 상태 출력"""
    mem = get_gpu_memory()
    if mem is None:
        print(f"   [{tag}] GPU 없음")
        return
    print(f"   [{tag}] GPU 메모리 — "
          f"할당: {mem['allocated']:.2f}GB, "
          f"예약: {mem['reserved']:.2f}GB, "
          f"최대: {mem['max_allocated']:.2f}GB")


# ==========================================
# 1. 데이터셋 클래스
# ==========================================
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=2048):
        self.max_length = max_length
        pad_id = tokenizer.pad_id if hasattr(tokenizer, 'pad_id') else 0

        # 사전 토큰화: __getitem__마다 encode 호출하지 않고 초기화 시 한 번만 수행
        # int32 텐서로 저장하여 Python list 대비 ~7배 메모리 절감
        print(f"   -> TextDataset: {len(texts)}개 텍스트 사전 토큰화 중...")
        all_tokens = []
        for text in texts:
            tokens = tokenizer.encode(text)
            if len(tokens) > max_length:
                tokens = tokens[:max_length]
            else:
                tokens = tokens + [pad_id] * (max_length - len(tokens))
            all_tokens.append(tokens)
        self.token_ids = torch.tensor(all_tokens, dtype=torch.int32)
        del all_tokens
        print(f"   -> TextDataset: 사전 토큰화 완료 ({len(self.token_ids)}개, {self.token_ids.element_size() * self.token_ids.nelement() / 1e9:.2f}GB)")

    def __len__(self):
        return len(self.token_ids)

    def __getitem__(self, idx):
        tokens = self.token_ids[idx]
        # 언어 모델 학습용: input은 마지막 토큰 제외, label은 첫 토큰 제외
        return tokens[:-1].long(), tokens[1:].long()

# ==========================================
# 2. SentencePiece BPE 토크나이저 (한국어 최적화)
# ==========================================
class KoreanBPETokenizer:
    """SentencePiece BPE 기반 한국어 토크나이저"""
    def __init__(self, vocab_size=50000):
        self.vocab_size = vocab_size
        self.tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))

        # ByteLevel pre-tokenizer: 한국어 문자를 바이트 단위로 처리
        self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        self.tokenizer.decoder = decoders.ByteLevel()

        # 패딩 설정
        self.tokenizer.enable_padding(
            pad_id=0,
            pad_token="<pad>",
            length=None  # 동적 패딩
        )

        # 특수 토큰 ID 저장용
        self.unk_id = None
        self.pad_id = 0
        self.eos_id = None
        self.bos_id = None

    def build_vocab(self, texts):
        """텍스트로부터 BPE 어휘 사전 구축"""
        print(">>> SentencePiece BPE 어휘 사전 구축 중...")
        print(f"   -> 텍스트 수: {len(texts)}개")

        # BPE 트레이너 설정
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=2,  # 최소 빈도 (한국어 특성 고려)
            special_tokens=["<unk>", "<s>", "</s>", "<pad>"],
            show_progress=True,
            continuing_subword_prefix="",
        )

        # 텍스트 이터레이터로 학습
        def text_iterator():
            for text in texts:
                if text and isinstance(text, str):
                    yield text

        # BPE 학습
        self.tokenizer.train_from_iterator(
            text_iterator(),
            trainer=trainer,
            length=len(texts)
        )

        # 후처리 설정 (BOS/EOS 추가)
        self.tokenizer.post_processor = processors.TemplateProcessing(
            single="<s> $A </s>",
            special_tokens=[
                ("<s>", self.tokenizer.token_to_id("<s>")),
                ("</s>", self.tokenizer.token_to_id("</s>")),
            ],
        )

        # 특수 토큰 ID 저장
        self.unk_id = self.tokenizer.token_to_id("<unk>")
        self.pad_id = self.tokenizer.token_to_id("<pad>")
        self.eos_id = self.tokenizer.token_to_id("</s>")
        self.bos_id = self.tokenizer.token_to_id("<s>")

        vocab_size_actual = self.tokenizer.get_vocab_size()
        print(f"   -> 어휘 사전 크기: {vocab_size_actual}")
        print(f"   -> 특수 토큰: <unk>={self.unk_id}, <pad>={self.pad_id}, <s>={self.bos_id}, </s>={self.eos_id}")

    def encode(self, text):
        """텍스트를 토큰 ID 리스트로 변환"""
        if not isinstance(text, str):
            text = str(text)
        encoding = self.tokenizer.encode(text)
        return encoding.ids

    def decode(self, token_ids):
        """토큰 ID 리스트를 텍스트로 변환"""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.cpu().tolist()
        elif not isinstance(token_ids, list):
            token_ids = list(token_ids)

        # 특수 토큰 제거
        filtered_ids = [
            tid for tid in token_ids
            if tid not in [self.pad_id, self.unk_id, self.eos_id, self.bos_id]
        ]

        try:
            return self.tokenizer.decode(filtered_ids)
        except Exception:
            return self.tokenizer.decode(token_ids)

    def get_vocab_size(self):
        """어휘 사전 크기 반환"""
        return self.tokenizer.get_vocab_size()

    def save(self, path):
        """토크나이저 저장"""
        self.tokenizer.save(path)
        print(f"   -> 토크나이저 저장: {path}")

    @classmethod
    def load(cls, path):
        """저장된 토크나이저 로드 (classmethod)"""
        instance = cls.__new__(cls)
        instance.tokenizer = Tokenizer.from_file(path)
        instance.vocab_size = instance.tokenizer.get_vocab_size()
        instance.unk_id = instance.tokenizer.token_to_id("<unk>")
        instance.pad_id = instance.tokenizer.token_to_id("<pad>") or 0
        instance.eos_id = instance.tokenizer.token_to_id("</s>")
        instance.bos_id = instance.tokenizer.token_to_id("<s>")
        instance.tokenizer.enable_padding(
            pad_id=instance.pad_id, pad_token="<pad>", length=None
        )
        return instance
