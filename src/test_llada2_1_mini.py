"""
LLaDA2.1-mini 最小验证脚本：加载模型并做一次简单生成。
模型权重缓存目录已设置为 /etc/moreh/checkpoint/huggingface_hub（需在 import 前生效）。
"""

import os

# 将 Hugging Face 模型下载缓存放到数据盘，避免占满 home
os.environ["HUGGINGFACE_HUB_CACHE"] = "/etc/moreh/checkpoint/huggingface_hub"


# 修复 KeyError: 'default' —— transformers 的 ROPE_INIT_FUNCTIONS 无 "default" 键，
# 而 LLaDA2 在无 rope_scaling 时会把 rope_type 设为 "default"，需在 import 模型前注册。
def _compute_default_rope_parameters(
    config, device=None, seq_len=None, layer_type=None
):
    """标准 RoPE（无 scaling），与 transformers 其它 ROPE 初始化函数签名一致。"""
    import torch

    # 始终在 CPU 上创建 RoPE 张量，避免 lazy 加载时 device 为 meta 导致 "Cannot copy out of meta tensor"
    base = getattr(config, "rope_theta", 10000.0)
    head_dim = getattr(config, "head_dim", None) or (
        config.hidden_size // config.num_attention_heads
    )
    partial = getattr(config, "partial_rotary_factor", 1.0)
    dim = int(head_dim * partial)
    inv_freq = 1.0 / (
        base ** (torch.arange(0, dim, 2, dtype=torch.int64, device="cpu").float() / dim)
    )
    return inv_freq, 1.0


from transformers import modeling_rope_utils

if "default" not in getattr(modeling_rope_utils, "ROPE_INIT_FUNCTIONS", {}):
    modeling_rope_utils.ROPE_INIT_FUNCTIONS["default"] = (
        _compute_default_rope_parameters
    )

# 日志：在 dlm/log 下按「时间_脚本名」写日志，tmux 里跑时也方便事后查看
import sys
from datetime import datetime

_script_name = os.path.splitext(os.path.basename(__file__))[0]
_log_dir = os.path.join(os.path.dirname(__file__), "..", "log")
os.makedirs(_log_dir, exist_ok=True)
_log_path = os.path.join(
    _log_dir,
    f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{_script_name}.log",
)
_log_file = open(_log_path, "w", encoding="utf-8")


class _Tee:
    def __init__(self, stream, file):
        self._stream = stream
        self._file = file

    def write(self, data):
        self._stream.write(data)
        self._file.write(data)
        self._file.flush()

    def flush(self):
        self._stream.flush()
        self._file.flush()


sys.stdout = _Tee(sys.__stdout__, _log_file)
sys.stderr = _Tee(sys.__stderr__, _log_file)
print(f"[log] 输出已同时写入: {os.path.abspath(_log_path)}")

import threading
import time

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# 模型 ID（在线加载）或本地路径
model_id = "inclusionAI/LLaDA2.1-mini"

# 单设备加载；设置环境变量 DLM_CPU=1 可强制用 CPU（便于对照 GPU 是否导致只出 EOS）
device = (
    "cpu"
    if os.environ.get("DLM_CPU")
    else ("cuda" if torch.cuda.is_available() else "cpu")
)
print(f"[device] 使用: {device}")

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    device_map=None,
)
model = model.to(torch.bfloat16).to(device)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

prompt = """Calculate 1+5-28*0.5-200=?"""
# prompt = """How do you feel about the future of AI?"""
out = tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt}],
    add_generation_prompt=True,
    tokenize=True,
    return_tensors="pt",
)
# 确保是 LongTensor，且为 (batch_size, seq_len)；LLaDA2 generate(inputs=...) 要求张量
# apply_chat_template 可能返回 Tensor / dict / BatchEncoding，统一按「可下标取 input_ids」处理
if isinstance(out, torch.Tensor):
    input_ids = out
else:
    input_ids = out["input_ids"]
input_ids = (
    input_ids
    if isinstance(input_ids, torch.Tensor)
    else torch.tensor(input_ids, dtype=torch.long)
)
if input_ids.dim() == 1:
    input_ids = input_ids.unsqueeze(0)
input_ids = input_ids.long().to(device)

# generate 无内置进度，用后台线程 + tqdm 显示「进行中」，避免长时间无输出
generated_tokens = [None]


def _run_generate():
    with torch.no_grad():
        generated_tokens[0] = model.generate(
            inputs=input_ids,
            eos_early_stop=True,
            gen_length=512,
            block_length=32,
            # steps=64,
            threshold=0.7,
            editing_threshold=0.5,
            temperature=0.0,
            max_post_steps=16,
        )


th = threading.Thread(target=_run_generate, daemon=True)
th.start()
with tqdm(desc="Generate", total=None, unit="", leave=True, ncols=80) as pbar:
    while th.is_alive():
        pbar.update(1)
        time.sleep(0.5)
th.join()
generated_tokens = generated_tokens[0]

# # 诊断：看返回的 token 组成（是否全是 mask_id/unk）
# import collections

# ids = generated_tokens[0].tolist()
# cnt = collections.Counter(ids)
# print("[诊断] 返回序列中 token 种类数:", len(cnt))
# print("[诊断] 出现最多的 5 个 token_id 及次数:", cnt.most_common(5))
# print(
#     "[诊断] mask_id=156895 解码为:",
#     repr(tokenizer.decode([156895], skip_special_tokens=False)),
# )
# print(
#     "[诊断] eos_id=156892 解码为:",
#     repr(tokenizer.decode([156892], skip_special_tokens=False)),
# )

# 与官方一致：直接 decode
generated_answer = tokenizer.decode(
    generated_tokens[0],
    skip_special_tokens=True,
)
print("[生成的回答]")
print(generated_answer.strip())
