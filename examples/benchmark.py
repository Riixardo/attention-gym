from functools import lru_cache
from typing import Optional, List
import random
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn.functional as F

from tabulate import tabulate
from torch.nn.attention.flex_attention import (
    _DEFAULT_SPARSE_BLOCK_SIZE,
    create_block_mask,
    create_mask,
    flex_attention,
    _score_mod_signature,
    _mask_mod_signature,
    noop_mask,
)

from triton.testing import do_bench

from attn_gym.masks.document_mask import length_to_offsets
from attn_gym.masks import (
    causal_mask,
    generate_sliding_window,
    generate_prefix_lm_mask,
    generate_doc_mask_mod,
    batchify_mask_mod,

)
from attn_gym.mods import generate_alibi_bias, generate_tanh_softcap

torch.set_default_device("cuda")

def generate_pattern_sparse_block_causal_mask(sparsity_ratio: float):
    """
    Generates a mask modifier function that creates a sparse block causal mask
    using a deterministic diagonal striping pattern based on block indices.
    This method avoids any pseudo-randomness.

    Args:
        sparsity_ratio: The target fraction of blocks within the lower
                        triangle (incl. diagonal) to be REMOVED (set to False).
                        Expected to be a multiple of 0.1 (0.0, 0.1, ..., 1.0)
                        for the pattern sparsity to exactly match the ratio.
                        Other values will be effectively rounded to the nearest 0.1.
    Returns:
        A function compatible with flex_attention's mask_mod argument.
    """
    modulus = torch.tensor(10, dtype=torch.int32, device="cuda")
    num_stripes_to_keep = torch.tensor(round(10.0 * (1.0 - sparsity_ratio)), dtype=torch.int32, device="cuda")

    def pattern_sparse_block_causal_mask_mod(b, h, q_idx, kv_idx):

        q_block_idx = q_idx // _DEFAULT_SPARSE_BLOCK_SIZE
        kv_block_idx = kv_idx // _DEFAULT_SPARSE_BLOCK_SIZE
        is_block_causal = q_block_idx >= kv_block_idx

        block_index_sum = q_block_idx + kv_block_idx
        stripe_index = block_index_sum % modulus
        keep_block_pattern = stripe_index < num_stripes_to_keep

        final_mask = is_block_causal & keep_block_pattern

        return final_mask

    return pattern_sparse_block_causal_mask_mod


AVAILABLE_EXAMPLES = {
    "causal_warmup": lambda: test_mask(mask_mod=causal_mask),
    "1": lambda: test_mask(mask_mod=generate_pattern_sparse_block_causal_mask(0.9), plot=True),
    "2": lambda: test_mask(mask_mod=generate_pattern_sparse_block_causal_mask(0.8), plot=True),
    "3": lambda: test_mask(mask_mod=generate_pattern_sparse_block_causal_mask(0.7), plot=True),
    "4": lambda: test_mask(mask_mod=generate_pattern_sparse_block_causal_mask(0.6), plot=True),
    "5": lambda: test_mask(mask_mod=generate_pattern_sparse_block_causal_mask(0.5), plot=True),
    "6": lambda: test_mask(mask_mod=generate_pattern_sparse_block_causal_mask(0.4), plot=True),
    "7": lambda: test_mask(mask_mod=generate_pattern_sparse_block_causal_mask(0.3), plot=True),
    "8": lambda: test_mask(mask_mod=generate_pattern_sparse_block_causal_mask(0.2), plot=True),
    "9": lambda: test_mask(mask_mod=generate_pattern_sparse_block_causal_mask(0.1), plot=True),
    "causal": lambda: test_mask(mask_mod=causal_mask, plot=True),
    # "alibi": lambda: test_mask(score_mod=generate_alibi_bias(16), skip_correctness=True),
    # "sliding_window": lambda: test_mask(mask_mod=generate_sliding_window(window_size=1024)),
    # "prefix_lm": lambda: test_mask(mask_mod=generate_prefix_lm_mask(prefix_length=1024)),
    # "document": lambda: run_document_masking(max_seq_len=32768, num_docs=12),
    # "softcap": lambda: test_mask(
    #     score_mod=generate_tanh_softcap(30, approx=False), skip_correctness=True
    # ),
    # "softcap_approx": lambda: test_mask(
    #     score_mod=generate_tanh_softcap(30, approx=True), skip_correctness=True
    # ),
}

torch.manual_seed(0)

torch._dynamo.config.cache_size_limit = 1000

# Compile the flex_attention function
flex_attention = torch.compile(flex_attention, dynamic=False)

# For better performance, you can use:
# flex_attention = torch.compile(_flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs")

data_type = torch.float16

# The kernels will utilize block sparsity to increase performance
print(f"Using the default sparsity block size: {_DEFAULT_SPARSE_BLOCK_SIZE}")


@lru_cache
def create_block_mask_cached(score_mod, B, H, M, N, device="cuda"):
    block_mask = create_block_mask(score_mod, B, H, M, N, device=device)
    return block_mask


def calculate_tflops(flops: float, time_ms: float, multiplier: int) -> float:
    return multiplier * flops * (1e3 / time_ms) / 1e12


def print_header(text):
    width = 91
    print("╔" + "═" * (width - 2) + "╗")
    print(f"║ {text.center(width - 4)} ║")
    print("╚" + "═" * (width - 2) + "╝")

relative_runtimes = [[], [], []]
block_sparsities = [[], [], []]

def test_mask(
    score_mod: Optional[_score_mod_signature] = None,
    mask_mod: Optional[_mask_mod_signature] = None,
    B: int = 1,
    H: int = 32,
    D: int = 64,
    skip_correctness: bool = False,
    print_mask: bool = True,
    plot: bool = False,
    device: str = "cuda",
):
    assert score_mod is not None or mask_mod is not None, "Must provide a score_mod or mask_mod"

    for i, S in enumerate([8192, 16384, 32768]):
        if mask_mod is not None:
            block_mask = create_block_mask_cached(mask_mod, 1, 1, S, S, device=device)
        else:
            block_mask = None
        sdpa_mask_fn = mask_mod if mask_mod is not None else score_mod
        mask = create_mask(sdpa_mask_fn, 1, 1, S, S, device=device)

        qkv = [
            torch.randn(B, H, S, D, device=device, dtype=data_type, requires_grad=True)
            for _ in range(3)
        ]
        gradOut = torch.randn(B, H, S, D, device=device, dtype=torch.float16)

        causal_fa2 = lambda: F.scaled_dot_product_attention(*qkv, is_causal=True)
        sdpa_mask = lambda: F.scaled_dot_product_attention(*qkv, attn_mask=mask)
        flex_attention_call = lambda: flex_attention(*qkv, score_mod=score_mod, block_mask=block_mask)

        results = []
        if block_mask is not None:
            density = (100 - block_mask.sparsity()) / 100
        else:
            density = 1.0
        causal_fav2_flops = 0.5 * B * H * D * S * S
        flops = density * B * H * D * S * S

        times = []
        for attn in (causal_fa2, sdpa_mask, flex_attention_call):
            fwd_time = do_bench(attn)
            fwd_out = attn()
            bwd_time = do_bench(lambda: fwd_out.backward(gradOut, retain_graph=True))  # noqa: F821
            times.append((fwd_time, bwd_time))

            del fwd_out
            torch.cuda.empty_cache()

        print_header(
            f"{score_mod.__name__ if score_mod is not None else mask_mod.__name__}".replace(
                "_", " "
            ).title()
        )
        # Inline correctness check
        if not skip_correctness:
            sdpa_mask_outs = []
            flex_outs = []

            for tensor in qkv:
                tensor.grad = None

            out1 = sdpa_mask()
            sdpa_mask_outs.append(out1)
            out1.backward(gradOut)
            sdpa_mask_outs += [tensor.grad for tensor in qkv]

            for tensor in qkv:
                tensor.grad = None

            out2 = flex_attention_call()
            flex_outs.append(out2)
            out2.backward(gradOut)
            flex_outs += [tensor.grad for tensor in qkv]
            for flex, sdpa_mask in zip(flex_outs, sdpa_mask_outs):
                torch.testing.assert_close(flex, sdpa_mask, atol=1e-1, rtol=1e-2)

            print("Correctness check passed ✅")

        (
            (causal_fa2_time, causal_fa2_bw_time),
            (sdpa_mask_time, sdpa_mask_bw_time),
            (flex_ms, flex_bw_ms),
        ) = times

        # Usage in your results formatting:
        results = [
            [
                "causal FA2",
                f"{causal_fa2_time:.4f}",
                f"{calculate_tflops(causal_fav2_flops, causal_fa2_time, 4):.2f}",
                f"{causal_fa2_bw_time:.4f}",
                f"{calculate_tflops(causal_fav2_flops, causal_fa2_bw_time, 10):.2f}",
            ],
            [
                "F.sdpa + mask",
                f"{sdpa_mask_time:.4f}",
                f"{calculate_tflops(flops, sdpa_mask_time, 4):.2f}",
                f"{sdpa_mask_bw_time:.4f}",
                f"{calculate_tflops(flops, sdpa_mask_bw_time, 10):.2f}",
            ],
            [
                "flexattention",
                f"{flex_ms:.4f}",
                f"{calculate_tflops(flops, flex_ms, 4):.2f}",
                f"{flex_bw_ms:.4f}",
                f"{calculate_tflops(flops, flex_bw_ms, 10):.2f}",
            ],
        ]
        print(
            tabulate(
                results,
                headers=[
                    "Operation",
                    "FW Time (ms)",
                    "FW FLOPS (TF/s)",
                    "BW Time (ms)",
                    "BW FLOPS (TF/s)",
                ],
                tablefmt="grid",
            )
        )
        if print_mask:
            print(f"\nBlock Mask:\n{block_mask}")
        if print_mask:
            size = S // _DEFAULT_SPARSE_BLOCK_SIZE
            upper_half = (size // 2) * size - (size // 2)
            upper_half_sparsity_contribition = 100 * upper_half / (size ** 2)
            lower_triangle_sparsity = (block_mask.sparsity() - upper_half_sparsity_contribition) * 2
            print(f"\nBlock Mask Lower Triangle Sparsity:\n{lower_triangle_sparsity}")

            if plot:
                relative_runtimes[i].append(flex_ms / causal_fa2_time)
                block_sparsities[i].append(lower_triangle_sparsity)


def run_document_masking(max_seq_len: int, num_docs: int):
    import random

    random.seed(0)

    def generate_random_lengths(total_length, num_documents):
        # Initialize all lengths to 1 to ensure each document has at least one token
        lengths = [1] * num_documents
        remaining_length = total_length - num_documents

        # Randomly distribute the remaining length
        for _ in range(remaining_length):
            index = random.randint(0, num_documents - 1)
            lengths[index] += 1

        return lengths

    lengths = generate_random_lengths(max_seq_len, num_docs)
    offsets = length_to_offsets(lengths, "cuda")
    document_causal_mask = generate_doc_mask_mod(causal_mask, offsets)
    test_mask(mask_mod=document_causal_mask, S=32768)


def main(examples: List[str] = ["all"]):
    """Run the benchmark with the given examples.

    Args:
        examples: List of examples to run. If "all" is specified, all examples will be run.
    """

    if "all" in examples:
        ex_to_run = list(AVAILABLE_EXAMPLES.keys())
    else:
        ex_to_run = examples

    for ex in ex_to_run:
        if ex in AVAILABLE_EXAMPLES:
            AVAILABLE_EXAMPLES[ex]()
            torch.cuda.empty_cache()
        else:
            print(f"Warning: Unknown example key '{ex}'. Skipping.")
    
    relative_runtimes_np = np.array(relative_runtimes)
    block_sparsities_np = np.array(block_sparsities)
    # Plot the main lines
    plt.plot(block_sparsities_np.T, relative_runtimes_np.T, label=["seq_len=8192", "seq_len=16384", "seq_len=32768"])
    plt.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, label='baseline')
    x_vals = np.linspace(0, 100, 101)
    plt.plot(x_vals, (1 - x_vals / 100), color='black', linestyle=':', label='theoretical')

    plt.xlabel("Sparsity Ratio of Lower Triangle")
    plt.ylabel("Relative Runtime Compared to FlashAttentionV2")
    plt.title("flexattention block sparse hdim64_nheads_32_bts1_fwd_causal")
    plt.legend()
    plt.savefig("runtimes_vs_sparsity.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    try:
        from jsonargparse import ArgumentParser
    except ImportError:
        raise ImportError("Be sure to run: pip install -e .'[viz]'")
    parser = ArgumentParser(description="Run specific examples or all examples.")
    parser.add_argument(
        "--examples",
        type=str,
        nargs="+",
        default=["all"],
        help="List of examples to run. Use space to separate multiple examples. "
        "Available options: "
        + ", ".join(sorted(AVAILABLE_EXAMPLES.keys()))
        + ", or 'all' to run all examples.",
    )

    args = parser.parse_args()
    main(**vars(args))
