import torch
import torch.nn as nn
import my_layernorm_backend


def test_layernorm(rows=4, cols=128, eps=1e-5, device="cuda"):
    print(f"=== LayerNorm Correctness Test ===")
    print(f"Input shape: ({rows}, {cols})")

    # Generate random input
    torch.manual_seed(42)
    x = torch.randn(rows, cols, device=device, dtype=torch.float32)

    # PyTorch official implementation (no gamma/beta)
    ln = nn.LayerNorm(cols, eps=eps, elementwise_affine=False).to(device)
    y_ref = ln(x)

    # Custom HIP implementation
    y_custom = my_layernorm_backend.layernorm_forward(x, eps)

    # Print outputs separately
    print("\nOfficial PyTorch output:")
    print(y_ref)

    print("\nCustom HIP output:")
    print(y_custom)

    # Comparison
    max_abs_diff = (y_ref - y_custom).abs().max().item()
    mean_abs_diff = (y_ref - y_custom).abs().mean().item()

    print(f"\nMax absolute difference: {max_abs_diff:.6e}")
    print(f"Mean absolute difference: {mean_abs_diff:.6e}")

    # Pass/Fail decision
    if max_abs_diff < 1e-4:
        print("Test PASSED. Error is within acceptable range.\n")
        return True
    else:
        print("Test FAILED. Error is too large.\n")
        return False


if __name__ == "__main__":
    all_pass = True
    all_pass &= test_layernorm(rows=2, cols=4)
    all_pass &= test_layernorm(rows=64, cols=256)
    all_pass &= test_layernorm(rows=128, cols=1024)
    all_pass &= test_layernorm(rows=512, cols=4096)

    if all_pass:
        print("All tests PASSED.")
    else:
        print("Some tests FAILED. Please check the implementation.")
