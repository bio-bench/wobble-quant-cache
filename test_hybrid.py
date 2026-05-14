import torch

from wobble.hybrid import HybridConfig, hybrid_decode, hybrid_encode


def test_hybrid_logic():
    print("--- Testing Block-Hybrid 2.25-bit Round-trip ---")
    D = 256
    N = 1

    original = torch.randn(N, D) * 0.1
    original[0, 0] = 5.0

    vip_indices = torch.arange(32)
    config = HybridConfig(vip_indices=vip_indices)

    q_vips, q_wobbles, meta, wobble_idx = hybrid_encode(original, config)

    reconstructed = hybrid_decode(q_vips, q_wobbles, meta, vip_indices, wobble_idx, D)

    mse = torch.mean((original - reconstructed) ** 2)
    cos_sim = torch.nn.functional.cosine_similarity(original, reconstructed)

    print(f"  MSE: {mse.item():.8f}")
    print(f"  Cosine Similarity: {cos_sim.item():.8f}")

    if cos_sim.item() > 0.98:
        print("\n  SUCCESS: Block-Hybrid logic is verified.")
    else:
        print("\n  FAILURE: Cosine similarity too low.")


if __name__ == "__main__":
    test_hybrid_logic()
