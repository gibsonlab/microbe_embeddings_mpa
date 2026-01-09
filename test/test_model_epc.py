import torch
from tqdm import tqdm
from gem.ml.models.embed_pool_concat import SGBEmbedPoolConcatPredictionModel as Model


def test_epc_model_perm_invariance(
        n_trials: int,
        seed: int = 123,
        model_epsilon_tolerance: float = 0.0001,
):
    torch.manual_seed(seed)
    sgb_marker_embed_dim = 6
    sgb_model_dim = 5
    batch_sz = 2
    n_sgbs = 7
    n_markers = 8

    model = Model(
        sgb_model_dim=sgb_model_dim,
        marker_embed_dim=sgb_marker_embed_dim,
        hidden_dim=11,
        use_sgb_pooling=True,
        sgb_pool_dim=10,
    )

    model.eval()
    for i in tqdm(range(n_trials), desc=f'Test:{Model.__name__}'):
        g = torch.rand(batch_sz, n_sgbs, n_markers, sgb_marker_embed_dim)
        g_mpadding = (torch.rand(batch_sz, n_sgbs, n_markers) < 0.75)
        g_spadding = (torch.rand(batch_sz, n_sgbs) < 0.75)

        # Permute the input.
        perm_indices = torch.randperm(g.shape[1])
        h = g[:, perm_indices, :, :]
        h_mpadding = g_mpadding[:, perm_indices, :]
        h_spadding = g_spadding[:, perm_indices]

        with torch.no_grad():
            out_g_logits = model(g, g_mpadding, g_spadding)
            out_h_logits = model(h, h_mpadding, h_spadding)
            assert out_g_logits.shape == torch.Size([batch_sz, n_sgbs])
            if i <= 5:
                print(f"i = {i}")
                print(out_g_logits)
            # if i == 0:
            #     print(out_g_logits)
            #     print(g_spadding)

            if not torch.allclose(out_h_logits, out_g_logits[:, perm_indices], atol=model_epsilon_tolerance):
                print("Violation of permutation invariance!")
                print(f"perm = {perm_indices}")
                print(f"out_h_logits:\n{out_h_logits}")
                print(f"PERM[out_g_logits]:\n{out_g_logits[:, perm_indices]}")
                raise AssertionError("Expected m(g) to be equal to m(PERM(g)).")


if __name__ == "__main__":
    test_epc_model_perm_invariance(n_trials=500)