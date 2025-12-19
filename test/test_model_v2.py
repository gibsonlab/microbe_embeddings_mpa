import torch
from tqdm import tqdm
from gem.ml.models.v2 import V2Layer as Layer, SGBAbundanceLayeredPredictionModel as Model

def test_layer(seed=123):
    torch.manual_seed(seed)
    sgb_marker_embed_dim = 6
    sgb_model_dim = 5
    sgb_proj_dim_per_head = 4
    layer_num_heads = 3
    batch_sz = 2
    n_sgbs = 7
    n_markers = 8

    layer = Layer(
        sgb_marker_embed_dim=sgb_marker_embed_dim,
        sgb_model_dim=sgb_model_dim,
        sgb_proj_dim_per_head=sgb_proj_dim_per_head,
        num_heads=layer_num_heads,
    )
    marker_padding_mask = torch.ones((batch_sz, n_sgbs, n_markers), dtype=torch.bool)
    sgb_padding_mask = torch.ones((batch_sz, n_sgbs), dtype=torch.bool)

    g = torch.rand(batch_sz, n_sgbs, n_markers, sgb_marker_embed_dim)
    Y = torch.ones(g.shape[:-2] + (sgb_model_dim,), dtype=g.dtype)
    A2, Y2 = layer(g, Y, marker_padding_mask, sgb_padding_mask)

    # print("A2: =======================================================")
    # print(A2)
    # print(A2.shape)
    #
    # print("Y2: =======================================================")
    # print(Y2)
    # print(Y2.shape)

    assert A2.shape == torch.Size([batch_sz, n_sgbs])
    assert Y2.shape == torch.Size([batch_sz, n_sgbs, sgb_model_dim])


def test_layer_perm_invariance(
        n_trials: int,
        seed: int = 123,
        model_epsilon_tolerance: float = 0.00001,
):
    """
    This is meant to be a unit test; it has assertion statements.
    If all is fine, this function finishes and returns nothing.
    """
    torch.manual_seed(seed)
    sgb_marker_embed_dim = 6
    sgb_model_dim = 5
    sgb_proj_dim_per_head = 4
    layer_num_heads = 3
    batch_sz = 2
    n_sgbs = 7
    n_markers = 8

    layer = Layer(
        sgb_marker_embed_dim=sgb_marker_embed_dim,
        sgb_model_dim=sgb_model_dim,
        sgb_proj_dim_per_head=sgb_proj_dim_per_head,
        num_heads=layer_num_heads,
    )

    layer.eval()
    for i in tqdm(range(n_trials), desc=f'Test:{Layer.__name__}'):
        g = torch.rand(batch_sz, n_sgbs, n_markers, sgb_marker_embed_dim)
        g_mpadding = (torch.rand(batch_sz, n_sgbs, n_markers) < 0.75)
        g_spadding = (torch.rand(batch_sz, n_sgbs) < 0.75)
        Y = torch.ones(g.shape[:-2] + (sgb_model_dim,), dtype=g.dtype)

        # Permute the input.
        perm_indices = torch.randperm(g.shape[1])
        h = g[:, perm_indices, :, :]
        h_mpadding = g_mpadding[:, perm_indices, :]
        h_spadding = g_spadding[:, perm_indices]

        with torch.no_grad():
            out_g_A, out_g_Y = layer(g, Y, g_mpadding, g_spadding)
            out_h_A, out_h_Y = layer(h, Y, h_mpadding, h_spadding)

            if not torch.allclose(out_h_A, out_g_A[:, perm_indices], atol=model_epsilon_tolerance):
                print("Violation of permutation invariance!")
                print(f"perm = {perm_indices}")
                print(f"out_h_A = {out_h_A}")
                print(f"PERM[out_g_A] = {out_g_A[:, perm_indices]}")
                raise AssertionError("Expected m(g) to be equal to m(PERM(g)).")

            if not torch.allclose(out_h_Y, out_g_Y[:, perm_indices], atol=model_epsilon_tolerance):
                print("Violation of permutation invariance!")
                print(f"perm = {perm_indices}")
                print(f"out_h_A:\n{out_h_Y}")
                print(f"PERM[out_g_A]:\n{out_g_Y[:, perm_indices]}")
                raise AssertionError("Expected m(g) to be equal to m(PERM(g)).")


def test_model_v2_perm_invariance(
        n_layers: int,
        n_trials: int,
        seed: int = 123,
        model_epsilon_tolerance: float = 0.0001,
):
    torch.manual_seed(seed)
    sgb_marker_embed_dim = 6
    sgb_model_dim = 5
    sgb_proj_dim_per_head = 4
    layer_num_heads = 3
    batch_sz = 2
    n_sgbs = 7
    n_markers = 8

    model = Model(
        num_layers=n_layers,
        sgb_model_dim=sgb_model_dim,
        marker_embed_dim=sgb_marker_embed_dim,
        sgb_proj_dim_per_head=sgb_proj_dim_per_head,
        layer_num_heads=layer_num_heads,
    )

    model.eval()
    for i in tqdm(range(n_trials), desc=f'Test:{Model.__name__} ({n_layers} layers)'):
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

            if not torch.allclose(out_h_logits, out_g_logits[:, perm_indices], atol=model_epsilon_tolerance):
                print("Violation of permutation invariance!")
                print(f"perm = {perm_indices}")
                print(f"out_h_logits:\n{out_h_logits}")
                print(f"PERM[out_g_logits]:\n{out_g_logits[:, perm_indices]}")
                raise AssertionError("Expected m(g) to be equal to m(PERM(g)).")


if __name__ == "__main__":
    test_layer()
    test_layer_perm_invariance(n_trials=500)
    test_model_v2_perm_invariance(n_layers=1, n_trials=500)
    test_model_v2_perm_invariance(n_layers=2, n_trials=500)
    test_model_v2_perm_invariance(n_layers=3, n_trials=500)