import torch
from tqdm import tqdm

from gem.ml.models.perm_invariant_blocks import MultiHeadSetPool, MultiHeadUnit
from gem.ml.models.abundance_pred import SGBEmbedding, ModelBlock


def test_perm_MultiHeadSetPool(
        n_trials: int,
        seed: int,
        input_dtype=torch.float32,
        model_epsilon_tolerance: float = 0.00001,
):
    """
    This is meant to be a unit test; it has assertion statements.
    If all is fine, this function finishes and returns nothing.
    """
    torch.manual_seed(seed)
    m = MultiHeadSetPool(
        genome_feature_dim=5,
        out_dim_per_head=6,
        num_heads=7,
    ).to(input_dtype)

    n_batch = 1
    n_genomes = 10
    for i in tqdm(range(n_trials), desc='Test-MultiHeadSetPool'):
        g = torch.rand(size=(n_batch, n_genomes, 5), dtype=input_dtype)
        g_mask = (torch.rand(n_batch, n_genomes) < 0.75)  # boolean tensor

        # Permute the input.
        perm_indices = torch.randperm(g.shape[1])
        h = g[:, perm_indices, :]
        h_mask = g_mask[:, perm_indices]
        with torch.no_grad():
            torch.manual_seed(42)  # Make dropout reproducible.
            mg = m(g, g_mask)
            torch.manual_seed(42)  # Make dropout reproducible.
            mh = m(h, h_mask)
            if not torch.allclose(mh, mg, atol=model_epsilon_tolerance):
                print("Violation of permutation invariance!")
                print(f"perm = {perm_indices}")
                print(f"g = {g}")
                print(f"h = {h}")
                print(f"m(g) = {mg}")
                print(f"m(h) = {mh}")
                print(f"Difference = {torch.abs(mg - mh)}")
                raise AssertionError("Expected m(g) to be equal to m(PERM(g)).")


def test_perm_MultiHeadUnit(
        n_trials: int,
        seed: int,
        input_dtype=torch.float32,
        model_epsilon_tolerance: float = 0.00001,
):
    """
    This is meant to be a unit test; it hash) = {mh}")
                print(f"Difference = {torch.abs(mg - mh)}")
                raise AssertionError("Expected m(g) to be equal to m(PERM(g)).")

 assertion statements.
    If all is fine, this function finishes and returns nothing.
    """
    torch.manual_seed(seed)
    m = MultiHeadUnit(
        model_dim=5,
        genome_dim=6,
        num_heads=7,
        key_query_dim=8,
        latent_collection_dim=9,
        combination_latent_dim=10,
    ).to(input_dtype)

    n_batch = 1
    n_genomes = 10
    for i in tqdm(range(n_trials), desc='Test-MultiHeadUnit'):
        gx = torch.rand(size=(n_batch, n_genomes, m.model_dim), dtype=input_dtype)
        g = torch.rand(size=(n_batch, n_genomes, m.genome_dim), dtype=input_dtype)
        g_mask = (torch.rand(n_batch, n_genomes) < 0.75)  # boolean tensor

        # Permute the input.
        perm_indices = torch.randperm(g.shape[1])
        hx = gx[:, perm_indices, :]
        h = g[:, perm_indices, :]
        h_mask = g_mask[:, perm_indices]
        with torch.no_grad():
            torch.manual_seed(42)  # Make dropout reproducible.
            mg = m(gx, g, g_mask)
            torch.manual_seed(42)  # Make dropout reproducible.
            mh = m(hx, h, h_mask)
            if not torch.allclose(mh, mg[:, perm_indices, :], atol=model_epsilon_tolerance):
                print("Violation of permutation invariance!")
                print(f"perm = {perm_indices}")
                print(f"g = {g}")
                print(f"h = {h}")
                print(f"m(g) = {mg}")
                print(f"m(h) = {mh}")
                print(f"Difference = {torch.abs(mg - mh)}")
                raise AssertionError("Expected m(g) to be equal to m(PERM(g)).")


def test_perm_SGBEmbedding(
        n_trials: int,
        seed: int,
        input_dtype=torch.float32,
        model_epsilon_tolerance: float = 0.00001,
):
    """
    This is meant to be a unit test; it has assertion statements.
    If all is fine, this function finishes and returns nothing.
    """
    torch.manual_seed(seed)
    m = SGBEmbedding(
        input_embed_dim=6,
        latent_dim=7,
        out_dim=8,
    ).to(input_dtype)

    n_batch = 1
    n_sgbs = 10
    n_markers = 11
    for i in tqdm(range(n_trials), desc='Test-SGBEmbedding'):
        g = torch.rand(size=(n_batch, n_sgbs, n_markers, 6), dtype=input_dtype)
        g_mask = (torch.rand(n_batch, n_sgbs, n_markers) < 0.75)  # boolean tensor

        # Permute the input.
        sgb_perm_indices = torch.randperm(g.shape[1])
        h1 = g[:, sgb_perm_indices, :, :]
        h1_mask = g_mask[:, sgb_perm_indices, :]

        marker_perm_indices = torch.randperm(g.shape[2])
        h2 = g[:, :, marker_perm_indices, :]
        h2_mask = g_mask[:, :, marker_perm_indices]
        with torch.no_grad():
            torch.manual_seed(42)  # Make dropout reproducible.
            mg = m(g, g_mask)
            torch.manual_seed(42)  # Make dropout reproducible.
            mh1 = m(h1, h1_mask)
            torch.manual_seed(42)  # Make dropout reproducible.
            mh2 = m(h2, h2_mask)

            if not torch.allclose(mh1, mg[:, sgb_perm_indices, :], atol=model_epsilon_tolerance):
                print("Violation of permutation invariance!")
                print(f"perm = {sgb_perm_indices}")
                print(f"g = {g}")
                print(f"h = {h1}")
                print(f"m(g) = {mg}")
                print(f"m(h) = {mh1}")
                print(f"Difference = {torch.abs(mg - mh1)}")
                raise AssertionError("Expected m(g) to be equal to m(PERM(g)).")

            if not torch.allclose(mh2, mg, atol=model_epsilon_tolerance):
                print("Violation of permutation invariance!")
                print(f"perm = {marker_perm_indices}")
                print(f"g = {g}")
                print(f"h = {h2}")
                print(f"m(g) = {mg}")
                print(f"m(h) = {mh2}")
                print(f"Difference = {torch.abs(mg - mh2)}")
                raise AssertionError("Expected m(g) to be equal to m(PERM(g)).")


def test_perm_ModelBlock(
        n_trials: int,
        seed: int,
        input_dtype=torch.float32,
        model_epsilon_tolerance: float = 0.00001,
):
    """
    This is meant to be a unit test; it has assertion statements.
    If all is fine, this function finishes and returns nothing.
    """
    torch.manual_seed(seed)
    m = ModelBlock(
        model_dim=5,
        genome_dim=6,
        num_heads=7,
        key_query_dim=8,
        latent_collection_dim=9,
        combination_latent_dim=10,
    ).to(input_dtype)

    n_batch = 1
    n_genomes = 10
    for i in tqdm(range(n_trials), desc='Test-MultiHeadUnit'):
        gx = torch.rand(size=(n_batch, n_genomes, 5), dtype=input_dtype)
        g = torch.rand(size=(n_batch, n_genomes, 6), dtype=input_dtype)
        g_mask = (torch.rand(n_batch, n_genomes) < 0.75)  # boolean tensor

        # Permute the input.
        perm_indices = torch.randperm(g.shape[1])
        hx = gx[:, perm_indices, :]
        h = g[:, perm_indices, :]
        h_mask = g_mask[:, perm_indices]
        with torch.no_grad():
            torch.manual_seed(42)  # Make dropout reproducible.
            mg = m(gx, g, g_mask)
            torch.manual_seed(42)  # Make dropout reproducible.
            mh = m(hx, h, h_mask)
            if not torch.allclose(mh, mg[:, perm_indices, :], atol=model_epsilon_tolerance):
                print("Violation of permutation invariance!")
                print(f"perm = {perm_indices}")
                print(f"g = {g}")
                print(f"h = {h}")
                print(f"m(g) = {mg}")
                print(f"m(h) = {mh}")
                print(f"Difference = {torch.abs(mg - mh)}")
                raise AssertionError("Expected m(g) to be equal to m(PERM(g)).")


if __name__ == "__main__":
    test_perm_MultiHeadSetPool(n_trials=500, seed=1000)
    test_perm_MultiHeadUnit(n_trials=500, seed=1001)
    test_perm_SGBEmbedding(n_trials=500, seed=1002)
    test_perm_ModelBlock(n_trials=500, seed=1003)
    print("all tests passed!")