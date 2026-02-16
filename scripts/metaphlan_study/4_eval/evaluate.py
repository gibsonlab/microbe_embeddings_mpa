from typing import *
from pathlib import Path
import json
import gc

from tqdm import tqdm
import torch
from torch import Tensor, nn

import numpy as np
import scipy
import pandas as pd

from gem.datasets.mpa import AbstractMetaphlanPreembeddedDataset, MetaphlanPreembeddedDatasetMemmapped, MetaphlanProfileParser
from gem.ml import SGBAbundancePredictionModel, SGBAbundanceLayeredPredictionModel, safe_kl_div_loss



class BaselinePredictor:
    def predict_abundances(self, sgb_ids: List[str], max_sgbs: int) -> np.ndarray:
        raise NotImplementedError()

    def predict_batch_abundances(self, batched_inputs: List[List[str]], max_sgbs: int) -> np.ndarray:
        return np.stack([self.predict_abundances(batch, max_sgbs) for batch in batched_inputs], axis=0)

    @staticmethod
    def add_padding_1d(pred_vector: np.ndarray, max_size: int) -> np.ndarray:
        if max_size > len(pred_vector):
            pred_vector = np.concatenate([pred_vector, np.zeros(max_size - len(pred_vector), dtype=pred_vector.dtype)])
        return pred_vector



""" Class definition. """
class UniformAbundancePredictor(BaselinePredictor):
    def __init__(self):
        pass

    def predict_abundances(self, sgb_ids: List[str], max_sgbs: int) -> np.ndarray:
        """ Input validation """
        # Assure sizes don't exceed what is specified.
        if len(sgb_ids) > max_sgbs:
            raise ValueError("Input has more IDs ({}) than the max allotted slots ({})".format(len(sgb_ids), max_sgbs))

        """ Compute output. """
        pred = np.ones(len(sgb_ids), dtype=float)
        pred = pred / np.sum(pred)
        pred = self.add_padding_1d(pred, max_sgbs)
        return pred


class TreeNode:
    """Represents a node in the phylogenetic tree."""
    
    def __init__(self, name: Optional[str] = None, node_id: int = -1):
        self.name = name
        self.node_id = node_id
        self.children: List[TreeNode] = []
        self.parent: Optional[TreeNode] = None
        self.branch_length: float = 0.0
        self.dist_from_root: float = 0.0
        self.depth: int = 0  # Depth in tree (for LCA)
        
    def is_leaf(self) -> bool:
        return len(self.children) == 0


class NewickParser:
    """Parser for Newick format phylogenetic trees."""
    
    def __init__(self, newick_string: str):
        self.newick = newick_string.strip()
        self.pos = 0
        self.node_counter = 0
        
    def parse(self) -> TreeNode:
        """Parse the Newick string and return the root node."""
        if self.newick.endswith(';'):
            self.newick = self.newick[:-1]
        root = self._parse_node()
        return root
    
    def _parse_node(self) -> TreeNode:
        """Recursively parse a node and its descendants."""
        node = TreeNode(node_id=self.node_counter)
        self.node_counter += 1
        
        if self.pos < len(self.newick) and self.newick[self.pos] == '(':
            # Internal node with children
            self.pos += 1  # skip '('
            
            while True:
                child = self._parse_node()
                child.parent = node
                node.children.append(child)
                
                if self.pos < len(self.newick) and self.newick[self.pos] == ',':
                    self.pos += 1  # skip ','
                else:
                    break
            
            if self.pos < len(self.newick) and self.newick[self.pos] == ')':
                self.pos += 1  # skip ')'
        
        # Parse node label
        label_start = self.pos
        while self.pos < len(self.newick) and self.newick[self.pos] not in ',:();':
            self.pos += 1
        
        if self.pos > label_start:
            node.name = self.newick[label_start:self.pos]
        
        # Parse branch length
        if self.pos < len(self.newick) and self.newick[self.pos] == ':':
            self.pos += 1  # skip ':'
            length_start = self.pos
            while self.pos < len(self.newick) and self.newick[self.pos] not in ',();':
                self.pos += 1
            
            length_str = self.newick[length_start:self.pos]
            try:
                node.branch_length = float(length_str)
            except ValueError:
                node.branch_length = 0.0
        
        return node


class OptimizedDistanceMatrixComputer:
    """
    Optimized distance matrix computation using efficient LCA queries.
    
    Key optimizations:
    1. Precompute all distances from root
    2. Use binary lifting for O(log H) LCA queries (H = height)
    3. Batch process columns to improve cache locality
    """
    
    def __init__(self, newick_string: str):
        """Initialize with a Newick format tree string."""
        parser = NewickParser(newick_string)
        self.root = parser.parse()
        self.leaf_nodes: Dict[str, TreeNode] = {}
        self.all_nodes: List[TreeNode] = []
        
        # Collect all nodes and leaves
        print("collecting all nodes and leaves...")
        self._collect_nodes(self.root)
        
        # Precompute distances and depths
        print("precomputing distances and tree depths...")
        self._compute_root_distances_and_depths(self.root, 0.0, 0)
        
        # Precompute binary lifting table for fast LCA
        print("precomputing binary lifting table for fast LCA lookup...")
        self._precompute_lca_table()
        
    def _collect_nodes(self, node: TreeNode) -> None:
        """Collect all nodes and leaf nodes."""
        self.all_nodes.append(node)
        if node.is_leaf():
            if node.name:
                self.leaf_nodes[node.name] = node
        else:
            for child in node.children:
                self._collect_nodes(child)
    
    def _compute_root_distances_and_depths(self, node: TreeNode, dist: float, depth: int) -> None:
        """Compute distance from root and depth for all nodes."""
        node.dist_from_root = dist
        node.depth = depth
        for child in node.children:
            self._compute_root_distances_and_depths(
                child, 
                dist + child.branch_length, 
                depth + 1
            )
    
    def _precompute_lca_table(self) -> None:
        """
        Precompute binary lifting table for O(log H) LCA queries.
        
        ancestor[node][i] = 2^i-th ancestor of node
        """
        n = len(self.all_nodes)
        if n == 0:
            return
        
        # Find maximum depth to determine table size
        max_depth = max(node.depth for node in self.all_nodes)
        log_depth = max_depth.bit_length()
        
        # Initialize table
        self.ancestor = {}
        for node in self.all_nodes:
            self.ancestor[node.node_id] = [None] * log_depth
        
        # Fill table
        for node in self.all_nodes:
            if node.parent is not None:
                self.ancestor[node.node_id][0] = node.parent
        
        # Binary lifting: ancestor[node][i] = ancestor[ancestor[node][i-1]][i-1]
        for i in range(1, log_depth):
            for node in self.all_nodes:
                if self.ancestor[node.node_id][i-1] is not None:
                    prev_ancestor = self.ancestor[node.node_id][i-1]
                    self.ancestor[node.node_id][i] = self.ancestor[prev_ancestor.node_id][i-1]
    
    def _find_lca_optimized(self, node1: TreeNode, node2: TreeNode) -> TreeNode:
        """
        Find LCA using binary lifting in O(log H) time.
        """
        # Make node1 the deeper node
        if node1.depth < node2.depth:
            node1, node2 = node2, node1
        
        # Bring node1 to same level as node2
        depth_diff = node1.depth - node2.depth
        log_depth = len(self.ancestor[node1.node_id])
        
        for i in range(log_depth):
            if depth_diff & (1 << i):
                node1 = self.ancestor[node1.node_id][i]
                if node1 is None:
                    break
        
        # If node2 is ancestor of node1
        if node1 == node2:
            return node1
        
        # Binary search for LCA
        for i in range(log_depth - 1, -1, -1):
            if (self.ancestor[node1.node_id][i] is not None and 
                self.ancestor[node2.node_id][i] is not None and
                self.ancestor[node1.node_id][i] != self.ancestor[node2.node_id][i]):
                node1 = self.ancestor[node1.node_id][i]
                node2 = self.ancestor[node2.node_id][i]
        
        return node1.parent if node1.parent is not None else node1
    
    def compute_distance_matrix(self, subset_leaves: List[str]) -> Tuple[np.ndarray, List[str]]:
        """
        Compute L×K distance matrix efficiently.
        
        Time Complexity: O(L*log(L) + K*L*log(H)) where H is tree height
        Space Complexity: O(N*log(H)) for LCA table, where N is total nodes
        
        For balanced trees, H = O(log(L)), giving O(L*log(L) + K*L*log(log(L)))
        For unbalanced trees, worst case is still better than naive O(LK) with large constants.
        """
        all_leaves = sorted(self.leaf_nodes.keys())
        L = len(all_leaves)
        K = len(subset_leaves)
        
        # Verify all subset leaves exist
        for leaf in subset_leaves:
            if leaf not in self.leaf_nodes:
                raise ValueError(f"Leaf '{leaf}' not found in tree")
        
        # Initialize distance matrix
        dist_matrix = np.zeros((L, K), dtype=float)
        
        # Precompute leaf nodes for all_leaves
        source_nodes = [self.leaf_nodes[leaf] for leaf in all_leaves]
        target_nodes = [self.leaf_nodes[leaf] for leaf in subset_leaves]
        
        # For each column (target leaf in subset)
        for k, target_node in enumerate(target_nodes):
            # Compute distances from target to all leaves
            for i, source_node in enumerate(source_nodes):
                if source_node == target_node:
                    dist_matrix[i, k] = 0.0
                else:
                    # Find LCA and compute distance
                    lca = self._find_lca_optimized(source_node, target_node)
                    dist = (source_node.dist_from_root + 
                           target_node.dist_from_root - 
                           2 * lca.dist_from_root)
                    dist_matrix[i, k] = dist
        
        return dist_matrix, all_leaves
    
    def get_all_leaf_names(self) -> List[str]:
        """Return sorted list of all leaf names."""
        return sorted(self.leaf_nodes.keys())


def compute_tree_distance_matrix_optimized(
    newick_file: str, 
    subset_leaves: List[str]
) -> Tuple[np.ndarray, List[str]]:
    """
    High-level function to compute distance matrix from a Newick file (optimized version).
    
    Args:
        newick_file: Path to Newick format file
        subset_leaves: List of K leaf names for columns
    
    Returns:
        Tuple of (distance_matrix, row_labels, column_labels)
    """
    with open(newick_file, 'r') as f:
        newick_string = f.read().strip()
    
    computer = OptimizedDistanceMatrixComputer(newick_string)
    dist_matrix, all_leaves = computer.compute_distance_matrix(subset_leaves)
    
    return dist_matrix, all_leaves


""" Closest leaf finder. """
def geometric_mean(x: np.ndarray, eps: float = 1e-6) -> float:
    return np.exp(np.mean(np.log(x + eps)))


class ClosestLeafFinder:
    def __init__(self, tree_file: Path, key_leaf_subset: List[str]) -> None:
        self.k_subset = set(key_leaf_subset)
        self.k_list = key_leaf_subset
        self.dist_mat, all_leaves = compute_tree_distance_matrix_optimized(tree_file, key_leaf_subset)
        self.leaf_ordering = {lname: i for i, lname in enumerate(all_leaves)}
        # self.dist_mat, self.leaf_ordering = setup_k_distances(tree_file, key_leaf_subset)
    
    def closest_leaf(self, query_leaf_name: str) -> str:
        """
        Find the closest node in subset K to query node X.
        
        Parameters:
        -----------
        query_leaf_name : str
            Name of the query leaf node
        k_distance_matrix : dict
            Precomputed distance matrix from setup_k_distances
        
        Returns:
        --------
        tuple : (closest_node_name, distance)
            The name of the closest K node and its distance to query_x
        """
        if query_leaf_name not in self.leaf_ordering:
            raise KeyError(f"Query node '{query_leaf_name}' not found in tree")
        if query_leaf_name in self.k_subset:
            return query_leaf_name

        query_idx = self.leaf_ordering[query_leaf_name]
        k_idx = np.argmin(self.dist_mat[query_idx])
        return self.k_list[k_idx]


class NearestNeighborAveragingPredictor(BaselinePredictor):
    """ 
    Predict an abundance by assigning a weight to each SGB in the specified query sample.
    The weight of SGB "X" is equal to the geometric mean of the abundance of NearestNeighbor(X) in the training samples.
    
    This class caches the result of input queries.
    
    Note: If "X" itself is in the training sample, then NearestNeighbor(X) = X, so we are counting the abundnaces of "X" itself in the training set.
    """
    def __init__(self, sgb_phylogenetic_tree_file: Path, train_df: pd.DataFrame):
        assert train_df.shape[0] > 0
        profile_df = MetaphlanProfileParser(train_df).sgb_profile_df
        zero_cols = profile_df.columns[(profile_df == 0).all()]
        profile_df = profile_df.drop(columns=zero_cols)
        profile_df = profile_df.rename(columns=lambda x: x[:-6] if x.endswith('_group') else x)

        self.profile_df = profile_df
        print("Initializing Nearest Neighbor predictor on {} training samples.".format(profile_df.shape[0]))
        print("# of training SGBs = {}".format(profile_df.shape[1]))

        self.training_sgbs = set(self.profile_df.columns)
        for sgb in self.training_sgbs:
            if sgb.endswith("_group"):
                raise Exception("unexpected suffix found")
        # tree = Phylo.read(sgb_phylogenetic_tree_file, "newick")
        self.nearest_leaf_finder = ClosestLeafFinder(
            sgb_phylogenetic_tree_file, 
            [sgb[3:] for sgb in self.training_sgbs], 
            # cache_outputs=False
        )  # note: in this current implementation, double-caching is redundant.
        self._weight_cache: Dict[str, float] = dict()

    def sgb_weights(self, sgb_ids: List[str]) -> np.ndarray:
        return np.array([
            self.calculate_sgb_weight(sgb_id)
            for sgb_id in sgb_ids
        ])

    def calculate_sgb_weight(self, sgb_id: str) -> float:
        """ Implement the nearest-neighbor lookup strategy. """
        """ Check if result was already computed. """
        if sgb_id in self._weight_cache:
            return self._weight_cache[sgb_id]
        
        """ Compute the nearest-neighbor in training set. """
        if sgb_id in self.training_sgbs:
            nearest_neighbor_sgb_id = sgb_id
        else:
            # First, strip the prefix "SGB" from the string. Note the [3:] in the input indexing.
            assert sgb_id.startswith("SGB"), f"Expected input sgb ID to start with prefix `SGB`, got the ID `{sgb_id}` instead."
            try:
                nearest_neighbor_sgb_id = self.nearest_leaf_finder.closest_leaf(sgb_id[3:])
                nearest_neighbor_sgb_id = f"SGB{nearest_neighbor_sgb_id}"  # we need to re-attach the "SGB" prefix.
            except KeyError:
                weight = 0.0
                return weight
            

        """ If not, then look up abundances in training samples. """
        nn_abunds = self.profile_df[nearest_neighbor_sgb_id]
        nn_abunds = nn_abunds[nn_abunds != 0]
        if nn_abunds.shape[0] == 0:
            raise ValueError(f"In training data, found 0 occurrences of SGB {sgb_id}'s nearest neighbor {nearest_neighbor_sgb_id}. This should not happen!")
        weight = geometric_mean(nn_abunds.to_numpy())
        self._weight_cache[sgb_id] = weight
        return weight
            

    def predict_abundances(self, sgb_ids: List[str], max_sgbs: int) -> np.ndarray:
        """ Input validation """
        # Assure sizes don't exceed what is specified.
        if len(sgb_ids) > max_sgbs:
            raise ValueError("Input has more IDs ({}) than the max allotted slots ({})".format(
                len(sgb_ids), max_sgbs
            ))

        """ Compute output. """
        wts = self.sgb_weights(sgb_ids)
        
        # normalize and add padding.
        if np.sum(wts) == 0:
            raise ValueError("Unable to make any reasonable prediction! This should not occur.")
        pred = wts / np.sum(wts)
        pred = self.add_padding_1d(pred, max_sgbs)
        return pred



# ===================================================== EVALUATION FUNCTION


def evaluate_method(
    inference_fn: Callable[[AbstractMetaphlanPreembeddedDataset, int], Tensor],
    dset: AbstractMetaphlanPreembeddedDataset
) -> pd.DataFrame:
    """
    :param inference_fn: A function (e.g. lambda expression) which takes a MicrobiomeSample as input and outputs a (abund_predictions) tensor. The abund_predictions should be a vector of LOGS of relative abundances.
    :param test_df: A dataframe listing the samples to test on (e.g. test_df in this jupyter notebook).
    """
    df_entries = []

    n_samples = len(dset)
    for sample_idx in tqdm(range(n_samples)):
        """ Prediction. """
        try:
            pred_sgb_log_abunds: Tensor = inference_fn(dset, sample_idx)
        except Exception as e:
            print(f"During inference of sample {sample_idx}, got an error.")
            print(f"The error was: {e}")
            print(f"Skipping sample {sample.sample_id}")
            continue
        assert len(pred_sgb_log_abunds.shape) == 1, "Expected a 1-d tensor for the true abundances."
        assert torch.logsumexp(pred_sgb_log_abunds, dim=0).isclose(torch.tensor(0.0), rtol=1e-3, atol=1e-4).item(), f"Expected logsumexp(pred) to be equal to 0.0, got {torch.logsumexp(pred_sgb_log_abunds, dim=0).item()}."
        # assert num_sgbs == pred_sgb_log_abunds.shape[0], "Pred abundances should have the same size/shape as the list of true SGB ids."
        
        """ Ground-truth. """
        true_sgb_abunds = dset.true_abundance_profile(sample_idx)
        assert len(true_sgb_abunds.shape) == 1, "Expected a 1-d tensor for the true abundances."
        assert true_sgb_abunds.sum().isclose(torch.tensor(1.0), rtol=1e-4, atol=1e-7).item(), f"Expected true relative abundances to sum to 1.0, got {true_sgb_abunds.sum()}"
        # assert num_sgbs == true_sgb_abunds.shape[0], "True abundances should have the same size/shape as the list of true SGB ids."

        """ Metric evaluation. """
        # KL Divergence
        kl_loss = safe_kl_div_loss(
            pred_sgb_log_abunds.unsqueeze(0),
            torch.log(true_sgb_abunds.unsqueeze(0)),
        ).item()
        
        # Root-Mean-Square-Log error
        log_10_conversion = np.log10(np.e)
        rmsle_loss = torch.sqrt(torch.mean(torch.square(
            log_10_conversion * (pred_sgb_log_abunds - torch.log(true_sgb_abunds))
        ))).item()

        # Spearman rank correlation
        spearman = scipy.stats.spearmanr(pred_sgb_log_abunds.numpy(), pred_sgb_log_abunds.numpy())
        
        df_entries.append({
            'SampleIdx': sample_idx,
            'KL_err': kl_loss,
            'RMSL_err': rmsle_loss,
            'SpearmanCorr': spearman.statistic,
            'NumSGB': len(true_sgb_abunds),
        })
    return pd.DataFrame(df_entries)
            

def perform_inference_torch_model(torch_model: nn.Module, dset: AbstractMetaphlanPreembeddedDataset, sample_idx: int, eval_device: str = 'cuda') -> Tensor:
    """
    :param torch_model: A pytorch model which takes as input a batched tensor of SGB embeddings, which was trained using the function 'main_training_loop'. This model should output logits.
    :param sample: A sample on which to perform inference on.
    """
    _, features, marker_mask, sgb_mask, _ = dset[sample_idx]
    features = features.unsqueeze(0).to(eval_device)
    marker_mask = marker_mask.unsqueeze(0).to(eval_device)
    sgb_mask = sgb_mask.unsqueeze(0).to(eval_device)

    torch_model.eval()
    with torch.no_grad():
        nn_output = torch_model(features, marker_mask, sgb_mask).to("cpu").to(torch.float32)
        nn_output = nn_output[0]
        n_finite_nn = (~torch.isinf(nn_output)).sum().item()
        n_finite_expected = torch.sum(sgb_mask).item()
        assert n_finite_expected == n_finite_nn, f"Expected {n_finite_expected} entries of model output to all be finite, got: {n_finite_nn}"

        gc.collect()
        torch.cuda.empty_cache()
        log_prob_output = nn.functional.log_softmax(nn_output, dim=-1)    # log probabilities
    return log_prob_output


def evaluate_torch_model(
    model_config_file: Path,
    model_state_file: Path,
    dset: AbstractMetaphlanPreembeddedDataset,
    device: str = 'cuda',
) -> pd.DataFrame:
    """ 
    A wrapper around the previously-defined 'evaluate_method' function.     (originally published in Vehtari et al 2019: https://arxiv.org/pdf/1903.08008.pdf)

    This function was written so that we can load the model just once from a previously-saved pytorch model state file.
    """
    assert model_config_file.exists(), f"Pytorch model config '{model_config_file}' does not exist!"
    assert model_state_file.exists(), f"Pytorch model state '{model_state_file}' does not exist! Was it trained and saved?"

    with open(model_config_file, "rt") as json_cfg:
        model_cfg = json.load(json_cfg)
        model_class_name = model_cfg['class']
        del model_cfg['class']  # unnecessary for evaluation.
        del model_cfg['init_rng_seed']  # unnecessary for evaluation.

    if model_class_name == "SGBAbundanceLayeredPredictionModel":
        model = SGBAbundanceLayeredPredictionModel(**model_cfg).to(device)
    elif model_class_name == "SGBAbundancePredictionModel":
        model = SGBAbundancePredictionModel(**model_cfg).to(device)
    model = torch.compile(model)
    model.load_state_dict(torch.load(model_state_file, weights_only=True))
    
    return evaluate_method(
        lambda _dset, _idx: perform_inference_torch_model(model, _dset, _idx, eval_device=device), 
        dset
    )


def evaluate_baseline_model(
    baseline_method: BaselinePredictor,
    dset_object: AbstractMetaphlanPreembeddedDataset,
    dset_df: pd.DataFrame,
) -> pd.DataFrame:
    def predict_fn(_dset, _idx):
        sample_id, _, _, sgb_padding_mask, _ = dset_object[_idx]
        subset_df = dset_df.loc[dset_df.index == sample_id]
        assert subset_df.shape[0] == 1, f"Expected 1 sample matching ID {sample_id}, got: {subset_df.shape[0]}"
        extractor = MetaphlanProfileParser(subset_df)
        target_sample = next(iter(extractor.samples()))
        
        abunds = baseline_method.predict_abundances(target_sample.sgb_ids, len(target_sample.sgb_ids))
        abunds = torch.from_numpy(abunds + 1e-8).to(torch.float32)  # add padding so that spurious "zeroes" don't result in NaNs or infs.
        abunds = abunds * sgb_padding_mask  # mask out SGBs not included in metric eval (because there is no PhyloPhlAn entry)
        abunds = abunds / abunds.sum()   # renormalize.
        return torch.log(abunds)

    return evaluate_method(predict_fn, dset_object)


# ==================================================== TEST EXECUTION =================

def main(model_options: List[str], plot_dir: Path, eval_device: str = 'cuda'):
    plot_dir.mkdir(exist_ok=True, parents=True)
    torch.set_float32_matmul_precision('high')
    train_df = pd.read_csv("/data/bwh-comppath-seq/youn/metaphlan_dset/model_training/train.tsv", sep="\t", index_col="SampleID")
    test_df = pd.read_csv("/data/bwh-comppath-seq/youn/metaphlan_dset/model_training/test.tsv", sep="\t", index_col="SampleID")

    test_dset = MetaphlanPreembeddedDatasetMemmapped(list(test_df.index))
    test_dset.load_memmap_tensors(Path("/data/bwh-comppath-seq/youn/metaphlan_dset/model_training/memmap_samples"))

    for model_option in model_options:
        if model_option == "evo-v1":
            model_basedir = Path("/data/bwh-comppath-seq/youn/metaphlan_dset/model_training/trained_model/evo_v1")
            evaluate_torch_model(
                model_config_file=model_basedir / "model_config.json",
                model_state_file=model_basedir / "model_weights.pt",
                dset=test_dset,
                device=eval_device,
            ).to_csv(plot_dir / "evo-v1.tsv", sep='\t')
        elif model_option == "evo-v2-d3-e100":
            model_basedir = Path("/data/bwh-comppath-seq/youn/metaphlan_dset/model_training/trained_model/evo_v2/depth3")
            evaluate_torch_model(
                model_config_file=model_basedir / "model_config.json",
                model_state_file=model_basedir / "model_weights.pt",
                dset=test_dset,
                device=eval_device,
            ).to_csv(plot_dir / 'evo-v2-d3-e100.tsv', sep='\t')
        elif model_option == "evo-v2-d3-e300":
            model_basedir = Path("/data/bwh-comppath-seq/youn/metaphlan_dset/model_training/trained_model/evo_v2/depth3_epoch300")
            evaluate_torch_model(
                model_config_file=model_basedir / "model_config.json",
                model_state_file=model_basedir / "model_weights.pt",
                dset=test_dset,
                device=eval_device,
            ).to_csv(plot_dir / 'evo-v2-d3-e300.tsv', sep='\t')
        elif model_option == "evo-v2-d5-e300":
            model_basedir = Path("/data/bwh-comppath-seq/youn/metaphlan_dset/model_training/trained_model/evo_v2/depth5_epoch300")
            evaluate_torch_model(
                model_config_file=model_basedir / "model_config.json",
                model_state_file=model_basedir / "model_weights.pt",
                dset=test_dset,
                device=eval_device,
            ).to_csv(plot_dir / 'evo-v2-d5-e300.tsv', sep='\t')
        elif model_option == "uniform":
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
            
                evaluate_baseline_model(
                    baseline_method=UniformAbundancePredictor(),
                    dset_object=test_dset,
                    dset_df=test_df,
                ).to_csv(plot_dir / 'uniform.tsv', sep='\t')
        elif model_option == "neighbor":
            nearest_neighbor_baseline_method = NearestNeighborAveragingPredictor(
                sgb_phylogenetic_tree_file=Path("/data/bwh-comppath-seq/youn/metaphlan_dset/metaphlan_database/mpa_vJan21_CHOCOPhlAnSGB_202103.nwk"),
                train_df=train_df,
            )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
            
                evaluate_baseline_model(
                    baseline_method=nearest_neighbor_baseline_method,
                    dset_object=test_dset,
                    dset_df=test_df,
                ).to_csv(plot_dir / 'neighbor.tsv', sep='\t')


if __name__ == "__main__":
    main(
        model_options=["evo-v2-d3-e100", "evo-v2-d3-e300", "evo-v2-d5-e300", "uniform", "neighbor"],
        plot_dir=Path("/data/bwh-comppath-seq/youn/metaphlan_dset/model_training/eval"),
        eval_device='cuda:0',
    )


    