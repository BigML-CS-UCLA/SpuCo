import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from spuco.group_inference.george_utils.fast_sil import silhouette_samples

def get_cluster_sils(data, pred_labels, compute_sil=True, cuda=False):
    unique_preds = sorted(np.unique(pred_labels))
    SIL_samples = silhouette_samples(data, pred_labels, cuda=cuda) if compute_sil else np.zeros(
        len(data))
    SILs_by_cluster = {
        int(label): float(np.mean(SIL_samples[pred_labels == label]))
        for label in unique_preds
    }
    SIL_global = float(np.mean(SIL_samples))
    return SILs_by_cluster, SIL_global


def compute_group_sizes(labels):
    result = dict(sorted(zip(*np.unique(labels, return_counts=True))))
    return {int(k): int(v) for k, v in result.items()}

class AutoKMixtureModel:
    def __init__(self, cluster_method, max_k, n_init=3, seed=None, sil_cuda=False, verbose=0,
                 search=True):
        if cluster_method == 'kmeans':
            cluster_cls = KMeans
            k_name = 'n_clusters'
        elif cluster_method == 'gmm':
            cluster_cls = GaussianMixture
            k_name = 'n_components'
        else:
            raise ValueError('Unsupported clustering method')

        self.cluster_cls = cluster_cls
        self.k_name = k_name
        self.search = search
        self.max_k = max_k
        self.n_init = n_init
        self.seed = seed
        self.sil_cuda = sil_cuda
        self.verbose = verbose

    def gen_inner_cluster_obj(self, k):
        # Return a clustering object according to the specified parameters
        return self.cluster_cls(**{self.k_name: k}, n_init=self.n_init, random_state=self.seed,
                                verbose=self.verbose)

    def fit(self, activ):
        best_score = -2
        k_min = 2 if self.search else self.max_k
        search = self.search and k_min != self.max_k
        for k in range(k_min, self.max_k + 1):
            cluster_obj = self.gen_inner_cluster_obj(k)
            pred_labels = cluster_obj.fit_predict(activ)
            if search:
                local_sils, global_sil = get_cluster_sils(activ, pred_labels, compute_sil=True,
                                                          cuda=self.sil_cuda)
                clustering_score = np.mean(list(local_sils.values()))
                if clustering_score >= best_score:
                    best_score = clustering_score
                    best_model = cluster_obj
                    best_k = k
            else:
                best_score, best_model, best_k = 0, cluster_obj, self.max_k

        self.best_k = best_k
        self.n_clusters = best_k
        self.best_score = best_score
        self.cluster_obj = best_model
        return self

    def predict(self, activ):
        return self.cluster_obj.predict(activ)

    def fit_predict(self, activ):
        self.fit(activ)
        return self.predict(activ)

    def predict_proba(self, X):
        return self.cluster_obj.predict_proba(X)

    def score(self, X):
        return self.cluster_obj.score(X)


class OverclusterModel:
    def __init__(self, cluster_method, max_k, oc_fac=5, n_init=3, search=True, sil_threshold=0.,
                 seed=None, sil_cuda=False, verbose=0, sz_threshold_pct=0.005, sz_threshold_abs=25):
        self.base_model = AutoKMixtureModel(cluster_method, max_k, n_init, seed, sil_cuda, verbose,
                                            search)
        self.oc_fac = oc_fac
        self.sil_threshold = sil_threshold
        self.sz_threshold_pct = sz_threshold_pct
        self.sz_threshold_abs = sz_threshold_abs
        self.requires_extra_info = True

    def get_oc_predictions(self, activ, orig_preds):
        # Split each cluster from base_model into sub-clusters, and save each of the
        # associated sub-clustering predictors in self.cluster_objs.
        # Collate and return the new predictions in oc_preds and val_oc_preds.
        self.cluster_objs = []
        oc_preds = np.zeros(len(activ), dtype=int)

        for i in self.pred_vals:
            sub_activ = activ[orig_preds == i]
            cluster_obj = self.base_model.gen_inner_cluster_obj(self.oc_fac).fit(sub_activ)
            self.cluster_objs.append(cluster_obj)
            sub_preds = cluster_obj.predict(sub_activ) + self.oc_fac * i
            oc_preds[orig_preds == i] = sub_preds

        return oc_preds

    def filter_overclusters(self, activ, orig_preds, oc_preds):
        # Keep an overcluster if its point have higher SIL than before
        # overclustering, AND it has higher average loss than the
        # original cluster, AND it contains sufficiently many training and
        # validation points.

        num_oc = np.amax(oc_preds) + 1
        # Compute original per-cluster SIL scores,
        # and the SIL scores after overclustering.
        orig_sample_sils = silhouette_samples(activ, orig_preds, cuda=self.sil_cuda)
        new_sample_sils = silhouette_samples(activ, oc_preds, cuda=self.sil_cuda)

        oc_orig_sils = [np.mean(orig_sample_sils[oc_preds == i]) for i in range(num_oc)]
        oc_new_sils = [np.mean(new_sample_sils[oc_preds == i]) for i in range(num_oc)]

        # Count number of points in each cluster after overclustering. Drop tiny clusters as these
        # will lead to unreliable optimization.
        oc_counts = np.bincount(oc_preds)

        tr_sz_threshold = max(len(activ) * self.sz_threshold_pct, self.sz_threshold_abs)

        # Decide which overclusters to keep
        oc_to_keep = []
        for i in range(num_oc):
            if oc_new_sils[i] > max(oc_orig_sils[i], self.sil_threshold) and \
              oc_counts[i] >= tr_sz_threshold:
                oc_to_keep.append(i)

        return oc_to_keep

    def create_label_map(self, num_orig_preds, oc_to_keep):
        # Map raw overclustering outputs to final "cluster labels," accounting for the
        # fact that some overclusters are re-merged.
        label_map = {}
        cur_cluster_ind = -1
        for i in range(num_orig_preds):
            # For each original cluster, if there were no
            # overclusters kept within it, keep the original cluster as-is.
            # Otherwise, it needs to be split.
            keep_all = True  # If we keep all overclusters, we can discard the original cluster
            for j in range(self.oc_fac):
                index = i * self.oc_fac + j
                if index not in oc_to_keep:
                    keep_all = False
            if not keep_all:
                cur_cluster_ind += 1

            # Updated cluster index corresponding to original cluster
            # (points in the original cluster assigned to a non-kept overcluster
            # are merged into this cluster)
            base_index = cur_cluster_ind
            for j in range(self.oc_fac):
                index = i * self.oc_fac + j
                if index in oc_to_keep:
                    cur_cluster_ind += 1
                    oc_index = cur_cluster_ind
                else:
                    assert (not keep_all)
                    oc_index = base_index
                label_map[index] = oc_index
        return label_map

    def fit(self, activ):
        orig_preds = self.base_model.fit_predict(activ)
        self.pred_vals = sorted(np.unique(orig_preds))
        num_orig_preds = len(self.pred_vals)
        oc_fac = self.oc_fac
        num_oc = num_orig_preds * oc_fac

        oc_preds = self.get_oc_predictions(activ, orig_preds)
        oc_to_keep = self.filter_overclusters(activ, orig_preds, oc_preds)
        self.label_map = self.create_label_map(num_orig_preds, oc_to_keep)

        new_preds = np.zeros(len(activ), dtype=int)
        for i in range(num_oc):
            new_preds[oc_preds == i] = self.label_map[i]

        self.n_clusters = max(self.label_map.values()) + 1  # Final number of output predictions
        return self

    def predict(self, activ):
        # Get clusters from base model
        base_preds = self.base_model.predict(activ)
        # Get overclusters
        oc_preds = np.zeros(len(activ), dtype=int)
        for i in self.pred_vals:
            subfeats = activ[base_preds == i]
            subpreds = self.cluster_objs[i].predict(subfeats) + self.oc_fac * i
            oc_preds[base_preds == i] = subpreds

        # Merge overclusters appropriately and return final predictions
        new_preds = np.zeros(len(activ), dtype=int)
        for i in range(len(self.pred_vals) * self.oc_fac):
            new_preds[oc_preds == i] = self.label_map[i]
        return new_preds

    @property
    def sil_cuda(self):
        return self.base_model.sil_cuda

    @property
    def n_init(self):
        return self.base_model.n_init

    @property
    def seed(self):
        return self.base_model.seed
