from collections import Counter
import numpy as np


# --- Nominal encoding (Algorithm 2, CatEncoder) ----------------------------
class JointOneHot:
    def __init__(self, categorical_features):
        self.categorical_features = list(categorical_features)
        self.categories_ = {}

    def fit(self, x_all):
        # Categories are learned on the COMBINED minority + majority rows so the
        for j in self.categorical_features:
            column = x_all[:, j]
            self.categories_[j] = sorted({v for v in column})
        return self

    def transform_nominal(self, x, weight):
        blocks = []
        for j in self.categorical_features:
            categories = self.categories_[j]
            onehot = np.zeros((x.shape[0], len(categories)), dtype=float)
            index = {cat: k for k, cat in enumerate(categories)}
            for row in range(x.shape[0]):
                onehot[row, index[x[row, j]]] = weight
            blocks.append(onehot)
        if not blocks:
            return np.empty((x.shape[0], 0), dtype=float)
        return np.hstack(blocks)


def _sigma_med(x_min_continuous):
    if x_min_continuous.shape[1] == 0:
        return 0.0
    stds = x_min_continuous.std(axis=0, ddof=0)
    return float(np.median(stds))


def _encode(x, continuous_idx, encoder, weight):
    continuous = x[:, continuous_idx].astype(float) if continuous_idx else np.empty(
        (x.shape[0], 0), dtype=float)
    nominal = encoder.transform_nominal(x, weight)
    return np.hstack([continuous, nominal])


# --- Geometric generation (Algorithm 3) ------------------------------------
def _hyperball(dimension, rng):
    if dimension == 0:
        return np.empty(0, dtype=float)
    v = rng.normal(size=dimension)
    norm = np.linalg.norm(v)
    if norm == 0.0:
        v[0], norm = 1.0, 1.0
    radius = rng.uniform() ** (1.0 / dimension)
    return radius * v / norm


def _geometric_continuous(center, neighbor, truncation, deformation, rng):
    sample = _hyperball(center.shape[0], rng)

    direction = neighbor - center
    distance = float(np.linalg.norm(direction))

    if distance == 0.0:
        return center.copy()

    unit_direction = direction / distance

    parallel_amount = float(np.dot(sample, unit_direction))
    parallel_component = parallel_amount * unit_direction
    perpendicular_component = sample - parallel_component

    if abs(truncation - parallel_amount) > 1.0:
        sample = sample - 2.0 * parallel_component

    sample = sample - deformation * perpendicular_component

    return center + distance * sample


# --- Selection mechanism (Algorithm 2, Surface) ----------------------------
def _knn_indices(query, pool, k):
    if len(pool) == 0:
        return np.empty(0, dtype=int)
    distances = np.linalg.norm(pool - query, axis=1)
    order = np.argsort(distances, kind="stable")
    return order[:min(k, len(pool))]


class GSMOTENC:
    def __init__(self, categorical_features, k_neighbors=5, selection_strategy="combined",
                 truncation_factor=1.0, deformation_factor=0.5, random_state=None):
        assert selection_strategy in {"minority", "majority", "combined"}, \
            "selection_strategy must be minority, majority, or combined"
        assert -1.0 <= truncation_factor <= 1.0, "truncation_factor in [-1, 1]"
        assert 0.0 <= deformation_factor <= 1.0, "deformation_factor in [0, 1]"
        self.categorical_features = list(categorical_features)
        self.k_neighbors = int(k_neighbors)
        self.selection_strategy = selection_strategy
        self.truncation_factor = float(truncation_factor)
        self.deformation_factor = float(deformation_factor)
        self.random_state = random_state

    def _split_columns(self, n_columns):
        categorical = set(self.categorical_features)
        continuous_idx = [j for j in range(n_columns) if j not in categorical]
        return continuous_idx

    def _generate_for_class(self, minority, majority, n_samples, cont_idx, rng):
        if n_samples <= 0:
            return np.empty((0, minority.shape[1]), dtype=object)

        encoder = JointOneHot(self.categorical_features).fit(
            np.vstack([minority, majority]) if len(majority) else minority
        )

        median_sigma = _sigma_med(
            minority[:, cont_idx].astype(float)
            if cont_idx
            else np.empty((len(minority), 0))
        )
        cat_weight = median_sigma / 2.0

        minority_enc = _encode(minority, cont_idx, encoder, cat_weight)
        majority_enc = (
            _encode(majority, cont_idx, encoder, cat_weight)
            if len(majority)
            else np.empty((0, minority_enc.shape[1]))
        )

        synthetic_samples = []

        for _ in range(n_samples):
            center_idx = rng.integers(len(minority))
            center_enc = minority_enc[center_idx]

            minority_nn = _knn_indices(
                center_enc, minority_enc, self.k_neighbors + 1
            )
            minority_nn = (
                np.array(
                    [idx for idx in minority_nn if idx !=
                        center_idx][: self.k_neighbors]
                )
                if len(minority_nn)
                else minority_nn
            )

            if len(minority_nn) == 0:
                minority_nn = np.array([center_idx])

            if self.selection_strategy == "minority":
                neighbor_pool = minority_nn
                neighbor_idx = rng.integers(len(neighbor_pool))

                neighbor_cont = minority[
                    neighbor_pool[neighbor_idx], cont_idx
                ].astype(float)

                minority_rows = minority[neighbor_pool]
                majority_rows = np.empty((0, minority.shape[1]), dtype=object)

            elif self.selection_strategy == "majority":
                majority_nn = _knn_indices(center_enc, majority_enc, 1)

                if len(majority_nn) == 0:
                    neighbor_cont = minority[minority_nn[0], cont_idx].astype(
                        float)
                    minority_rows = minority[minority_nn]
                    majority_rows = np.empty(
                        (0, minority.shape[1]), dtype=object)
                else:
                    neighbor_cont = majority[
                        majority_nn[0], cont_idx
                    ].astype(float)

                    minority_rows = np.empty(
                        (0, minority.shape[1]), dtype=object)
                    majority_rows = majority[majority_nn]

            else:
                majority_nn = _knn_indices(center_enc, majority_enc, 1)

                minority_choice = minority_nn[rng.integers(len(minority_nn))]
                candidate_enc = minority_enc[minority_choice]

                if len(majority_nn) == 0:
                    neighbor_cont = minority[
                        minority_choice, cont_idx
                    ].astype(float)
                else:
                    minority_dist = np.linalg.norm(candidate_enc - center_enc)
                    majority_dist = np.linalg.norm(
                        majority_enc[majority_nn[0]] - center_enc
                    )

                    if majority_dist < minority_dist:
                        neighbor_cont = majority[
                            majority_nn[0], cont_idx
                        ].astype(float)
                    else:
                        neighbor_cont = minority[
                            minority_choice, cont_idx
                        ].astype(float)

                minority_rows = minority[minority_nn]
                majority_rows = (
                    majority[majority_nn]
                    if len(majority_nn)
                    else np.empty((0, minority.shape[1]), dtype=object)
                )

            center_cont = (
                minority[center_idx, cont_idx].astype(float)
                if cont_idx
                else np.empty(0)
            )

            synthetic_cont = _geometric_continuous(
                center_cont,
                neighbor_cont,
                self.truncation_factor,
                self.deformation_factor,
                rng,
            )

            neighbor_set = (
                np.vstack([minority_rows, majority_rows])
                if len(majority_rows)
                else minority_rows
            )

            sample = np.empty(minority.shape[1], dtype=object)

            for offset, feature_idx in enumerate(cont_idx):
                sample[feature_idx] = float(synthetic_cont[offset])

            for feature_idx in self.categorical_features:
                values = [neighbor_set[row_idx, feature_idx]
                          for row_idx in range(len(neighbor_set))]
                sample[feature_idx] = Counter(values).most_common(1)[0][0]

            synthetic_samples.append(sample)

        return np.array(synthetic_samples, dtype=object)

    def fit_resample(self, X, y):
        X = np.asarray(X, dtype=object)
        y = np.asarray(y)
        assert X.shape[0] == y.shape[0], "X and y length mismatch"
        n_columns = X.shape[1]
        continuous_idx = self._split_columns(n_columns)
        rng = np.random.default_rng(self.random_state)

        counts = Counter(y.tolist())
        majority_label, majority_count = counts.most_common(1)[0]
        x_maj = X[y == majority_label]

        new_rows, new_labels = [X], [y]
        for label, count in counts.items():
            if label == majority_label:
                continue
            x_min = X[y == label]
            n_needed = majority_count - count
            generated = self._generate_for_class(
                x_min, x_maj, n_needed, continuous_idx, rng)
            if len(generated):
                new_rows.append(generated)
                new_labels.append(np.array([label] * len(generated)))

        X_resampled = np.vstack(new_rows)
        y_resampled = np.concatenate(new_labels)
        return X_resampled, y_resampled


# --- Small self-check demo (ASCII only) ------------------------------------
def main():
    rng = np.random.default_rng(0)
    n_maj, n_min = 200, 40
    cont_maj = rng.normal(0.0, 1.0, size=(n_maj, 2))
    cont_min = rng.normal(3.0, 1.0, size=(n_min, 2))
    nom_maj = rng.choice(["x", "y"], size=(n_maj, 1))
    nom_min = rng.choice(["y", "z"], size=(n_min, 1))
    X = np.vstack([
        np.hstack([cont_maj.astype(object), nom_maj]),
        np.hstack([cont_min.astype(object), nom_min]),
    ])
    y = np.array(["maj"] * n_maj + ["min"] * n_min)

    sampler = GSMOTENC(categorical_features=[2], k_neighbors=5,
                       selection_strategy="combined", random_state=0)
    X_res, y_res = sampler.fit_resample(X, y)
    counts = Counter(y_res.tolist())
    print(f"Before: {Counter(y.tolist())}")
    print(f"After:  {counts}")
    print(f"Balanced: {counts['maj'] == counts['min']}")


if __name__ == "__main__":
    main()
