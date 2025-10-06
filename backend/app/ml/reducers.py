from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd


def optimal_components(X: pd.DataFrame, variance_threshold: float = 0.95, max_components: int = 50) -> int:
    from sklearn.decomposition import PCA
    n_samples, n_features = X.shape
    max_components = min(n_features, max_components)
    pca = PCA(n_components=min(n_samples, n_features, max_components))
    pca.fit(X)
    explained_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.argmax(explained_variance_ratio >= variance_threshold) + 1
    return max(1, min(n_components, max_components))


def apply_reduction(X: pd.DataFrame, method: str, n_components: int, target=None) -> Tuple[np.ndarray, Dict]:
    if method == 'pca':
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=n_components, random_state=42)
        reduced_data = reducer.fit_transform(X)
        return reduced_data, {
            'explained_variance_ratio': reducer.explained_variance_ratio_.tolist(),
            'singular_values': reducer.singular_values_.tolist()
        }
    if method == 'ica':
        from sklearn.decomposition import FastICA
        reducer = FastICA(n_components=n_components, random_state=42)
        return reducer.fit_transform(X), {}
    if method == 'tsne':
        from sklearn.manifold import TSNE
        if X.shape[1] > 50:
            from sklearn.decomposition import PCA
            X = PCA(n_components=min(50, X.shape[1])).fit_transform(X)
        tsne = TSNE(n_components=n_components, random_state=42)
        reduced_data = tsne.fit_transform(X)
        return reduced_data, {'kl_divergence': tsne.kl_divergence_}
    if method == 'umap':
        try:
            import umap
            reducer = umap.UMAP(n_components=n_components, random_state=42)
            return reducer.fit_transform(X), {}
        except Exception:
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=n_components, random_state=42)
            return reducer.fit_transform(X), {}
    if method == 'feature_selection':
        from sklearn.feature_selection import SelectKBest, f_classif, f_regression
        if target is not None:
            y = target
            selector = SelectKBest(f_regression if len(np.unique(y)) > 10 else f_classif, k=n_components)
            reduced_data = selector.fit_transform(X, y)
            selected_features = X.columns[selector.get_support()].tolist()
            return reduced_data, {'selected_features': selected_features}
    if method == 'selectfrommodel':
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
        from sklearn.feature_selection import SelectFromModel
        if target is not None:
            y = target
            estimator = RandomForestRegressor(n_estimators=100, random_state=42) if len(np.unique(y)) > 10 else RandomForestClassifier(n_estimators=100, random_state=42)
            selector = SelectFromModel(estimator=estimator, max_features=n_components)
            selector.fit(X, y)
            support = selector.get_support()
            selected_features = X.columns[support].tolist()
            return X[selected_features].values, {'selected_features': selected_features}
    return X.values, {}


