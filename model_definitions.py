import numpy as np

class CustomLDAClassifier:
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.V = None  # Ma trận chiếu
        self.mu_z = {}  # Kỳ vọng các lớp trong không gian mới
        self.Sigma_inv = None  # Nghịch đảo ma trận hiệp phương sai chung
        self.priors = {}  # Xác suất tiên nghiệm
        self.class_labels = None

    def fit(self, X, y):
        self.class_labels = np.unique(y)
        n_samples, n_features = X.shape
        
        # 1. Tính toán S_W và S_B
        mean_vectors = {}
        for cl in self.class_labels:
            mean_vectors[cl] = np.mean(X[y == cl], axis=0)
            
        overall_mean = np.mean(X, axis=0)
        
        S_W = np.zeros((n_features, n_features))
        for cl in self.class_labels:
            X_k = X[y == cl]
            mu_k = mean_vectors[cl]
            S_W += (X_k - mu_k).T @ (X_k - mu_k)
            
        S_B = np.zeros((n_features, n_features))
        for cl in self.class_labels:
            n_k = X[y == cl].shape[0]
            mu_k = mean_vectors[cl]
            diff = (mu_k - overall_mean).reshape(n_features, 1)
            S_B += n_k * (diff @ diff.T)
            
        # Giải bài toán trị riêng cho S_W^-1 S_B (thêm epsilon để S_W khả nghịch)
        epsilon = 1e-6
        S_W_reg = S_W + epsilon * np.eye(n_features)
        S_W_inv = np.linalg.inv(S_W_reg)
        M = S_W_inv @ S_B
        
        eigvals, eigvecs = np.linalg.eig(M)
        eigvals, eigvecs = eigvals.real, eigvecs.real
        
        # Sắp xếp trị riêng giảm dần
        idx = np.argsort(eigvals)[::-1]
        self.V = eigvecs[:, idx[:self.n_components]]
        
        # Chiếu dữ liệu sang không gian mới Z
        Z = X @ self.V
        
        # Tính kỳ vọng của các lớp trên Z và Ma trận hiệp phương sai chung Sigma
        Sigma = np.zeros((self.n_components, self.n_components))
        for cl in self.class_labels:
            Z_k = Z[y == cl]
            self.mu_z[cl] = np.mean(Z_k, axis=0)
            self.priors[cl] = Z_k.shape[0] / n_samples
            
            diff = Z_k - self.mu_z[cl]
            Sigma += diff.T @ diff
            
        # Ước lượng không chệch
        Sigma /= (n_samples - len(self.class_labels))
        # Thêm regularization cho Sigma để tránh suy biến
        Sigma += 1e-6 * np.eye(self.n_components)
        self.Sigma_inv = np.linalg.inv(Sigma)

    def discriminant_function(self, z, cl):
        mu_k = self.mu_z[cl]
        prior = self.priors[cl]
        # δ_k(z) = z^T Σ^-1 μ_k - 1/2 μ_k^T Σ^-1 μ_k + ln π_k
        term1 = z @ self.Sigma_inv @ mu_k
        term2 = -0.5 * (mu_k @ self.Sigma_inv @ mu_k)
        term3 = np.log(prior)
        return term1 + term2 + term3

    def predict(self, X):
        Z = X @ self.V
        preds = []
        for z in Z:
            scores = [self.discriminant_function(z, cl) for cl in self.class_labels]
            preds.append(self.class_labels[np.argmax(scores)])
        return np.array(preds)
