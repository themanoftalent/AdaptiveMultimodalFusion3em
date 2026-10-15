from __future__ import annotations

import numpy as np
from PIL import Image


class MacenkoNormalizer:
    """Mathematical SVD-based stain normalizer (Macenko et al., 2009).

    Transforms a target H&E stained histopathology image to match the stain
    characteristics (stain matrix and concentration distribution) of a reference image.
    """

    def __init__(self, reference_image_path: str | None = None, beta: float = 0.15, alpha: float = 1.0):
        self.beta = beta
        self.alpha = alpha

        # Default reference values (derived from typical TCIA H&E slide) if no reference is fitted
        self.reference_stain_matrix = np.array([
            [0.5626, 0.2137],
            [0.7201, 0.8012],
            [0.4062, 0.5580]
        ], dtype=np.float32)
        self.reference_max_concentrations = np.array([[1.9705], [1.0308]], dtype=np.float32)

        if reference_image_path:
            self.fit(reference_image_path)

    @staticmethod
    def _read_rgb(path: str) -> np.ndarray:
        return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)

    def fit(self, reference_image_path: str) -> None:
        """Fits the stain normalizer to a reference image."""
        try:
            image = self._read_rgb(reference_image_path)
            stain_matrix, max_concentrations = self.get_stain_matrix_and_concentrations(image)
            self.reference_stain_matrix = stain_matrix
            self.reference_max_concentrations = max_concentrations
        except Exception as e:
            print(f"Warning: Failed to fit reference image {reference_image_path} ({e}). Using default reference.")

    def get_stain_matrix_and_concentrations(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Extracts the stain matrix and maximum concentrations using SVD."""
        # 1. Convert RGB to Optical Density (OD)
        pixels = image.reshape(-1, 3).astype(np.float32)
        od = -np.log((pixels + 1.0) / 255.0)

        # 2. Filter background pixels (OD threshold beta)
        mask = (od > self.beta).any(axis=1)
        od_hat = od[mask]

        if len(od_hat) < 100:
            # Fallback if too few pixels are stained
            return self.reference_stain_matrix, self.reference_max_concentrations

        # 3. Calculate covariance matrix and find eigenvectors
        cov = np.cov(od_hat.T)
        _, evecs = np.linalg.eigh(cov)

        # eigh returns eigenvalues in ascending order, so the two largest eigenvectors are the last two.
        v1 = evecs[:, 2]
        v2 = evecs[:, 1]

        # 4. Project onto the plane defined by the two largest eigenvectors
        projected = np.dot(od_hat, np.stack([v1, v2], axis=1))

        # 5. Calculate angle of each projected pixel
        phi = np.arctan2(projected[:, 1], projected[:, 0])

        # 6. Find robust extreme angles (alpha-percentiles)
        min_phi = np.percentile(phi, self.alpha)
        max_phi = np.percentile(phi, 100.0 - self.alpha)

        # 7. Reconstruct extreme stain vectors in OD space
        stain_v1 = v1 * np.cos(min_phi) + v2 * np.sin(min_phi)
        stain_v2 = v1 * np.cos(max_phi) + v2 * np.sin(max_phi)

        # 8. Order stain vectors consistently: Hematoxylin (blue/purple) absorbs more red light (index 0) than Eosin (pink)
        if stain_v1[0] > stain_v2[0]:
            stain_matrix = np.stack([stain_v1, stain_v2], axis=1)
        else:
            stain_matrix = np.stack([stain_v2, stain_v1], axis=1)

        # Normalize stain vectors to unit length
        stain_matrix = stain_matrix / np.linalg.norm(stain_matrix, axis=0, keepdims=True)

        # 9. Compute concentrations C = W^-1 * OD
        pinv_W = np.linalg.pinv(stain_matrix)
        C = np.dot(pinv_W, od_hat.T)

        # Find 99th percentile of concentrations
        max_concentrations = np.percentile(C, 99, axis=1, keepdims=True)

        return stain_matrix, max_concentrations

    def transform_array(self, image: np.ndarray) -> np.ndarray:
        """Transforms a target image to match the reference stain characteristics."""
        shape = image.shape
        if len(shape) != 3 or shape[2] != 3:
            raise ValueError("Input image must be RGB with shape (H, W, 3).")

        try:
            # Get target stain matrix and concentrations
            stain_matrix, max_concentrations = self.get_stain_matrix_and_concentrations(image)

            # Convert full image to OD
            pixels = image.reshape(-1, 3).astype(np.float32)
            od = -np.log((pixels + 1.0) / 255.0)

            # Compute concentrations for all pixels
            pinv_W = np.linalg.pinv(stain_matrix)
            C = np.dot(pinv_W, od.T)

            # Normalize concentrations
            C_norm = C * (self.reference_max_concentrations / (max_concentrations + 1e-8))

            # Reconstruct to OD using reference stain matrix
            od_norm = np.dot(self.reference_stain_matrix, C_norm).T

            # Convert back to RGB
            normalized = np.exp(-od_norm) * 255.0 - 1.0
            normalized = np.clip(normalized, 0, 255).astype(np.uint8)
            return normalized.reshape(shape)

        except Exception as e:
            # Safe fallback: return original image if transformation fails
            print(f"Warning: Macenko transform failed ({e}). Returning original image.")
            return image.astype(np.uint8)

    def transform_file(self, input_path: str, output_path: str) -> None:
        image = self._read_rgb(input_path)
        normalized = self.transform_array(image)
        Image.fromarray(normalized).save(output_path)
