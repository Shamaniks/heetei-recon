import numpy as np

from utils.npz_reader import NpzReader


def intensity_to_viridis(intensity: np.ndarray) -> np.ndarray:
    """
    Float32/uint8 intensity to viridis approximation.

    Parameters
    ----------
    intensity : np.ndarray, shape (N,), dtype float32 or uint8 or any other numeric
        The array of restored laser intensities for each point.

    Returns
    -------
    colors : np.ndarray, shape (N, 3), dtype float64
        RGB colors mapped to Viridis palette in range [0.0, 1.0].
    """
    intens_float = intensity.astype(np.float32)
    
    int_min = intens_float.min()
    int_max = intens_float.max()
    
    if np.isclose(int_max, int_min):
        t = np.zeros_like(intens_float)
    else:
        t = (intens_float - int_min) / (int_max - int_min)

    r = 0.267 - 0.482 * t + 4.145 * t**2 - 12.015 * t**3 + 14.341 * t**4 - 5.250 * t**5
    g = 0.004 + 1.411 * t + 2.285 * t**2 - 9.123 * t**3 + 9.873 * t**4 - 3.443 * t**5
    b = 0.329 + 1.384 * t - 1.915 * t**2 + 0.169 * t**3 + 0.655 * t**4 - 0.443 * t**5

    colors = np.clip(np.stack([r, g, b], axis=1), 0.0, 1.0).astype(np.float64)
    return colors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Converts intensity to approximate viridis [0.0, 1.0] colors for visualization"
    )
    parser.add_argument("--source", required=True,
                        help="Source file for convertation")
    parser.add_argument("--colors", default=None,
                        help="Output file colors")
    return parser.parse_args()


if __name__ == "__main__":
    with NpzReader(args.source) as nr:
        intensity = nr.get_intensity()
    print("[viridis] readed intensity")

    colors = intensity_to_viridis(intensity)
    print("[viridis] intensity converted to colors")

    np.savez_compressed(args.colors, colors=colors)
    print(f"[viridis] intensity array converted to {args.colors}")
