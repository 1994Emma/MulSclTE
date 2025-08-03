from scipy import signal
import numpy as np
import ruptures as rpt


class ChangePointDetector:
    """
    A utility class for detecting change points in a sequence using
    peak detection (scipy.signal) or change point detection (ruptures).
    """
    # Supported methods for peak-based and ruptures-based detection
    valid_peaks_methods = ["argrelextrema", "peaks", "peask_cwt"]
    valid_change_points_methods = ["rbf", "binary_seg_search", "win_search", "dynp"]

    def __init__(self):
        pass

    @classmethod
    def detect_change_points(cls, points_seq, dcp_method, win_size=40, n_bkps=10, min_size=30, jump=5, pen=10):
        """
        Unified interface for change point detection.

        Args:
            points_seq (np.ndarray or list): Input sequence of scores.
            dcp_method (str): Method to use; one of the supported peak or change point methods.
            win_size (int): Window size or smoothing order for peak detection.
            n_bkps (int): Number of breakpoints to detect (used in ruptures-based methods).
            min_size (int): Minimum segment length (ruptures).
            jump (int): Subsampling interval (ruptures).
            pen (float): Penalty value for 'rbf' model in ruptures. Larger value results in fewer segments.

        Returns:
            List[int]: Indices of detected change points.
        """
        assert (dcp_method in cls.valid_peaks_methods) or (
                dcp_method in cls.valid_change_points_methods), \
            f"Illegal dcp_method '{dcp_method}'. Must be one of {cls.valid_peaks_methods + cls.valid_change_points_methods}."

        if not isinstance(points_seq, np.ndarray):
            points_seq = np.array(points_seq)

        if dcp_method in cls.valid_peaks_methods:
            return cls.detect_change_points_by_signal_pkg(points_seq, dcp_method, order=win_size)
        else:
            return cls.detect_change_points_by_rpt_pkg(points_seq, dcp_method, n_bkps, pen, win_size, min_size, jump)

    @classmethod
    def detect_change_points_by_signal_pkg(cls, seq, peak_detect_method, order):
        """
        Detects peaks using scipy.signal.

        Args:
            seq (np.ndarray): Input sequence.
            peak_detect_method (str): Peak detection method: 'argrelextrema', 'peaks', or 'peask_cwt'.
            order (int): Smoothing window size or distance for peak detection.

        Returns:
            List[int]: Indices of detected peaks.
        """
        if peak_detect_method == "argrelextrema":
            idxes = signal.argrelextrema(seq, np.greater, order=order)[0].tolist()
        elif peak_detect_method == "peaks":
            idxes, _ = signal.find_peaks(seq, distance=order)
        elif peak_detect_method == "peask_cwt":
            idxes = signal.find_peaks_cwt(seq, np.arange(1, order))
        else:
            raise ValueError(f"Unsupported peak detection method: {peak_detect_method}")

        return idxes

    @classmethod
    def detect_change_points_by_rpt_pkg(cls, points, dcp_method, n_bkps=10, pen=10, win_size=40, min_size=30, jump=5):
        """
        Detects change points using the ruptures package.

        Args:
            points (np.ndarray): Input sequence.
            dcp_method (str): Detection method: 'rbf', 'binary_seg_search', 'win_search', or 'dynp'.
            n_bkps (int): Number of breakpoints to find.
            pen (float): Penalty for 'rbf' model.
            win_size (int): Window width for window-based method.
            min_size (int): Minimum segment length.
            jump (int): Subsampling rate.

        Returns:
            List[int]: Detected change points (end of segments).
        """
        if dcp_method == "rbf":
            algo = rpt.Pelt(model="rbf", min_size=min_size, jump=jump).fit(points)
            change_points = algo.predict(pen=pen)
        elif dcp_method == "binary_seg_search":
            algo = rpt.Binseg(model="l2", min_size=min_size, jump=jump).fit(points)
            change_points = algo.predict(n_bkps=n_bkps)
        elif dcp_method == "win_search":
            algo = rpt.Window(width=win_size, model="l2", min_size=min_size, jump=jump).fit(points)
            change_points = algo.predict(n_bkps=n_bkps)
        elif dcp_method == "dynp":
            algo = rpt.Dynp(model="l2", min_size=min_size, jump=jump).fit(points)
            change_points = algo.predict(n_bkps=n_bkps)
        else:
            raise ValueError(f"Unsupported change point detection method: {dcp_method}")

        return change_points
