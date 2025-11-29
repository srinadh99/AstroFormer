import os
from typing import Optional, Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.proportion import proportion_confint


class ConformalCPAnalysis:
    """Conformal Prediction analysis (global CP and Mondrian CP).

    This class expects:
      - calibration_data: DataFrame with columns ['true_label', class_1, ..., class_K]
      - test_data: DataFrame with the same columns

    It can:
      - calibrate global or Mondrian CP
      - compute coverage, set-size, singleton, F1, and ROC-style curves
      - plot and optionally save all figures.
    """

    def __init__(
        self,
        cp_mode: str = "mondrian",
        nonconf_type: str = "baseline",
        alphas_cov: Optional[np.ndarray] = None,
        alphas_roc: Optional[np.ndarray] = None,
    ):
        if cp_mode not in ("mondrian", "global"):
            raise ValueError("cp_mode must be 'mondrian' or 'global'")
        self.cp_mode = cp_mode
        self.nonconf_type = nonconf_type

        # alpha grid for coverage/set-size/F1/singleton
        self.alphas_cov = (
            np.linspace(0.01, 0.99, 20) if alphas_cov is None else np.asarray(alphas_cov)
        )
        # separate alpha grid for ROC if you want
        self.alphas_roc = (
            np.linspace(0.0, 0.99, 20) if alphas_roc is None else np.asarray(alphas_roc)
        )

        # data will be set with fit()
        self.calibration_data: Optional[pd.DataFrame] = None
        self.test_data: Optional[pd.DataFrame] = None
        self.labels: Optional[List[str]] = None

        # results, to be filled by computations
        self.conditional_coverage_: Optional[Dict[str, List[float]]] = None
        self.marginal_coverage_: Optional[np.ndarray] = None
        self.conditional_setsize_: Optional[Dict[str, List[float]]] = None
        self.marginal_setsize_: Optional[np.ndarray] = None
        self.singleton_rate_: Optional[np.ndarray] = None
        self.f1_macro_: Optional[np.ndarray] = None
        self.fpr_: Optional[np.ndarray] = None
        self.tpr_: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------
    def fit(self, calibration_data: pd.DataFrame, test_data: pd.DataFrame):
        """Attach calibration and test data, and infer label columns."""
        if "true_label" not in calibration_data.columns:
            raise ValueError("calibration_data must have a 'true_label' column")
        if "true_label" not in test_data.columns:
            raise ValueError("test_data must have a 'true_label' column")

        self.calibration_data = calibration_data.copy()
        self.test_data = test_data.copy()
        self.labels = [c for c in calibration_data.columns if c != "true_label"]
        return self

    # ------------------------------------------------------------------
    # Public driver
    # ------------------------------------------------------------------
    def run_full_analysis(
        self,
        example_alpha: float = 0.1,
        cutoff: int = 3,
        save_dir: Optional[str] = None,
        show_plots: bool = True,
    ):
        """Run a full CP analysis: example, coverage, set-size, ROC + plots.

        Parameters
        ----------
        example_alpha : float
            Alpha level used for the single example printout and set-size stats.
        cutoff : int
            How many smallest alphas to ignore on one end in set-size plot.
        save_dir : str or None
            If not None, all plots are saved to this directory.
        show_plots : bool
            If True, show plots. If False, only save them.
        """
        if self.calibration_data is None or self.test_data is None:
            raise RuntimeError("Call .fit(calibration_data, test_data) first.")

        # -------------------------------
        # 1) Example at example_alpha + q_hat
        # -------------------------------
        qhat_example = self._calibrate(self.calibration_data, alpha=example_alpha)
        row0 = self.test_data.iloc[0]
        pred_set0 = self._predict_set(row0, qhat_example)
        argmax_pred = row0[self.labels].astype(float).idxmax()

        print(f"CP mode:          {self.cp_mode}")
        print(f"Example alpha:    {example_alpha}")
        print("True label:       ", row0["true_label"])
        print("Argmax prediction:", argmax_pred)
        print("CP prediction set:", pred_set0)
        print("Set size:         ", len(pred_set0))
        print("-" * 60)

        # Print q_hat thresholds
        print(f"q_hat thresholds at alpha = {example_alpha}:")
        for lab in self.labels:
            q_val = qhat_example.get(lab, None)
            if q_val is None:
                continue
            print(f"  {lab}: {q_val:.6f}")
        print("-" * 60)

        # ---------------------------------------------
        # 2) Set-size distribution at example_alpha
        #    + two examples for each size 0, 1, 2
        # ---------------------------------------------
        size_to_indices: Dict[int, List[int]] = {}
        for idx, row in self.test_data.iterrows():
            pred_set = self._predict_set(row, qhat_example)
            s = len(pred_set)
            if s not in size_to_indices:
                size_to_indices[s] = []
            size_to_indices[s].append(idx)

        n_test = len(self.test_data)
        print(f"Set-size distribution on test data at alpha = {example_alpha}:")
        for size in sorted(size_to_indices.keys()):
            count = len(size_to_indices[size])
            print(
                f"  Set size {size}: {count} / {n_test} "
                f"({count / n_test:.2%} of test points)"
            )

        # Print up to two examples for each of set sizes 0, 1, 2 (if they exist)
        n_classes = len(self.labels)

        for target_size in range(0, n_classes + 1):
            if target_size not in size_to_indices:
                continue
        
            print(f"\nExamples with prediction set size = {target_size}:")
            example_indices = size_to_indices[target_size][:2]  # at most two
            for idx in example_indices:
                row = self.test_data.loc[idx]
                pred_set = self._predict_set(row, qhat_example)
                argmax_pred_row = row[self.labels].astype(float).idxmax()
                print(
                    f"  Test index {idx}: "
                    f"true_label={row['true_label']}, "
                    f"argmax_pred={argmax_pred_row}, "
                    f"pred_set={pred_set}"
                )

        print("-" * 60)

        # ---------------------------------------------
        # 3) Curves: coverage, set size, singletons, F1
        # ---------------------------------------------
        self._compute_coverage_setsize_singleton_f1()

        # -----------------
        # 4) ROC-style TPR/FPR
        # -----------------
        self._compute_roc()

        # -----------------
        # 5) Optional save dir
        # -----------------
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            coverage_path = os.path.join(save_dir, f"coverage_{self.cp_mode}.png")
            setsize_path = os.path.join(save_dir, f"setsize_{self.cp_mode}.png")
            singleton_path = os.path.join(save_dir, f"singleton_{self.cp_mode}.png")
            f1_path = os.path.join(save_dir, f"f1_{self.cp_mode}.png")
            roc_path = os.path.join(save_dir, f"roc_{self.cp_mode}.png")
        else:
            coverage_path = setsize_path = singleton_path = f1_path = roc_path = None

        # -----------------
        # 6) Plots
        # -----------------
        print("Generating coverage plot...")
        self.plot_coverage(save_path=coverage_path, show=show_plots)

        print("Generating set-size plot...")
        self.plot_setsize(cutoff=cutoff, save_path=setsize_path, show=show_plots)

        # Uncomment if you want these as well:
        # print("Generating singleton-rate plot...")
        # self.plot_singleton(save_path=singleton_path, show=show_plots)

        # print("Generating F1-macro plot...")
        # self.plot_f1(save_path=f1_path, show=show_plots)

        print("Generating ROC-style plot...")
        self.plot_roc(save_path=roc_path, show=show_plots)

        return {
            "coverage_path": coverage_path,
            "setsize_path": setsize_path,
            # "singleton_path": singleton_path,
            # "f1_path": f1_path,
            "roc_path": roc_path,
        }

    # ------------------------------------------------------------------
    # Nonconformity and calibration
    # ------------------------------------------------------------------
    def _get_nonconformity_dict(self) -> Dict[str, callable]:
        """Return a dict mapping label -> nonconformity function A(row)."""
        if self.nonconf_type != "baseline":
            raise NotImplementedError("Only baseline nonconformity is implemented.")
        if self.labels is None:
            raise RuntimeError("Labels not set. Did you call fit()?")

        nonconf = {}
        for lab in self.labels:
            nonconf[lab] = (lambda row, lab=lab: 1.0 - float(row[lab]))
        return nonconf

    def _calibrate(self, calibration_data: pd.DataFrame, alpha: float) -> Dict[str, float]:
        """Calibrate thresholds for global or Mondrian CP.

        Returns
        -------
        qhat_dict : dict[label -> threshold]
        """
        labels = [c for c in calibration_data.columns if c != "true_label"]
        nonconf = self._get_nonconformity_dict()

        if self.cp_mode == "mondrian":
            # per-class (Mondrian)
            scores_by_label: Dict[str, List[float]] = {lab: [] for lab in labels}
            for _, row in calibration_data.iterrows():
                true_lab = row["true_label"]
                if true_lab not in nonconf:
                    continue
                a = nonconf[true_lab](row)
                scores_by_label[true_lab].append(a)

            qhat_dict: Dict[str, float] = {}
            for lab, scores in scores_by_label.items():
                scores_arr = np.asarray(scores, dtype=float)
                n_lab = len(scores_arr)
                if n_lab == 0:
                    qhat_dict[lab] = 1.0
                    continue
                scores_sorted = np.sort(scores_arr)
                k = int(np.ceil((n_lab + 1) * (1.0 - alpha)))
                if k > n_lab:
                    qhat_dict[lab] = 1.0
                else:
                    qhat_dict[lab] = float(scores_sorted[k - 1])
            return qhat_dict

        elif self.cp_mode == "global":
            # global thresholds
            all_scores: List[float] = []
            for _, row in calibration_data.iterrows():
                true_lab = row["true_label"]
                if true_lab not in nonconf:
                    continue
                a = nonconf[true_lab](row)
                all_scores.append(a)

            scores_arr = np.asarray(all_scores, dtype=float)
            n = len(scores_arr)
            if n == 0:
                raise ValueError("No valid calibration scores.")

            scores_sorted = np.sort(scores_arr)
            k = int(np.ceil((n + 1) * (1.0 - alpha)))
            if k > n:
                q_scalar = 1.0
            else:
                q_scalar = float(scores_sorted[k - 1])

            return {lab: q_scalar for lab in labels}

        else:
            raise ValueError("cp_mode must be 'mondrian' or 'global'.")

    # ------------------------------------------------------------------
    # Prediction for one example
    # ------------------------------------------------------------------
    def _predict_set(self, row: pd.Series, qhat_dict: Dict[str, float]) -> List[str]:
        """Return CP prediction set for a single example (row)."""
        if self.labels is None:
            raise RuntimeError("Labels not set. Did you call fit()?")

        nonconf = self._get_nonconformity_dict()
        pred_set: List[str] = []
        for lab in self.labels:
            if lab not in qhat_dict:
                continue
            a = nonconf[lab](row)
            # IMPORTANT: use <= for proper conformal coverage guarantees
            if a <= qhat_dict[lab]:
                pred_set.append(lab)
        return pred_set

    # ------------------------------------------------------------------
    # Coverage, set-size, singletons, F1
    # ------------------------------------------------------------------
    def _compute_coverage_setsize_singleton_f1(self):
        """Compute conditional/marginal coverage, set size, singleton rate,
        and macro-F1 as functions of alpha (alphas_cov grid)."""
        if self.calibration_data is None or self.test_data is None:
            raise RuntimeError("Call fit() first.")
        if self.labels is None:
            raise RuntimeError("Labels not set. Did you call fit()?")

        labels = self.labels
        alphas = self.alphas_cov
        test_data = self.test_data

        cond_cov = {lab: [] for lab in labels}
        cond_size = {lab: [] for lab in labels}
        marg_cov: List[float] = []
        marg_size: List[float] = []
        singleton_rate: List[float] = []
        f1_macro: List[float] = []

        N_test = len(test_data)

        for a in alphas:
            # per-label stats
            correct_per_label = {lab: 0 for lab in labels}
            setsize_per_label = {lab: 0 for lab in labels}

            # TP/FP/FN/TN for F1
            tp = {lab: 0 for lab in labels}
            fp = {lab: 0 for lab in labels}
            fn = {lab: 0 for lab in labels}
            tn = {lab: 0 for lab in labels}

            # singleton counter (marginal)
            singleton_count = 0

            qhat_dict = self._calibrate(self.calibration_data, alpha=a)

            for _, row in test_data.iterrows():
                true_lab = row["true_label"]
                pred_set = self._predict_set(row, qhat_dict)

                # coverage and set size, per true label
                if true_lab in pred_set:
                    correct_per_label[true_lab] += 1
                setsize_per_label[true_lab] += len(pred_set)

                # singleton indicator (size == 1, regardless of correctness)
                if len(pred_set) == 1:
                    singleton_count += 1

                # TP/FP/FN/TN per label (for F1)
                for lab in labels:
                    in_set = lab in pred_set
                    is_true = (lab == true_lab)
                    tp[lab] += int(in_set and is_true)
                    fp[lab] += int(in_set and not is_true)
                    fn[lab] += int((not in_set) and is_true)
                    tn[lab] += int((not in_set) and not is_true)

            # per-class coverage and average set size
            for lab in labels:
                mask = test_data["true_label"] == lab
                n_lab = int(mask.sum())
                if n_lab > 0:
                    cov_val = correct_per_label[lab] / n_lab
                    size_val = setsize_per_label[lab] / n_lab
                else:
                    cov_val = np.nan
                    size_val = np.nan
                cond_cov[lab].append(cov_val)
                cond_size[lab].append(size_val)

            # marginal coverage and average set size
            total_correct = sum(correct_per_label.values())
            total_size = sum(setsize_per_label.values())
            marg_cov.append(total_correct / N_test)
            marg_size.append(total_size / N_test)

            # singleton rate (marginal)
            singleton_rate.append(singleton_count / N_test)

            # macro F1 (averaged over labels)
            f1_vals = []
            for lab in labels:
                num = 2 * tp[lab]
                denom = 2 * tp[lab] + fp[lab] + fn[lab]
                if denom > 0:
                    f1_vals.append(num / denom)
            f1_macro.append(np.mean(f1_vals) if f1_vals else np.nan)

        self.conditional_coverage_ = cond_cov
        self.conditional_setsize_ = cond_size
        self.marginal_coverage_ = np.asarray(marg_cov)
        self.marginal_setsize_ = np.asarray(marg_size)
        self.singleton_rate_ = np.asarray(singleton_rate)
        self.f1_macro_ = np.asarray(f1_macro)

    # ------------------------------------------------------------------
    # ROC-style computation (aggregated, separate alpha grid)
    # ------------------------------------------------------------------
    def _compute_fpr_tpr(self, alpha: float):
        """Compute global FPR and TPR for a given alpha."""
        if self.calibration_data is None or self.test_data is None:
            raise RuntimeError("Call fit() first.")
        if self.labels is None:
            raise RuntimeError("Labels not set. Did you call fit()?")

        labels = self.labels
        tp = {lab: 0 for lab in labels}
        fp = {lab: 0 for lab in labels}
        fn = {lab: 0 for lab in labels}
        tn = {lab: 0 for lab in labels}

        qhat_dict = self._calibrate(self.calibration_data, alpha)

        for _, row in self.test_data.iterrows():
            true_lab = row["true_label"]
            pred_set = self._predict_set(row, qhat_dict)

            for lab in labels:
                in_set = (lab in pred_set)
                is_true = (lab == true_lab)
                tp[lab] += int(in_set and is_true)
                fp[lab] += int(in_set and not is_true)
                fn[lab] += int((not in_set) and is_true)
                tn[lab] += int((not in_set) and not is_true)

        total_fp = sum(fp.values())
        total_tn = sum(tn.values())
        total_tp = sum(tp.values())
        total_fn = sum(fn.values())

        fpr = total_fp / (total_fp + total_tn) if (total_fp + total_tn) > 0 else 0.0
        tpr = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        return fpr, tpr

    def _compute_roc(self):
        """Compute FPR and TPR arrays for alphas_roc."""
        fprs: List[float] = []
        tprs: List[float] = []
        for a in self.alphas_roc:
            fpr, tpr = self._compute_fpr_tpr(a)
            fprs.append(fpr)
            tprs.append(tpr)
        self.fpr_ = np.asarray(fprs)
        self.tpr_ = np.asarray(tprs)

    # ------------------------------------------------------------------
    # Plotting methods (with saving)
    # ------------------------------------------------------------------
    def plot_coverage(self, save_path: Optional[str] = None, show: bool = True):
        """Plot conditional and marginal coverage vs (1 - alpha),
        with a binomial confidence band for the marginal coverage."""
        if self.conditional_coverage_ is None or self.marginal_coverage_ is None:
            raise RuntimeError("Coverage not computed; call run_full_analysis() first.")

        if self.test_data is None:
            raise RuntimeError("No test data; call fit() first.")

        labels = self.labels
        alphas = self.alphas_cov
        label_names = [lab.replace("_", " ") for lab in labels]

        # Compute binomial CI for marginal coverage at each alpha
        N = len(self.test_data)
        # approximate counts as round(marginal_coverage * N)
        counts = np.round(self.marginal_coverage_ * N).astype(int)
        nobs = np.full_like(counts, N)
        lower, upper = proportion_confint(counts, nobs, alpha=0.01)  # 99% CI

        fig, ax = plt.subplots()

        # Confidence band for marginal coverage
        ax.fill_between(
            1 - alphas,
            lower,
            upper,
            alpha=0.15,
            label="99% CI (marginal)",
        )

        # Per-class coverage
        for i, lab in enumerate(labels):
            ax.plot(1 - alphas, self.conditional_coverage_[lab], label=label_names[i])

        # Marginal coverage curve
        ax.plot(
            1 - alphas,
            self.marginal_coverage_,
            label="marginal",
            linewidth=2,
            linestyle="dashed",
        )

        ax.set_xlabel(r"1 - $\alpha$", fontsize=12)
        ax.set_ylabel("Coverage", fontsize=12)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True)
        ax.legend(loc="upper left", fontsize=7, ncol=2)
        ax.set_title(f"Coverage vs 1 - alpha ({self.cp_mode} CP)")

        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, bbox_inches="tight", dpi=300)

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig, ax

    def plot_setsize(
        self,
        cutoff: int = 3,
        save_path: Optional[str] = None,
        show: bool = True,
    ):
        """Plot conditional and marginal average set size vs (1 - alpha).

        cutoff: how many smallest alphas to remove from the left side
                (i.e., drop the most conservative coverage points).
        """
        if self.conditional_setsize_ is None or self.marginal_setsize_ is None:
            raise RuntimeError("Set size not computed; call run_full_analysis() first.")

        labels = self.labels
        alphas = self.alphas_cov
        label_names = [lab.replace("_", " ") for lab in labels]

        # Slice consistently: drop the first 'cutoff' alphas from both x and y
        alphas_sliced = alphas[cutoff:]
        fig, ax = plt.subplots()

        for i, lab in enumerate(labels):
            y = np.asarray(self.conditional_setsize_[lab])[cutoff:]
            ax.plot(
                1 - alphas_sliced,
                y,
                label=label_names[i],
            )

        ax.plot(
            1 - alphas_sliced,
            self.marginal_setsize_[cutoff:],
            linewidth=2,
            linestyle="dashed",
            label="marginal",
        )

        ax.set_xlabel(r"1 - $\alpha$", fontsize=12)
        ax.set_ylabel("Average set size", fontsize=12)
        ax.grid(True)
        ax.legend(loc="upper left", fontsize=9, ncol=2)
        ax.set_title(f"Average set size vs 1 - alpha ({self.cp_mode} CP)")

        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, bbox_inches="tight", dpi=300)

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig, ax

    def plot_singleton(self, save_path: Optional[str] = None, show: bool = True):
        """Plot marginal singleton rate vs alpha."""
        if self.singleton_rate_ is None:
            raise RuntimeError("Singleton rate not computed; call run_full_analysis() first.")

        alphas = self.alphas_cov

        fig, ax = plt.subplots()
        ax.plot(alphas, self.singleton_rate_, marker="o")

        ax.set_xlabel(r"$\alpha$", fontsize=12)
        ax.set_ylabel("Singleton rate", fontsize=12)
        ax.grid(True)
        ax.set_ylim(0, 1)
        ax.set_title(f"Singleton rate vs alpha ({self.cp_mode} CP)")

        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, bbox_inches="tight", dpi=300)

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig, ax

    def plot_f1(self, save_path: Optional[str] = None, show: bool = True):
        """Plot macro-F1 vs (1 - alpha)."""
        if self.f1_macro_ is None:
            raise RuntimeError("F1 scores not computed; call run_full_analysis() first.")

        alphas = self.alphas_cov

        fig, ax = plt.subplots()
        ax.plot(1 - alphas, self.f1_macro_, marker="o")

        ax.set_xlabel(r"1 - $\alpha$", fontsize=12)
        ax.set_ylabel("Macro F1 score", fontsize=12)
        ax.grid(True)
        ax.set_ylim(0, 1)
        ax.set_title(f"Macro F1 vs alpha ({self.cp_mode} CP)")

        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, bbox_inches="tight", dpi=300)

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig, ax

    def plot_roc(self, save_path: Optional[str] = None, show: bool = True):
        """Plot ROC-style curve (TPR vs FPR) across alphas_roc."""
        if self.fpr_ is None or self.tpr_ is None:
            raise RuntimeError("ROC values not computed; call run_full_analysis() first.")

        fig, ax = plt.subplots()
        sc = ax.scatter(self.fpr_, self.tpr_, s=20, c=self.alphas_roc)
        cbar = fig.colorbar(sc)
        cbar.set_label(r"error rate $\alpha$")

        ax.set_xlabel("False positive rate FP/(FP + TN)", fontsize=12)
        ax.set_ylabel("True positive rate TP/(TP + FN)", fontsize=12)
        ax.grid(True)
        ax.set_title(f"TPR vs FPR ({self.cp_mode} CP)")

        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, bbox_inches="tight", dpi=300)

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig, ax
