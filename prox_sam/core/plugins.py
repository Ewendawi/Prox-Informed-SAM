import numpy as np
try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None
from typing import Any, Optional, Sequence, Tuple, Union

class Plugin:
    """Base class for plugins that attach to the training loop."""
    def before_train_task(self, strategy): pass
    def after_train_task(self, strategy): pass
    def after_train_epoch(self, strategy): pass
    def before_training_step(self, strategy): pass
    def after_training_step(self, strategy): pass

class StabilityMetric(Plugin):
    @staticmethod
    def compute_forgetting_from_matrix(acc_matrix: np.ndarray) -> float:
        n_tasks = int(acc_matrix.shape[0])
        if n_tasks <= 1:
            return 0.0
        # Mirror `compute_forgetting(self.n_tasks)` semantics.
        current_task_id = n_tasks
        forgettings = []
        for k in range(current_task_id - 1):
            max_acc = float(np.max(acc_matrix[:current_task_id, k]))
            current_acc = float(acc_matrix[current_task_id - 1, k])
            forgettings.append(max_acc - current_acc)
        return float(np.mean(forgettings)) if forgettings else 0.0

    @staticmethod
    def compute_learning_success_from_matrix(acc_matrix: np.ndarray) -> float:
        return float(np.mean(np.diag(acc_matrix))) if acc_matrix.size else 0.0

    @staticmethod
    def compute_retention_ratio_from_matrix(acc_matrix: np.ndarray) -> float:
        ls = StabilityMetric.compute_learning_success_from_matrix(acc_matrix)
        if ls < 1e-6:
            return 0.0
        final_avg = float(np.mean(acc_matrix[-1]))
        return float(final_avg / ls)

    @staticmethod
    def compute_all_metrics_offline(
        acc_matrix: Union[np.ndarray, Sequence[Sequence[float]]],
        acc_history: dict[int, Sequence[Tuple[int, float]]],
        task_switch_steps: Sequence[dict[str, Any]],
        *,
        recovery_window: int = 5000,
        recovery_scale: float = 0.80,
        sg_window_steps: int = 100,
        acc_bounds: Tuple[float, float] = (0.0, 100.0),
        auc_clamp_to_bounds: bool = True,
        auc_clamp_ratio_01: bool = True,
    ) -> dict[str, Any]:
        """
        Computes aggregate + per-task continual-learning metrics from saved artifacts only.

        Uses only:
        - `acc_matrix` (classic CL accuracy matrix)
        - `acc_history` (per-task [(global_step, acc), ...])
        - `task_switch_steps` (list of {"from_task","to_task","global_step"} records)
        and the *_offline helpers for stability metrics.
        """
        acc_matrix_np = np.asarray(acc_matrix, dtype=float)
        if acc_matrix_np.ndim != 2 or acc_matrix_np.shape[0] != acc_matrix_np.shape[1]:
            raise ValueError("acc_matrix must be a square [n_tasks, n_tasks] matrix")
        n_tasks = int(acc_matrix_np.shape[0])

        switch_records = []
        for rec in task_switch_steps or []:
            if not isinstance(rec, dict):
                continue
            if "global_step" not in rec or rec["global_step"] is None:
                continue
            try:
                global_step = int(rec["global_step"])
            except Exception:
                continue
            switch_records.append(
                {
                    "from_task": int(rec.get("from_task", -1)),
                    "to_task": int(rec.get("to_task", -1)),
                    "global_step": global_step,
                }
            )
        switch_records.sort(key=lambda r: r["global_step"])
        all_switch_steps = [r["global_step"] for r in switch_records]

        final_step: Optional[int] = None
        for history in (acc_history or {}).values():
            for step, _acc in history:
                try:
                    s = int(step)
                except Exception:
                    continue
                final_step = s if final_step is None else max(final_step, s)
        if final_step is None and all_switch_steps:
            final_step = max(all_switch_steps)

        def _switch_steps_affecting_task(task_id: int) -> list[int]:
            # A switch "to_task = t" affects all older tasks k < t.
            return [r["global_step"] for r in switch_records if r["to_task"] > task_id]

        task_sg_areas: dict[int, list[float]] = {k: [] for k in range(n_tasks)}
        task_peak_drops: dict[int, list[float]] = {k: [] for k in range(n_tasks)}
        task_recovery_times: dict[int, list[int]] = {k: [] for k in range(n_tasks)}
        task_first_drops: dict[int, list[float]] = {k: [] for k in range(n_tasks)}
        task_last_drops: dict[int, list[float]] = {k: [] for k in range(n_tasks)}
        task_acc_auc: dict[int, float] = {k: 0.0 for k in range(n_tasks)}
        task_acc_auc_norm: dict[int, float] = {k: 0.0 for k in range(n_tasks)}
        task_acc_auc_ratio_01: dict[int, float] = {k: 0.0 for k in range(n_tasks)}
        task_has_acc_history: dict[int, bool] = {k: False for k in range(n_tasks)}

        for k in range(n_tasks):
            history_k = (acc_history or {}).get(k, (acc_history or {}).get(str(k), []))
            switch_steps_k = _switch_steps_affecting_task(k)
            task_has_acc_history[k] = bool(history_k)
            auc_metrics = StabilityMetric.compute_acc_auc_offline(
                acc_history=history_k,
                start_step=None,
                end_step=final_step,
                acc_bounds=acc_bounds,
                clamp_to_bounds=bool(auc_clamp_to_bounds),
                clamp_ratio_01=bool(auc_clamp_ratio_01),
            )
            task_acc_auc[k] = float(auc_metrics["acc_auc"])
            task_acc_auc_norm[k] = float(auc_metrics["acc_auc_norm"])
            task_acc_auc_ratio_01[k] = float(auc_metrics["acc_auc_ratio_01"])
            if not switch_steps_k:
                continue

            drop_metrics = StabilityMetric.compute_drop_metrics_offline(
                acc_history=history_k,
                switch_steps=switch_steps_k,
                final_step=final_step,
            )
            recovery_times = StabilityMetric.compute_recovery_time_offline(
                acc_history=history_k,
                switch_steps=switch_steps_k,
                recovery_window=int(recovery_window),
                recovery_scale=float(recovery_scale),
                final_step=final_step,
            )
            sg_areas = StabilityMetric.compute_sg_area_offline(
                acc_history=history_k,
                switch_steps=switch_steps_k,
                window_steps=int(sg_window_steps),
                final_step=final_step,
            )

            task_peak_drops[k] = list(drop_metrics["peak_drop"])
            task_recovery_times[k] = list(recovery_times)
            task_sg_areas[k] = list(sg_areas)
            task_first_drops[k] = list(drop_metrics["first_drop"])
            task_last_drops[k] = list(drop_metrics["last_drop"])

        all_sg_areas = [val for k in range(n_tasks) for val in task_sg_areas[k]]
        all_peak_drops = [val for k in range(n_tasks) for val in task_peak_drops[k]]
        all_recovery_times = [val for k in range(n_tasks) for val in task_recovery_times[k]]
        all_first_drops = [val for k in range(n_tasks) for val in task_first_drops[k]]
        all_last_drops = [val for k in range(n_tasks) for val in task_last_drops[k]]

        stability_gap = float(np.mean(all_sg_areas)) if all_sg_areas else 0.0
        peak_drop = float(np.mean(all_peak_drops)) if all_peak_drops else 0.0
        first_drop = float(np.mean(all_first_drops)) if all_first_drops else 0.0
        last_drop = float(np.mean(all_last_drops)) if all_last_drops else 0.0
        recovery_time_median = float(np.median(all_recovery_times)) if all_recovery_times else 0.0
        recovery_rate = (
            float(sum(1 for t in all_recovery_times if t < int(recovery_window)) / len(all_recovery_times))
            if all_recovery_times
            else 0.0
        )
        task_auc_norm_vals = [task_acc_auc_norm[k] for k in range(n_tasks) if task_has_acc_history[k]]
        acc_auc_norm_mean = float(np.mean(task_auc_norm_vals)) if task_auc_norm_vals else 0.0
        task_auc_ratio_01_vals = [task_acc_auc_ratio_01[k] for k in range(n_tasks) if task_has_acc_history[k]]
        acc_auc_ratio_01_mean = float(np.mean(task_auc_ratio_01_vals)) if task_auc_ratio_01_vals else 0.0

        per_task_metrics: dict[int, dict[str, Any]] = {}
        for k in range(n_tasks):
            per_task_metrics[k] = {
                "peak_drop": task_peak_drops[k],
                "recovery_time": task_recovery_times[k],
                "sg_area": task_sg_areas[k],
                "first_drop": task_first_drops[k],
                "last_drop": task_last_drops[k],
                "acc_auc": task_acc_auc[k],
                "acc_auc_norm": task_acc_auc_norm[k],
                "acc_auc_ratio_01": task_acc_auc_ratio_01[k],
            }

        return {
            "acc_matrix": acc_matrix_np.tolist(),
            "forgetting": StabilityMetric.compute_forgetting_from_matrix(acc_matrix_np),
            "learning_success": StabilityMetric.compute_learning_success_from_matrix(acc_matrix_np),
            "retention_ratio": StabilityMetric.compute_retention_ratio_from_matrix(acc_matrix_np),
            "stability_gap": stability_gap,
            "peak_drop": peak_drop,
            "first_drop": first_drop,
            "last_drop": last_drop,
            "acc_auc_norm_mean": acc_auc_norm_mean,
            "acc_auc_ratio_01_mean": acc_auc_ratio_01_mean,
            "recovery_time_median": recovery_time_median,
            "recovery_rate": recovery_rate,
            "recovery_window": int(recovery_window),
            "recovery_scale": float(recovery_scale),
            "sg_window_steps": int(sg_window_steps),
            "acc_bounds": (float(acc_bounds[0]), float(acc_bounds[1])),
            "per_task_metrics": per_task_metrics,
            "all_sg_areas": all_sg_areas,
            "all_peak_drops": all_peak_drops,
            "all_first_drops": all_first_drops,
            "all_last_drops": all_last_drops,
            "all_recovery_times": all_recovery_times,
            "acc_history": acc_history,
            "task_switch_steps": list(task_switch_steps),
            "switch_steps": all_switch_steps,
            "final_step": final_step,
        }

    @staticmethod
    def _evaluate_val_loaders(strategy) -> dict[int, float]:
        """
        Evaluates `strategy.model` on each loader in `strategy.val_loaders`.

        Returns:
            Mapping {task_id: accuracy_percent}.
        """
        if torch is None:
            raise ModuleNotFoundError("torch is required for _evaluate_val_loaders")
        if not hasattr(strategy, "val_loaders") or not strategy.val_loaders:
            return {}

        model = strategy.model
        device = strategy.device
        was_training = model.training

        accs: dict[int, float] = {}
        model.eval()
        try:
            with torch.no_grad():
                for k, loader in enumerate(strategy.val_loaders):
                    correct = 0
                    total = 0
                    for inputs, targets in loader:
                        inputs, targets = inputs.to(device), targets.to(device)
                        outputs = model(inputs)
                        _, predicted = outputs.max(1)
                        total += targets.size(0)
                        correct += predicted.eq(targets).sum().item()
                    if total > 0:
                        accs[int(k)] = float(100.0 * correct / total)
        finally:
            if was_training:
                model.train()

        return accs

    @staticmethod
    def compute_recovery_time_offline(
        acc_history: Sequence[Tuple[int, float]],
        switch_steps: Union[int, Sequence[int]],
        recovery_window: int,
        recovery_scale: float,
        final_step: Optional[int] = None,
    ) -> list[int]:
        """
        Offline Recovery Time for a single task k across multiple task switches.

        For each switch `i`, uses:
        - ref_acc_i = the last observed accuracy in the previous segment (prev_switch, switch_steps[i]]
        - threshold_i = recovery_scale * ref_acc_i

        Returns the first time (within `recovery_window` steps after `switch_steps[i]`) where
        acc >= threshold_i, constrained to the interval ending at the next switch step
        (or `final_step` for the last switch, if provided).
        If not recovered, returns `recovery_window` (censored).
        """
        if isinstance(switch_steps, (int, np.integer)):
            switch_steps_list = [int(switch_steps)]
        else:
            switch_steps_list = [int(s) for s in switch_steps]

        if not switch_steps_list:
            return []

        history_sorted = sorted(((int(s), float(a)) for (s, a) in acc_history), key=lambda x: x[0])
        ref_accs_list = StabilityMetric._compute_ref_accs_from_history(history_sorted, switch_steps_list)

        results: list[int] = []
        n_switches = len(switch_steps_list)
        for i, (switch_step, ref_acc) in enumerate(zip(switch_steps_list, ref_accs_list)):
            if ref_acc < 1e-2:
                results.append(recovery_window)
                continue

            end_step = switch_steps_list[i + 1] if i + 1 < n_switches else final_step
            threshold = recovery_scale * ref_acc
            recovered = None
            for step, acc in history_sorted:
                if step <= switch_step:
                    continue
                if end_step is not None and step > end_step:
                    break
                if step > switch_step + recovery_window:
                    break
                if acc >= threshold:
                    recovered = int(step - switch_step)
                    break
            results.append(int(recovered if recovered is not None else recovery_window))
        return results

    @staticmethod
    def compute_sg_area_offline(
        acc_history: Sequence[Tuple[int, float]],
        switch_steps: Union[int, Sequence[int]],
        window_steps: int = 100,
        final_step: Optional[int] = None,
    ) -> list[float]:
        """
        Offline SG_area for a single task k across multiple task switches.

        For each switch `i`, uses:
        - ref_acc_i = the last observed accuracy in the previous segment (prev_switch, switch_steps[i]]

        Then integrates max(0, ref_acc_i - acc) over observed points in the first `window_steps`
        after `switch_steps[i]`, constrained to the interval ending at the next switch step
        (or `final_step` for the last switch, if provided), and normalizes by `window_steps`.
        """
        if isinstance(switch_steps, (int, np.integer)):
            switch_steps_list = [int(switch_steps)]
        else:
            switch_steps_list = [int(s) for s in switch_steps]

        if not switch_steps_list or window_steps <= 0:
            return []

        history_sorted = sorted(((int(s), float(a)) for (s, a) in acc_history), key=lambda x: x[0])
        ref_accs_list = StabilityMetric._compute_ref_accs_from_history(history_sorted, switch_steps_list)

        results: list[float] = []
        n_switches = len(switch_steps_list)
        for i, (switch_step, ref_acc) in enumerate(zip(switch_steps_list, ref_accs_list)):
            segment_end = switch_steps_list[i + 1] if i + 1 < n_switches else final_step
            window_end = switch_step + window_steps
            if segment_end is not None:
                window_end = min(window_end, segment_end)

            window_history = [(s, a) for (s, a) in history_sorted if switch_step < s <= window_end]
            if not window_history:
                results.append(0.0)
                continue

            area = 0.0
            prev_step = switch_step
            for step, acc in window_history:
                delta_step = step - prev_step
                drop_val = max(0.0, ref_acc - acc)
                area += drop_val * delta_step
                prev_step = step
            results.append(float(area / float(window_steps)))
        return results

    @staticmethod
    def compute_drop_metrics_offline(
        acc_history: Sequence[Tuple[int, float]],
        switch_steps: Union[int, Sequence[int]],
        final_step: Optional[int] = None,
    ) -> dict[str, list[float]]:
        """
        Offline drop metrics for a single task k across multiple task switches.

        For each switch `i`, defines:
        - ref_acc_i = the last observed accuracy in the previous segment (prev_switch, S_i]

        Over the post-switch segment (S_i, end_step_i], where end_step_i is the next switch step
        (or `final_step` for the last switch, if provided), computes:
        - first_drop_i = max(0, ref_acc_i - first_acc_i)
        - last_drop_i  = max(0, ref_acc_i - last_acc_i)
        - peak_drop_i  = max(0, ref_acc_i - min_acc_i)
        """
        if isinstance(switch_steps, (int, np.integer)):
            switch_steps_list = [int(switch_steps)]
        else:
            switch_steps_list = [int(s) for s in switch_steps]

        if not switch_steps_list:
            return {"first_drop": [], "last_drop": [], "peak_drop": []}

        history_sorted = sorted(((int(s), float(a)) for (s, a) in acc_history), key=lambda x: x[0])
        ref_accs_list = StabilityMetric._compute_ref_accs_from_history(history_sorted, switch_steps_list)

        first_drops: list[float] = []
        last_drops: list[float] = []
        peak_drops: list[float] = []

        n_switches = len(switch_steps_list)
        for i, (switch_step, ref_acc) in enumerate(zip(switch_steps_list, ref_accs_list)):
            end_step = switch_steps_list[i + 1] if i + 1 < n_switches else final_step

            first_acc = None
            last_acc = None
            min_acc = None

            for step, acc in history_sorted:
                if step <= switch_step:
                    continue
                if end_step is not None and step > end_step:
                    break
                if first_acc is None:
                    first_acc = acc
                last_acc = acc
                min_acc = acc if min_acc is None else min(min_acc, acc)

            if first_acc is None:
                first_acc = ref_acc
            if last_acc is None:
                last_acc = ref_acc
            if min_acc is None:
                min_acc = ref_acc

            first_drops.append(float(max(0.0, ref_acc - first_acc)))
            last_drops.append(float(max(0.0, ref_acc - last_acc)))
            peak_drops.append(float(max(0.0, ref_acc - min_acc)))

        return {"first_drop": first_drops, "last_drop": last_drops, "peak_drop": peak_drops}

    @staticmethod
    def compute_acc_auc_offline(
        acc_history: Sequence[Tuple[int, float]],
        *,
        start_step: Optional[int] = None,
        end_step: Optional[int] = None,
        acc_bounds: Tuple[float, float] = (0.0, 100.0),
        clamp_to_bounds: bool = True,
        clamp_ratio_01: bool = True,
    ) -> dict[str, float]:
        """
        Offline AUC over steps for a single task's accuracy history.

        Uses a right-continuous step function: between two evaluation points, accuracy is treated
        as constant at the last observed value. If `end_step` extends beyond the last observation,
        the last observed accuracy is extended to `end_step`.

        Returns three values:
        - acc_auc: the (step-weighted) area under accuracy
        - acc_auc_norm: acc_auc / (end_step - start_step), i.e., average accuracy over time
        - acc_auc_ratio_01: AUC ratio w.r.t. bounds in `acc_bounds` (defaults to [0,1])
        """
        history_sorted = sorted(((int(s), float(a)) for (s, a) in acc_history), key=lambda x: x[0])
        if not history_sorted:
            return {"acc_auc": 0.0, "acc_auc_norm": 0.0, "acc_auc_ratio_01": 0.0}

        if start_step is None:
            start_step = int(history_sorted[0][0])
        if end_step is None:
            end_step = int(history_sorted[-1][0])

        start_step = int(start_step)
        end_step = int(end_step)
        if end_step <= start_step:
            return {"acc_auc": 0.0, "acc_auc_norm": 0.0, "acc_auc_ratio_01": 0.0}

        prev_step = start_step

        lo, hi = float(acc_bounds[0]), float(acc_bounds[1])
        if hi <= lo:
            raise ValueError("acc_bounds must have upper > lower")

        prev_acc = None
        for step, acc in history_sorted:
            if step <= start_step:
                prev_acc = acc
            else:
                break
        if prev_acc is None:
            prev_acc = float(history_sorted[0][1])
        if clamp_to_bounds:
            prev_acc = float(min(hi, max(lo, prev_acc)))

        area = 0.0
        for step, acc in history_sorted:
            if step <= start_step:
                continue
            if step > end_step:
                break
            if clamp_to_bounds:
                acc = float(min(hi, max(lo, acc)))
            area += prev_acc * float(step - prev_step)
            prev_step = step
            prev_acc = acc

        if prev_step < end_step:
            area += prev_acc * float(end_step - prev_step)

        duration = float(end_step - start_step)
        acc_auc = float(area)
        acc_auc_norm = float(area / duration)
        acc_auc_ratio_01 = float((acc_auc_norm - lo) / (hi - lo))
        if clamp_ratio_01:
            acc_auc_ratio_01 = float(min(1.0, max(0.0, acc_auc_ratio_01)))
        return {"acc_auc": acc_auc, "acc_auc_norm": acc_auc_norm, "acc_auc_ratio_01": acc_auc_ratio_01}

    @staticmethod
    def _compute_ref_accs_from_history(
        acc_history_sorted: Sequence[Tuple[int, float]],
        switch_steps: Sequence[int],
    ) -> list[float]:
        """
        Computes per-switch reference accuracies from history.

        For switch i at step S_i, ref_acc_i is the last observed accuracy in
        (S_{i-1}, S_i] where S_{-1} is -infinity.
        """
        ref_accs: list[float] = []
        prev_switch = -10**30
        idx = 0

        for switch_step in switch_steps:
            last = None
            while idx < len(acc_history_sorted) and acc_history_sorted[idx][0] <= switch_step:
                step, acc = acc_history_sorted[idx]
                if step > prev_switch:
                    last = acc
                idx += 1

            if last is None:
                last = 0.0
            ref_accs.append(float(last))
            prev_switch = switch_step

        return ref_accs

    def __init__(
        self,
        n_tasks,
        eval_every_n_batches: Optional[int] = 50,
    ):
        self.n_tasks = n_tasks
        if eval_every_n_batches is None:
            self.eval_every_n_batches: Optional[int] = None
        else:
            eval_every_n_batches = int(eval_every_n_batches)
            if eval_every_n_batches < 1:
                raise ValueError("eval_every_n_batches must be >= 1 (or None)")
        self.eval_every_n_batches = eval_every_n_batches
        # Matrix to store accuracy: [task_id, time_step (task_id)]
        self.acc_matrix = np.zeros((n_tasks, n_tasks))
        
        # Structure for Stability Gap:
        # Dictionary mapping task_id -> list of (step, acc_on_that_task)
        self.acc_history = {k: [] for k in range(n_tasks)}
        
        # Loss Log
        # {task_id: [loss_epoch_0, loss_epoch_1, ...]}
        self.metric_history = {} # Stores detailed metrics: {task_id: {key: [val_epoch_0...]}}

        # Task switches (needed for offline stability metrics)
        self.task_switch_steps = []  # [{"from_task": int, "to_task": int, "global_step": int}, ...]

    def before_train_task(self, strategy):
        self._record_task_switch(current_task_id=strategy.task_id, global_step=strategy.global_step)
        self.metric_history[strategy.task_id] = {}
    
    def after_train_task(self, strategy):
        pass

    def after_train_epoch(self, strategy):
        # Record training loss for the current task
        if strategy.task_id in self.metric_history:
            self.metric_history[strategy.task_id] = strategy.current_metrics    

        # Online Evaluation for Stability Gap
        # We need to evaluate on all PREVIOUS tasks (k < current_task_id)
        # strategy.val_loaders should be a list where index k corresponds to task k
        accs = self._evaluate_val_loaders(strategy)
        if accs:
            # Print current task accuracy if available
            if strategy.task_id in accs:
                print(f"[Eval] Acc: {accs[strategy.task_id]:.2f}%")
            
            # Update separate SG history using global step for consistency
            self._update_step(strategy.global_step, strategy.task_id, accs)

    def _update_step(self, step, current_task_id, accs):
        self._record_acc(step, current_task_id, accs)
            
    def _record_acc(self, step, current_task_id, accs):
        """
        Called frequently to track step-level accuracies.
        """
        current_task_id = int(current_task_id)
        for k, v in accs.items():
            k = int(k)
            # Keep the classic CL accuracy matrix updated online for the current task.
            if 0 <= current_task_id < self.n_tasks and 0 <= k < self.n_tasks:
                self.acc_matrix[current_task_id, k] = v

            # Record full history
            if k not in self.acc_history:
                self.acc_history[k] = []
            self.acc_history[k].append((step, v))

    def _record_task_switch(self, current_task_id, global_step):
        current_task_id = int(current_task_id)
        if global_step is None or current_task_id <= 0:
            return

        switch_record = {
            "from_task": current_task_id - 1,
            "to_task": current_task_id,
            "global_step": int(global_step),
        }
        if self.task_switch_steps and self.task_switch_steps[-1].get("to_task") == switch_record["to_task"]:
            self.task_switch_steps[-1] = switch_record
        else:
            self.task_switch_steps.append(switch_record)

    def after_training_step(self, strategy):
        """
        Called per-batch to track step-level accuracy for SG metrics.
        Evaluates on previous tasks and records (global_step, acc).
        """
        if self.eval_every_n_batches is None:
            return

        step_in_task = getattr(strategy, "step_in_task", None)
        global_step = getattr(strategy, "global_step", None)
        if step_in_task is not None:
            if int(step_in_task) % self.eval_every_n_batches != 0:
                return
        elif global_step is not None:
            if int(global_step) % self.eval_every_n_batches != 0:
                return

        accs = self._evaluate_val_loaders(strategy)
        if not accs:
            return
        
        # Update history with global step
        self._update_step(strategy.global_step, strategy.task_id, accs)
    
    def to_dict(self):
        return {
            "n_tasks": int(self.n_tasks),
            "acc_matrix": self.acc_matrix.tolist(),
            "acc_history": self.acc_history,
            "task_switch_steps": self.task_switch_steps,
            "metric_history": self.metric_history,
            "eval_every_n_batches": self.eval_every_n_batches,
        }
