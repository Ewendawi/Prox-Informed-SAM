import copy
import math

import torch

from prox_sam.core.base import BaseStrategy
from prox_sam.core.utils import AverageMeter


class ProxSAM(BaseStrategy):
    """Proximal-Informed SAM Strategy."""

    def __init__(self, model, optimizer_cls, buffer, device, args, plugins=None):
        super().__init__(model, optimizer_cls, buffer, device, args, plugins)
        self.anchor_model = None
        self.debug_records = []
        self.delta_ema = None
        self.delta_last = 0.0

    def _named_anchor_params(self):
        if self.anchor_model is None:
            return {}
        return dict(self.anchor_model.named_parameters())

    def _bn_param_names(self):
        bn_names = set()
        for module_name, module in self.model.named_modules():
            if not isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                continue
            if module.weight is not None:
                name = f"{module_name}.weight" if module_name else "weight"
                bn_names.add(name)
            if module.bias is not None:
                name = f"{module_name}.bias" if module_name else "bias"
                bn_names.add(name)
        return bn_names

    def _prox_named_params(self):
        """Returns (name, param, anchor_param) triplets for prox regularization.

        This implementation uses **backbone-only prox**: any parameter whose name
        contains 'classifier' is excluded so the head is not constrained.
        """
        anchor_params = self._named_anchor_params()
        prox_backbone_only = getattr(self.args, 'prox_backbone_only', True)
        exclude_bn = bool(getattr(self.args, 'prox_exclude_bn', False))
        bn_param_names = self._bn_param_names() if exclude_bn else set()
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if 'classifier' in name and prox_backbone_only:
                continue
            if name in bn_param_names:
                continue
            anchor_param = anchor_params.get(name)
            if anchor_param is None:
                continue
            yield name, param, anchor_param

    def compute_prox_loss(self, lambda_prox=None):
        """Computes the proximal regularization value.

        We use a *penalty-form trust region* around an anchor model $w_0$:

            prox(w; w0) = (lambda/2) * ||w - w0||_2^2

        This method returns the **scalar prox penalty** (a tensor on `self.device`)
        for the parameter subset selected by `_prox_named_params()`.

        Current behavior:

        - **Backbone-only prox**: `_prox_named_params()` excludes any parameter
          whose name contains `'classifier'`, so the classification head is
          completely unconstrained.
        - **Incremental-shape safety**: if a tensor has expanded along dim-0
          while keeping `shape[1:]` identical, we only regularize the old slice
          `p[:n_old]` that existed in the anchor. (If the head were included,
          this would match class-incremental classifier expansion.)

        Notes:

        - In the SAM path we often call this under `torch.no_grad()` because we
          only want the *value* for logging/total_loss, not its autograd graph.
        - The gradient is added analytically in `add_prox_grad_()`.
        """
        if self.anchor_model is None:
            return torch.tensor(0.0, device=self.device)

        if lambda_prox is None:
            lambda_prox = float(self.args.lambda_prox)
        lambda_prox = float(lambda_prox)
        if lambda_prox <= 0:
            return torch.tensor(0.0, device=self.device)

        prox_loss = torch.tensor(0.0, device=self.device)

        for name, p, p_anchor in self._prox_named_params():
            # Typical case: parameter exists in both models with identical shape.
            if p.shape == p_anchor.shape:
                # (lambda/2) * ||p - p0||^2
                prox_loss += (lambda_prox / 2) * (p - p_anchor).norm(2) ** 2
                continue

            # Shape mismatch: allow dim-0 expansion, but only constrain the
            # portion that corresponds to the anchor parameter.
            if len(p.shape) > 0 and p.shape[1:] == p_anchor.shape[1:]:
                n_old = p_anchor.shape[0]
                prox_loss += (lambda_prox / 2) * (p[:n_old] - p_anchor).norm(2) ** 2
                continue

            # Any other mismatch is ignored (should be rare; usually model and
            # anchor are structurally identical except for controlled expansion).

        return prox_loss

    def add_prox_grad_(self, lambda_prox=None):
        """Adds the proximal gradient term to existing `.grad` buffers.

        This is used in the SAM path.

        Why we need this:

        - `sam_step()` recomputes gradients using only the task loss (CE) at the
          perturbed weights `w+eps`, leaving `.grad = ∇CE(w+eps)`.
        - If we want to optimize `CE(w) + (lambda/2)||w-w0||^2`, we still need to
          incorporate the prox gradient at the **unperturbed** point `w`.

        For the penalty (lambda/2)*||w-w0||^2, the analytic gradient is:

            ∇_w prox = lambda * (w - w0)

        This method adds that term *in-place* on `.grad` for the parameters
        selected by `_prox_named_params()`.

        Important:

        - This does **not** create autograd edges; it mutates `.grad` directly.
        - The head is **not** affected because `_prox_named_params()` excludes
          `'classifier'` parameters.
        """
        if self.anchor_model is None:
            return

        if lambda_prox is None:
            lambda_prox = float(self.args.lambda_prox)
        lambda_prox = float(lambda_prox)
        if lambda_prox <= 0:
            return

        for name, p, p_anchor in self._prox_named_params():
            # If a parameter was not involved in the current loss, autograd may
            # leave `p.grad` as None; we skip in that case.
            if p.grad is None:
                continue

            # Standard case.
            if p.shape == p_anchor.shape:
                p.grad.add_(p.data - p_anchor.data, alpha=lambda_prox)
                continue

            # Dim-0 expansion: only add gradient to the anchored slice.
            if len(p.shape) > 0 and p.shape[1:] == p_anchor.shape[1:]:
                n_old = p_anchor.shape[0]
                p.grad[:n_old].add_(p.data[:n_old] - p_anchor.data, alpha=lambda_prox)
                continue

            # Other mismatches are ignored.

    def _named_trainable_params(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                yield name, param

    def _sam_named_params(self):
        named_params = list(self._named_trainable_params())
        if not getattr(self.args, 'sam_backbone_only', False):
            return named_params
        return [(n, p) for n, p in named_params if 'classifier' not in n]

    def _bn_modules(self):
        return [
            m
            for m in self.model.modules()
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)
            and getattr(m, 'track_running_stats', False)
        ]

    def _snapshot_bn_running_stats(self):
        snapshot = []
        for m in self._bn_modules():
            running_mean = m.running_mean.detach().clone() if m.running_mean is not None else None
            running_var = m.running_var.detach().clone() if m.running_var is not None else None
            num_batches_tracked = (
                m.num_batches_tracked.detach().clone() if hasattr(m, 'num_batches_tracked') else None
            )
            snapshot.append((m, running_mean, running_var, num_batches_tracked))
        return snapshot

    def _bn_stats_delta(self, before, after):
        mean_sq = 0.0
        var_sq = 0.0
        nbt_delta = 0
        for (_, m0, v0, n0), (_, m1, v1, n1) in zip(before, after):
            if m0 is not None and m1 is not None:
                mean_sq += (m1 - m0).pow(2).sum().item()
            if v0 is not None and v1 is not None:
                var_sq += (v1 - v0).pow(2).sum().item()
            if n0 is not None and n1 is not None:
                nbt_delta += int((n1 - n0).item())
        return {
            'bn_running_mean_delta_l2': mean_sq**0.5,
            'bn_running_var_delta_l2': var_sq**0.5,
            'bn_num_batches_tracked_delta': nbt_delta,
        }

    def _disable_bn_running_stats_momentum0(self):
        """Temporarily disables BN running-stats updates without switching to eval().

        In SAM we do a second forward/backward at perturbed weights (w+eps). We
        want gradients from that perturbed pass, but we don't want its batch
        statistics to update BN running buffers (running_mean/running_var).

        Method C: keep BN in train mode (so it still uses batch stats), but set
        BN momentum to 0 so running buffers are not updated by the perturbed
        forward pass.
        """
        bns = []
        momenta = []
        for m in self._bn_modules():
            bns.append(m)
            momenta.append(m.momentum)
            m.momentum = 0.0
        return bns, momenta

    def _restore_bn_running_stats_momentum0(self, bns, momenta):
        for m, mom in zip(bns, momenta):
            m.momentum = mom

    def update_anchor(self):
        """Updates the anchor model to the current model state"""
        self.anchor_model = copy.deepcopy(self.model)
        self.anchor_model.eval()
        for p in self.anchor_model.parameters():
            p.requires_grad = False
        self.delta_ema = None
        self.delta_last = 0.0
            
    def train_task(self, train_loader, val_loaders, epochs):
        super().train_task(train_loader, val_loaders, epochs)
        self.update_anchor()

    def compute_prox_drift_l2(self):
        """Returns delta_t = ||theta - theta_anchor||_2 over the prox-constrained parameter subset."""
        if self.anchor_model is None:
            return 0.0

        drift_sq = 0.0
        for _, p, p_anchor in self._prox_named_params():
            if p.shape == p_anchor.shape:
                diff = p.data - p_anchor.data
                drift_sq += float(diff.pow(2).sum().item())
                continue

            if len(p.shape) > 0 and p.shape[1:] == p_anchor.shape[1:]:
                n_old = p_anchor.shape[0]
                diff = p.data[:n_old] - p_anchor.data
                drift_sq += float(diff.pow(2).sum().item())
                continue

        return float(math.sqrt(max(0.0, drift_sq)))

    def compute_task_grad_norm_prox_subset(self):
        """Returns ||g||_2 over the prox-constrained parameter subset using current `.grad` buffers."""
        if self.anchor_model is None:
            return 0.0

        norm_terms = []
        for _, p, p_anchor in self._prox_named_params():
            if p.grad is None:
                continue
            g = p.grad.detach()

            if p.shape == p_anchor.shape:
                norm_terms.append(g.norm(p=2))
                continue

            if len(p.shape) > 0 and p.shape[1:] == p_anchor.shape[1:]:
                n_old = p_anchor.shape[0]
                norm_terms.append(g[:n_old].norm(p=2))
                continue

        if not norm_terms:
            return 0.0
        return float(torch.norm(torch.stack(norm_terms), p=2).item())

    def _get_lambda_prox(self, epoch=None, total_epochs=None):
        """Returns effective lambda_prox, optionally scheduled within a task."""
        lambda_base = float(getattr(self.args, 'lambda_prox', 0.0))
        if lambda_base <= 0:
            return 0.0

        start_epoch = int(getattr(self.args, 'prox_start_epoch', 0))
        if epoch < start_epoch:
            return 0.0

        schedule = str(getattr(self.args, 'lambda_schedule', 'constant')).lower()
        if schedule in {'inc_linear', 'inc_cosine', 'inc_poly'}:
            if epoch is None or total_epochs is None:
                return lambda_base
            
            denom = max(1, int(total_epochs) - 1 - start_epoch)
            t = (int(epoch) - start_epoch) / float(denom)
            t = max(0.0, min(1.0, t))
            if schedule == 'inc_linear':
                scale = t
            elif schedule == 'inc_cosine':
                scale = 0.5 * (1.0 - math.cos(math.pi * t))
            else:
                power = float(getattr(self.args, 'lambda_warmup_power', 1.0))
                power = max(0.0, power)
                scale = t**power
            return lambda_base * scale
        return lambda_base

    def get_prox_informed_rho(self, lambda_prox, prox_loss=None, grad_norm=None):
        """Returns rho with prox-informed coupling.

        Supported modes (set via `args.coupling_mode`):

        - `delta_ema` (default): rho_t = gamma * EMA(delta_t)
        - `grad_ratio`: rho_t = rho_max / (1 + beta * r_t), where r_t = ||p_t||/(||g_t||+eps)
        """
        rho_base = float(getattr(self.args, 'rho_base', 0.0))
        if (
            self.anchor_model is None
            or lambda_prox <= 0
            or not bool(getattr(self.args, 'use_coupling', False))
        ):
            return rho_base

        mode = str(getattr(self.args, 'coupling_mode', 'delta_ema')).lower()

        delta = None
        if prox_loss is not None and torch.is_tensor(prox_loss):
            # prox_loss = (lambda/2) * delta^2  =>  delta = sqrt(2*prox_loss/lambda)
            delta = float((2.0 * prox_loss / float(lambda_prox)).clamp_min(0.0).sqrt().item())
        if delta is None:
            delta = self.compute_prox_drift_l2()

        rho_min = float(getattr(self.args, 'rho_min', 1e-4))
        rho_max = float(getattr(self.args, 'rho_max', 0.5))

        if mode in {'grad_ratio', 'grad-ratio', 'ratio'}:
            if grad_norm is None:
                grad_norm = self.compute_task_grad_norm_prox_subset()
            p_norm = float(lambda_prox) * float(delta)
            eps = float(getattr(self.args, 'grad_ratio_eps', 1e-12))
            beta = float(getattr(self.args, 'grad_ratio_beta', 1.0))
            r_t = p_norm / (float(grad_norm) + eps)
            rho = rho_max / (1.0 + beta * r_t)
            return max(rho_min, min(float(rho), rho_max))

        # Default: drift/EMA coupling.
        ema_beta = float(getattr(self.args, 'delta_ema_beta', 0.1))
        ema_beta = max(0.0, min(1.0, ema_beta))
        if self.delta_ema is None:
            self.delta_ema = float(delta)
        else:
            self.delta_ema = (1.0 - ema_beta) * float(self.delta_ema) + ema_beta * float(delta)
        self.delta_last = float(delta)

        gamma = getattr(self.args, 'delta_gamma', 1.0)
        gamma = float(gamma)

        rho = gamma * float(self.delta_ema)
        return max(rho_min, min(rho, rho_max))

    def sam_step(self, loss_fn, inputs, targets, rho):
        # SAM in two passes:
        # - Pass A (done outside): compute grads at w to build perturbation eps ~ rho * g / ||g||.
        # - Pass B (done here): evaluate grads at perturbed weights w+eps, then restore weights to w.
        # After this function returns, params are back at w, but .grad contains d/dw L(w+eps).

        # 1) Save current state for ALL trainable params
        state = {n: p.data.clone() for n, p in self._named_trainable_params()}

        # 2) Virtual ascent on selected parameters (optionally backbone-only)
        sam_named_params = self._sam_named_params()
        sam_params = [p for _, p in sam_named_params]

        grad_norm_terms = [p.grad.norm(p=2) for p in sam_params if p.grad is not None]
        if len(grad_norm_terms) == 0:
            return {}
        grad_norm = torch.norm(torch.stack(grad_norm_terms), p=2)
        scale = rho / (grad_norm + 1e-12)

        for p in sam_params:
            if p.grad is None:
                continue
            p.data.add_(p.grad, alpha=scale)

        sam_grad_norm = float(grad_norm.detach().item())
        sam_eps_norm = float((scale.detach() * grad_norm.detach()).item())
        sam_scale = float(scale.detach().item())

        # 3) Calc loss/grad at perturbed weights
        bn_mode = str(getattr(self.args, 'sam_bn_mode', 'momentum0')).lower()
        bn_before = self._snapshot_bn_running_stats()
        restore_handle = None
        if bn_mode == 'save_restore':
            restore_handle = ('save_restore', bn_before)
        elif bn_mode == 'momentum0':
            restore_handle = ('momentum0',) + self._disable_bn_running_stats_momentum0()
        elif bn_mode == 'none':
            restore_handle = ('none',)
        else:
            restore_handle = ('momentum0',) + self._disable_bn_running_stats_momentum0()

        outputs = self.model(inputs)
        loss = loss_fn(outputs, targets)
        self.optimizer.zero_grad()
        loss.backward()

        bn_after = self._snapshot_bn_running_stats()
        bn_delta = self._bn_stats_delta(bn_before, bn_after)

        if restore_handle[0] == 'save_restore':
            _, snap = restore_handle
            for m, running_mean, running_var, num_batches_tracked in snap:
                if running_mean is not None and m.running_mean is not None:
                    m.running_mean.copy_(running_mean)
                if running_var is not None and m.running_var is not None:
                    m.running_var.copy_(running_var)
                if num_batches_tracked is not None and hasattr(m, 'num_batches_tracked'):
                    m.num_batches_tracked.copy_(num_batches_tracked)
        elif restore_handle[0] == 'momentum0':
            _, bns, momenta = restore_handle
            self._restore_bn_running_stats_momentum0(bns, momenta)
        elif restore_handle[0] == 'none':
            pass

        # 4) Restore parameters back to w
        for n, p in self._named_trainable_params():
            p.data.copy_(state[n])

        return {
            'sam_grad_norm': sam_grad_norm,
            'sam_eps_norm': sam_eps_norm,
            'sam_scale': sam_scale,
            **bn_delta,
        }

    def train_epoch(self, train_loader, **kwargs):
        self.model.train()
        
        # Trackers
        losses = AverageMeter()
        ce_losses = AverageMeter()
        prox_losses = AverageMeter()
        rhos = AverageMeter()
        drifts = AverageMeter()
        drift_emas = AverageMeter()
        param_norms = AverageMeter()
        
        # Check args
        epoch = int(kwargs.get('epoch', 0))
        total_epochs = kwargs.get('total_epochs')

        use_sam = bool(getattr(self.args, 'use_sam', False))
        if self.anchor_model is None:
            use_sam = False

        sam_epochs = int(getattr(self.args, 'sam_epochs', 0))
        sam_phase = str(getattr(self.args, 'sam_phase', 'first')).lower()
        if use_sam:
            if sam_epochs <= 0:
                use_sam = False
            else:
                if sam_phase == 'all':
                    start, end = 0, float('inf')
                elif sam_phase == 'last' and total_epochs is not None:
                    total_epochs = int(total_epochs)
                    start = max(0, total_epochs - sam_epochs)
                    end = total_epochs
                else:
                    start, end = 0, sam_epochs
                if not (start <= epoch < end):
                    use_sam = False

        debug_sam = bool(getattr(self.args, 'debug_sam', False))
        debug_every_n_steps = int(getattr(self.args, 'debug_every_n_steps', 1))
        sam_batch_mode = str(getattr(self.args, 'sam_batch_mode', 'all')).lower()
        if sam_batch_mode not in {'all', 'current', 'split'}:
            sam_batch_mode = 'all'
            
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            inputs_current, targets_current = inputs, targets
            
            # Increment step counters
            self.global_step += 1
            self.step_in_task += 1
            
            # Replay: keep total batch size fixed at args.batch_size.
            # The DataLoader is expected to yield only the current-task portion; we top up with replay.
            total_budget = int(getattr(self.args, 'batch_size', int(inputs.size(0))))
            total_budget = max(1, total_budget)
            current_bs = int(inputs.size(0))
            replay_bs = max(0, total_budget - current_bs)

            if replay_bs > 0 and self.buffer.current_size > 0:
                buf_inputs, buf_targets = self.buffer.sample(replay_bs)
            else:
                buf_inputs, buf_targets = None, None

            inputs_all, targets_all = inputs_current, targets_current
            if buf_inputs is not None:
                inputs_all = torch.cat([inputs_current, buf_inputs], dim=0)
                targets_all = torch.cat([targets_current, buf_targets], dim=0)

            if sam_batch_mode in {'current', 'split'}:
                sam_inputs, sam_targets = inputs_current, targets_current
            else:
                sam_inputs, sam_targets = inputs_all, targets_all

            # --- Metrics: Param Norm ---
            p_norm = 0.0
            for p in self.model.parameters():
                p_norm += p.data.norm(2).item() ** 2
            p_norm = p_norm ** 0.5
            param_norms.update(p_norm)
            
            # Prox (proximal regularization / "stay close to previous task")
            # Anchor is a frozen snapshot of the model taken at the end of the last task.
            # We penalize drift: (lambda/2) * ||theta - theta_anchor||^2.
            # If a tensor expanded in dim0 across tasks (e.g., classifier head grew for new classes),
            # only the "old" slice is regularized to avoid constraining newly-added parameters.
            lambda_prox = self._get_lambda_prox(epoch=epoch, total_epochs=total_epochs)
            prox_loss = torch.tensor(0.0, device=self.device)
            if self.anchor_model is not None and lambda_prox > 0:
                if use_sam:
                    # SAM on CE only; prox is not backprop'd through autograd in the SAM path.
                    with torch.no_grad():
                        prox_loss = self.compute_prox_loss(lambda_prox=lambda_prox)
                else:
                    prox_loss = self.compute_prox_loss(lambda_prox=lambda_prox)

            if self.args.debug_prox and (self.global_step % debug_every_n_steps == 0):
                delta_current = 0.0
                if self.anchor_model is not None and lambda_prox > 0:
                    delta_current = self.compute_prox_drift_l2()
                    lr = float(self.optimizer.param_groups[0].get('lr', 0.0))
                    self.debug_records.append({
                        'global_step': int(self.global_step),
                        'step_in_task': int(self.step_in_task),
                        'task_id': int(self.task_id),
                        'epoch': int(epoch),
                        'lr': lr,
                        'lambda_prox': lambda_prox,
                        'param_drift': float(delta_current),
                    })
            
            self.optimizer.zero_grad()
            if use_sam:
                # Pass A: compute perturbation direction from SAM batch CE gradient only.
                outputs_sam = self.model(sam_inputs)
                ce_loss_sam = self.criterion(outputs_sam, sam_targets)
                ce_loss_sam.backward()
            else:
                # Non-SAM: standard ER + prox on the concatenated batch.
                outputs = self.model(inputs_all)
                ce_loss = self.criterion(outputs, targets_all)
                total_loss = ce_loss + prox_loss
                total_loss.backward()
            
            # Step 2: SAM
            rho_val = 0.0
            sam_debug = {}
            ce_loss_mem = None
            g_norm_prox = 0.0
            p_norm_prox = 0.0
            r_t = 0.0
            if use_sam:
                # SAM needs a second backward pass:
                # 1) use current grads to compute the perturbation (virtual ascent),
                # 2) recompute grads at perturbed weights, then restore weights and step with those grads.
                coupling_mode = str(getattr(self.args, 'coupling_mode', 'delta_ema')).lower()
                if coupling_mode in {'grad_ratio', 'grad-ratio', 'ratio'}:
                    g_norm_prox = self.compute_task_grad_norm_prox_subset()
                    if torch.is_tensor(prox_loss) and lambda_prox > 0:
                        delta = float((2.0 * prox_loss / float(lambda_prox)).clamp_min(0.0).sqrt().item())
                    else:
                        delta = delta_current
                    p_norm_prox = float(lambda_prox) * float(delta)
                    eps = float(getattr(self.args, 'grad_ratio_eps', 1e-12))
                    r_t = float(p_norm_prox) / (float(g_norm_prox) + eps)

                rho = self.get_prox_informed_rho(
                    lambda_prox,
                    prox_loss=prox_loss,
                    grad_norm=g_norm_prox if g_norm_prox > 0 else None,
                )
                rho_val = rho
                sam_debug = self.sam_step(self.criterion, sam_inputs, sam_targets, rho) or {}

                # Split-batch: add replay gradients at the unperturbed weights (no SAM on memory).
                if sam_batch_mode == 'split' and buf_inputs is not None:
                    outputs_mem = self.model(buf_inputs)
                    ce_loss_mem = self.criterion(outputs_mem, buf_targets)
                    ce_loss_mem.backward()
                
                # Restore prox gradient after SAM:
                # sam_step() recomputes grads using only the criterion loss at perturbed weights, so it
                # zeroes out the prox contribution. For (lambda/2)*||w-w0||^2, d/dw is lambda*(w-w0).
                self.add_prox_grad_(lambda_prox=lambda_prox)
            
            self.optimizer.step()

            # Logging losses (avoid extra forwards under SAM).
            if use_sam:
                if sam_batch_mode == 'all':
                    ce_loss = ce_loss_sam.detach()
                elif sam_batch_mode == 'split':
                    if ce_loss_mem is None:
                        ce_loss = ce_loss_sam.detach()
                    else:
                        bs_cur = int(inputs_current.size(0))
                        bs_mem = int(buf_inputs.size(0))
                        ce_loss = (
                            ce_loss_sam.detach() * bs_cur + ce_loss_mem.detach() * bs_mem
                        ) / float(max(1, bs_cur + bs_mem))
                else:  # current-only
                    ce_loss = ce_loss_sam.detach()
                total_loss = ce_loss + prox_loss.detach()
            else:
                total_loss = total_loss.detach()

            if debug_sam and (debug_every_n_steps > 0) and (self.global_step % debug_every_n_steps == 0):
                lr = float(self.optimizer.param_groups[0].get('lr', 0.0))
                record = {
                    'global_step': int(self.global_step),
                    'step_in_task': int(self.step_in_task),
                    'task_id': int(self.task_id),
                    'epoch': int(epoch),
                    'use_sam': bool(use_sam),
                    'sam_phase': sam_phase,
                    'sam_batch_mode': sam_batch_mode,
                    'sam_bn_mode': str(getattr(self.args, 'sam_bn_mode', 'momentum0')).lower(),
                    'rho': float(rho_val),
                    'lr': lr,
                    'lambda_prox': float(lambda_prox),
                    'ce_loss': float(ce_loss.detach().item()),
                    'prox_loss': float(prox_loss.detach().item()) if torch.is_tensor(prox_loss) else float(prox_loss),
                    'total_loss': float(total_loss.detach().item()),
                    'param_norm': float(p_norm),
                    'delta': float(self.delta_last),
                    'delta_ema': float(self.delta_ema) if self.delta_ema is not None else 0.0,
                    'coupling_mode': str(getattr(self.args, 'coupling_mode', 'delta_ema')).lower(),
                    'g_norm_prox': float(g_norm_prox),
                    'p_norm_prox': float(p_norm_prox),
                    'r_t': float(r_t),
                }
                for k, v in (sam_debug or {}).items():
                    if isinstance(v, (int, float, bool, str)):
                        record[k] = v
                self.debug_records.append(record)
            
            # Per-batch SG tracking: call hook every 50 steps during first 500 steps of task
            for p in self.plugins:
                p.after_training_step(self)
            
            # Update Metrics
            if use_sam and sam_batch_mode == 'current':
                meter_n = int(inputs_current.size(0))
            else:
                meter_n = int(inputs_all.size(0))

            losses.update(float(total_loss.item()), meter_n)
            ce_losses.update(float(ce_loss.item()), meter_n)
            prox_losses.update(float(prox_loss.item()), meter_n)
            rhos.update(float(rho_val), meter_n)
            drifts.update(float(self.delta_last), meter_n)
            drift_emas.update(float(self.delta_ema) if self.delta_ema is not None else 0.0, meter_n)

        return {
            'loss': losses.avg,
            'ce_loss': ce_losses.avg,
            'prox_loss': prox_losses.avg,
            'rho': rhos.avg,
            'delta': drifts.avg,
            'delta_ema': drift_emas.avg,
            'param_norm': param_norms.avg
        }
