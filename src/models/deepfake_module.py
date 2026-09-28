
from typing import Any, Dict, List, Optional, Tuple

import torch
from lightning import LightningModule
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from torchmetrics import MaxMetric, MeanMetric, MinMetric
from torchmetrics.classification.accuracy import Accuracy


class SSLDeepFakeLitModule(LightningModule):
    """LightningModule for SSL-based deepfake detection."""

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.net = net
        self.criterion = torch.nn.CrossEntropyLoss()

        self.train_acc = Accuracy(task="multiclass", num_classes=2)
        self.val_acc = Accuracy(task="multiclass", num_classes=2)
        self.test_acc = Accuracy(task="multiclass", num_classes=2)

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.val_acc_best = MaxMetric()
        self.val_eer_best = MinMetric()

        self.train_scores: List[torch.Tensor] = []
        self.train_targets: List[torch.Tensor] = []
        self.val_scores: List[torch.Tensor] = []
        self.val_targets: List[torch.Tensor] = []
        self.test_scores: List[torch.Tensor] = []
        self.test_targets: List[torch.Tensor] = []

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.net(x)

    def on_train_start(self) -> None:
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()
        self.val_eer_best.reset()

    def on_train_epoch_start(self) -> None:
        self.train_scores.clear()
        self.train_targets.clear()

    def on_validation_epoch_start(self) -> None:
        self.val_scores.clear()
        self.val_targets.clear()

    def on_test_epoch_start(self) -> None:
        self.test_scores.clear()
        self.test_targets.clear()

    def model_step(
        self, batch: Tuple[Dict[str, torch.Tensor], torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        scores = torch.softmax(logits, dim=1)[:, 1]
        return loss, preds, y, scores

    @staticmethod
    def _compute_eer(scores: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = scores.detach().float().flatten().cpu()
        targets = targets.detach().long().flatten().cpu()

        spoof_mask = targets == 1
        bonafide_mask = targets == 0
        if scores.numel() == 0 or not bool(spoof_mask.any()) or not bool(bonafide_mask.any()):
            nan = torch.tensor(float("nan"))
            return nan, nan

        thresholds = torch.unique(scores).sort().values
        thresholds = torch.cat(
            [
                thresholds.new_tensor([float("inf")]),
                thresholds.flip(0),
                thresholds.new_tensor([-float("inf")]),
            ]
        )

        preds_spoof = scores.unsqueeze(0) >= thresholds.unsqueeze(1)
        far = preds_spoof[:, bonafide_mask].float().mean(dim=1)
        frr = (~preds_spoof[:, spoof_mask]).float().mean(dim=1)
        idx = torch.argmin(torch.abs(far - frr))
        eer = (far[idx] + frr[idx]) / 2.0
        return eer, thresholds[idx]

    def _gather_epoch_outputs(
        self, scores: List[torch.Tensor], targets: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        local_scores = torch.cat(scores).to(self.device) if scores else torch.empty(0, device=self.device)
        local_targets = (
            torch.cat(targets).long().to(self.device) if targets else torch.empty(0, dtype=torch.long, device=self.device)
        )

        if self.trainer.world_size <= 1:
            return local_scores.cpu(), local_targets.cpu()

        local_len = torch.tensor([local_scores.numel()], device=self.device)
        lengths = self.all_gather(local_len).flatten()
        max_len = int(lengths.max().item())

        padded_scores = torch.zeros(max_len, device=self.device, dtype=local_scores.dtype)
        padded_targets = torch.zeros(max_len, device=self.device, dtype=local_targets.dtype)
        padded_scores[: local_scores.numel()] = local_scores
        padded_targets[: local_targets.numel()] = local_targets

        gathered_scores = self.all_gather(padded_scores)
        gathered_targets = self.all_gather(padded_targets)

        score_chunks = []
        target_chunks = []
        for rank_idx, length in enumerate(lengths.tolist()):
            score_chunks.append(gathered_scores[rank_idx, :length].cpu())
            target_chunks.append(gathered_targets[rank_idx, :length].cpu())

        return torch.cat(score_chunks), torch.cat(target_chunks)

    def training_step(
        self, batch: Tuple[Dict[str, torch.Tensor], torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss, preds, targets, scores = self.model_step(batch)
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.train_scores.append(scores.detach().cpu())
        self.train_targets.append(targets.detach().cpu())
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        self._log_learning_rates()
        return loss

    def _log_learning_rates(self) -> None:
        if not self.trainer.optimizers:
            return

        param_groups = self.trainer.optimizers[0].param_groups
        if not param_groups:
            return

        self.log("train/lr", param_groups[0]["lr"], on_step=True, on_epoch=False, prog_bar=True)
        if len(param_groups) > 1:
            for group_idx, param_group in enumerate(param_groups):
                self.log(
                    f"train/lr_group_{group_idx}",
                    param_group["lr"],
                    on_step=True,
                    on_epoch=False,
                    prog_bar=False,
                )

    def on_train_epoch_end(self) -> None:
        scores, targets = self._gather_epoch_outputs(self.train_scores, self.train_targets)
        eer, threshold = self._compute_eer(scores, targets)
        self.log("train/eer", eer.to(self.device), prog_bar=True, sync_dist=False)
        self.log("train/eer_threshold", threshold.to(self.device), sync_dist=False)

    def validation_step(
        self, batch: Tuple[Dict[str, torch.Tensor], torch.Tensor], batch_idx: int
    ) -> None:
        loss, preds, targets, scores = self.model_step(batch)
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.val_scores.append(scores.detach().cpu())
        self.val_targets.append(targets.detach().cpu())
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        acc = self.val_acc.compute()
        self.val_acc_best(acc)
        scores, targets = self._gather_epoch_outputs(self.val_scores, self.val_targets)
        eer, threshold = self._compute_eer(scores, targets)
        if not torch.isnan(eer):
            self.val_eer_best(eer)
            self.log("val/eer_best", self.val_eer_best.compute().to(self.device), prog_bar=True)
        self.log("val/eer", eer.to(self.device), prog_bar=True, sync_dist=False)
        self.log("val/eer_threshold", threshold.to(self.device), sync_dist=False)
        self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(
        self, batch: Tuple[Dict[str, torch.Tensor], torch.Tensor], batch_idx: int
    ) -> None:
        loss, preds, targets, scores = self.model_step(batch)
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.test_scores.append(scores.detach().cpu())
        self.test_targets.append(targets.detach().cpu())
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        scores, targets = self._gather_epoch_outputs(self.test_scores, self.test_targets)
        eer, threshold = self._compute_eer(scores, targets)
        self.log("test/eer", eer.to(self.device), prog_bar=True, sync_dist=False)
        self.log("test/eer_threshold", threshold.to(self.device), sync_dist=False)
        self._log_test_distribution_plot(scores, targets, eer, threshold)

    @rank_zero_only
    def _log_test_distribution_plot(
        self, scores: torch.Tensor, targets: torch.Tensor, eer: torch.Tensor, threshold: torch.Tensor
    ) -> None:
        if scores.numel() == 0 or torch.isnan(eer) or not self.logger:
            return

        try:
            import matplotlib.pyplot as plt
            import wandb
        except ImportError:
            return

        if wandb.run is None:
            return

        bonafide_scores = scores[targets == 0].numpy()
        spoof_scores = scores[targets == 1].numpy()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(bonafide_scores, bins=60, alpha=0.65, density=True, label="bonafide")
        ax.hist(spoof_scores, bins=60, alpha=0.65, density=True, label="spoof")
        ax.axvline(float(threshold), color="black", linestyle="--", linewidth=2, label=f"EER threshold={float(threshold):.4f}")
        ax.set_xlabel("Spoof probability")
        ax.set_ylabel("Density")
        ax.set_title(f"Test score distributions - EER={float(eer):.4f}")
        ax.legend()
        ax.grid(alpha=0.2)
        wandb.log({"test/eer_distribution": wandb.Image(fig)})
        plt.close(fig)

    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
