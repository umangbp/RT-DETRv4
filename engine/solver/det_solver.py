"""
RT-DETRv4: Painlessly Furthering Real-Time Object Detection with Vision Foundation Models
Copyright (c) 2025 The RT-DETRv4 Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
"""

import time
import json
import datetime
import math
import hashlib
import shutil
import subprocess
from pathlib import Path

import torch

from ..misc import dist_utils, stats

from ._solver import BaseSolver
from .det_engine import train_one_epoch, evaluate
from ..optim.lr_scheduler import FlatCosineLRScheduler


class DetSolver(BaseSolver):

    def fit(self, ):
        self.train()
        args = self.cfg

        n_parameters, model_stats = stats(self.cfg)
        print(model_stats)
        print("-"*42 + "Start training" + "-"*43)

        self.self_lr_scheduler = False
        if args.lrsheduler is not None:
            iter_per_epoch = len(self.train_dataloader)
            print("     ## Using Self-defined Scheduler-{} ## ".format(args.lrsheduler))
            self.lr_scheduler = FlatCosineLRScheduler(self.optimizer, args.lr_gamma, iter_per_epoch, total_epochs=args.epoches,
                                                warmup_iter=args.warmup_iter, flat_epochs=args.flat_epoch, no_aug_epochs=args.no_aug_epoch)
            self.self_lr_scheduler = True
        n_parameters = sum([p.numel() for p in self.model.parameters() if p.requires_grad])
        print(f'number of trainable parameters: {n_parameters}')

        top1 = 0
        best_stat = {'epoch': -1, }
        best_result = None
        best_result_path = (
            self.output_dir / "training_best.json"
            if self.output_dir
            else None
        )

        def record_best_result(
            checkpoint_name,
            epoch,
            metric_name,
            metric_value,
        ):
            nonlocal best_result

            best_result = {
                "checkpoint": checkpoint_name,
                "epoch": int(epoch),
                "metric_name": metric_name,
                "metric_value": float(metric_value),
                "weights": "ema" if self.ema else "model",
            }

            if (
                best_result_path is not None
                and dist_utils.is_main_process()
            ):
                with best_result_path.open(
                    "w",
                    encoding="utf-8",
                ) as f:
                    json.dump(
                        best_result,
                        f,
                        indent=2,
                        sort_keys=True,
                    )
                    f.write("\n")

        # Restore exact best-checkpoint provenance when resuming.
        if (
            self.last_epoch > 0
            and best_result_path is not None
            and best_result_path.is_file()
        ):
            with best_result_path.open(
                "r",
                encoding="utf-8",
            ) as f:
                best_result = json.load(f)

            # Restore the true global-best metric across resumes.
            top1 = float(best_result["metric_value"])

        # evaluate again before resume training
        if self.last_epoch > 0:
            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.evaluator,
                self.device
            )
            for k in test_stats:
                best_stat['epoch'] = self.last_epoch
                best_stat[k] = test_stats[k][0]

                # Preserve the historical global best when resuming.
                if best_result is None:
                    top1 = test_stats[k][0]
                else:
                    top1 = max(
                        top1,
                        float(best_result["metric_value"]),
                    )

                print(f'best_stat: {best_stat}')

        best_stat_print = best_stat.copy()
        start_time = time.time()
        start_epoch = self.last_epoch + 1
        for epoch in range(start_epoch, args.epoches):

            self.train_dataloader.set_epoch(epoch)
            # self.train_dataloader.dataset.set_epoch(epoch)
            if dist_utils.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)

            if epoch == self.train_dataloader.collate_fn.stop_epoch:
                self.load_resume_state(str(self.output_dir / 'best_stg1.pth'))
                self.ema.decay = self.train_dataloader.collate_fn.ema_restart_decay
                print(f'Refresh EMA at epoch {epoch} with decay {self.ema.decay}')

            train_stats, grad_percentages = train_one_epoch(
                self.self_lr_scheduler,
                self.lr_scheduler,
                self.model,
                self.criterion,
                self.train_dataloader,
                self.optimizer,
                self.device,
                epoch,
                max_norm=args.clip_max_norm,
                print_freq=args.print_freq,
                ema=self.ema,
                scaler=self.scaler,
                lr_warmup_scheduler=self.lr_warmup_scheduler,
                writer=self.writer,
                teacher_model=self.teacher_model, # NEW: Pass teacher model to train_one_epoch
            )

            if not self.self_lr_scheduler:  # update by epoch 
                if self.lr_warmup_scheduler is None or self.lr_warmup_scheduler.finished():
                    self.lr_scheduler.step()

            self.last_epoch += 1
            if dist_utils.is_main_process() and hasattr(self.criterion, 'distill_adaptive_params') and \
                self.criterion.distill_adaptive_params and self.criterion.distill_adaptive_params.get('enabled', False):

                params = self.criterion.distill_adaptive_params
                default_weight = params.get('default_weight')

                avg_percentage = sum(grad_percentages) / len(grad_percentages) if grad_percentages else 0.0

                current_weight = self.criterion.weight_dict.get('loss_distill', 0.0)
                new_weight = current_weight
                reason = 'unchanged'

                if avg_percentage < 1e-6:
                    if default_weight is not None:
                        new_weight = default_weight
                        reason = 'reset_to_default_zero_grad'
                elif epoch >= self.train_dataloader.collate_fn.stop_epoch:
                    if default_weight is not None:
                        new_weight = default_weight
                        reason = 'ema_phase_default'
                else:
                    rho = params['rho']
                    delta = params['delta']
                    lower_bound = rho - delta
                    upper_bound = rho + delta
                    if not (lower_bound <= avg_percentage <= upper_bound):
                        target_percentage = upper_bound if avg_percentage < lower_bound else lower_bound
                        if current_weight > 1e-6:
                            p_current = avg_percentage / 100.0
                            p_target = target_percentage / 100.0
                            numerator = p_target * (1.0 - p_current)
                            denominator = p_current * (1.0 - p_target)
                            if abs(denominator) >= 1e-9:
                                ratio = numerator / denominator
                                ratio = max(ratio, 0.1)  # clamp non-positive to 0.1
                                new_weight = current_weight * ratio
                                new_weight = min(max(new_weight, current_weight / 10.0), current_weight * 10.0)
                                reason = f'adjusted_to_{target_percentage:.2f}%'

                if abs(new_weight - current_weight) > 0:
                    self.criterion.weight_dict['loss_distill'] = new_weight
                print(f"Epoch {epoch}: avg encoder grad {avg_percentage:.2f}% | distill {current_weight:.6f} -> {new_weight:.6f} ({reason})")

            if self.output_dir and epoch < self.train_dataloader.collate_fn.stop_epoch:
                checkpoint_paths = [self.output_dir / 'last.pth']
                # extra checkpoint before LR drop and every 100 epochs
                if (epoch + 1) % args.checkpoint_freq == 0:
                    checkpoint_paths.append(self.output_dir / f'checkpoint{epoch:04}.pth')
                for checkpoint_path in checkpoint_paths:
                    dist_utils.save_on_master(self.state_dict(), checkpoint_path)

            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.evaluator,
                self.device
            )

            # TODO
            for k in test_stats:
                if self.writer and dist_utils.is_main_process():
                    for i, v in enumerate(test_stats[k]):
                        self.writer.add_scalar(f'Test/{k}_{i}'.format(k), v, epoch)

                if k in best_stat:
                    best_stat['epoch'] = epoch if test_stats[k][0] > best_stat[k] else best_stat['epoch']
                    best_stat[k] = max(best_stat[k], test_stats[k][0])
                else:
                    best_stat['epoch'] = epoch
                    best_stat[k] = test_stats[k][0]

                if best_stat[k] > top1:
                    best_stat_print['epoch'] = epoch
                    top1 = best_stat[k]

                    if self.output_dir:
                        if epoch >= self.train_dataloader.collate_fn.stop_epoch:
                            checkpoint_name = 'best_stg2.pth'
                        else:
                            checkpoint_name = 'best_stg1.pth'

                        dist_utils.save_on_master(
                            self.state_dict(),
                            self.output_dir / checkpoint_name
                        )

                        record_best_result(
                            checkpoint_name=checkpoint_name,
                            epoch=epoch,
                            metric_name=k,
                            metric_value=top1,
                        )

                best_stat_print[k] = max(best_stat[k], top1)
                print(f'best_stat: {best_stat_print}')  # global best

                if best_stat['epoch'] == epoch and self.output_dir:
                    if epoch >= self.train_dataloader.collate_fn.stop_epoch:
                        if test_stats[k][0] > top1:
                            top1 = test_stats[k][0]

                            dist_utils.save_on_master(
                                self.state_dict(),
                                self.output_dir / 'best_stg2.pth'
                            )

                            record_best_result(
                                checkpoint_name='best_stg2.pth',
                                epoch=epoch,
                                metric_name=k,
                                metric_value=top1,
                            )
                    else:
                        previous_top1 = top1
                        top1 = max(test_stats[k][0], top1)

                        dist_utils.save_on_master(
                            self.state_dict(),
                            self.output_dir / 'best_stg1.pth'
                        )

                        if test_stats[k][0] >= previous_top1:
                            record_best_result(
                                checkpoint_name='best_stg1.pth',
                                epoch=epoch,
                                metric_name=k,
                                metric_value=top1,
                            )

                elif epoch >= self.train_dataloader.collate_fn.stop_epoch:
                    best_stat = {'epoch': -1, }
                    self.ema.decay -= 0.0001
                    self.load_resume_state(str(self.output_dir / 'best_stg1.pth'))
                    print(f'Refresh EMA at epoch {epoch} with decay {self.ema.decay}')


            log_stats = {
                **{f'train_{k}': v for k, v in train_stats.items()},
                **{f'test_{k}': v for k, v in test_stats.items()},
                'epoch': epoch,
                'n_parameters': n_parameters
            }

            if self.output_dir and dist_utils.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                # for evaluation logs
                if coco_evaluator is not None:
                    (self.output_dir / 'eval').mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ['latest.pth']
                        if epoch % 50 == 0:
                            filenames.append(f'{epoch:03}.pth')
                        for name in filenames:
                            torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                    self.output_dir / "eval" / name)

        if (
            self.output_dir
            and dist_utils.is_main_process()
            and best_result is not None
        ):
            source_config = Path(self.cfg.source_config_path)
            config_snapshot = self.output_dir / "training_config.yml"

            shutil.copy2(
                source_config,
                config_snapshot,
            )

            def sha256_file(path):
                digest = hashlib.sha256()

                with open(path, "rb") as f:
                    for chunk in iter(
                        lambda: f.read(1024 * 1024),
                        b"",
                    ):
                        digest.update(chunk)

                return digest.hexdigest()

            checkpoint_path = (
                self.output_dir
                / best_result["checkpoint"]
            )

            if not checkpoint_path.is_file():
                raise RuntimeError(
                    "Best checkpoint recorded by training does not exist: "
                    f"{checkpoint_path}"
                )

            try:
                git_commit = subprocess.check_output(
                    ["git", "rev-parse", "HEAD"],
                    text=True,
                ).strip()
            except Exception:
                git_commit = None

            try:
                git_dirty = bool(
                    subprocess.check_output(
                        ["git", "status", "--porcelain"],
                        text=True,
                    ).strip()
                )
            except Exception:
                git_dirty = None

            result = {
                "schema_version": 1,
                "status": "completed",
                "source": {
                    "git_commit": git_commit,
                    "git_dirty": git_dirty,
                },
                "config": {
                    "file": "training_config.yml",
                    "sha256": sha256_file(config_snapshot),
                },
                "result": {
                    "checkpoint": best_result["checkpoint"],
                    "checkpoint_sha256": sha256_file(
                        checkpoint_path
                    ),
                    "epoch": best_result["epoch"],
                    "metric": {
                        "name": best_result["metric_name"],
                        "value": best_result["metric_value"],
                    },
                    "weights": best_result["weights"],
                },
            }

            result_path = (
                self.output_dir
                / "training_result.json"
            )

            with result_path.open(
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(
                    result,
                    f,
                    indent=2,
                    sort_keys=True,
                )
                f.write("\n")

            print(
                f"Training result written to {result_path}"
            )

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))


    def val(self, ):
        self.eval()

        module = self.ema.module if self.ema else self.model
        test_stats, coco_evaluator = evaluate(module, self.criterion, self.postprocessor,
                self.val_dataloader, self.evaluator, self.device)

        if self.output_dir:
            dist_utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth")

        return


    def state_dict(self):
        """State dict, train/eval"""
        state = {}
        state['date'] = datetime.datetime.now().isoformat()

        # For resume
        state['last_epoch'] = self.last_epoch

        for k, v in self.__dict__.items():
            if k == 'teacher_model':
                continue
            if hasattr(v, 'state_dict'):
                v = dist_utils.de_parallel(v)
                state[k] = v.state_dict()

        return state