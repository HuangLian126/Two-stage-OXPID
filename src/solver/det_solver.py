'''
by lyuwenyu
'''
import time 
import json
import datetime
import math

import torch 

from src.misc import dist
from src.data import get_coco_api_from_dataset

from .solver import BaseSolver
from .det_engine import train_one_epoch, evaluate


class DetSolver(BaseSolver):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.best_stat = {'epoch': -1, 'coco_eval_bbox': -1.0}

    def state_dict(self, last_epoch):
        state = super().state_dict(last_epoch)
        state['best_stat'] = self.best_stat.copy()
        return state

    def load_state_dict(self, state):
        super().load_state_dict(state)
        # Older checkpoints have no best score; track it from the next evaluation.
        self.best_stat = state.get(
            'best_stat', {'epoch': -1, 'coco_eval_bbox': -1.0}
        ).copy()
    
    def fit(self, ):
        print("Start training")
        self.train()

        args = self.cfg 
        
        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('number of params:', n_parameters)

        base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)

        start_time = time.time()
        for epoch in range(self.last_epoch + 1, args.epoches):
            if dist.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)
            
            train_stats = train_one_epoch(
                self.model, self.criterion, self.train_dataloader, self.optimizer, self.device, epoch,
                args.clip_max_norm, print_freq=args.log_step, ema=self.ema, scaler=self.scaler)

            self.lr_scheduler.step()

            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module, self.criterion, self.postprocessor, self.val_dataloader, base_ds, self.device, self.output_dir
            )

            # COCO bbox AP averaged over IoU=0.50:0.95, not AP at IoU=0.50 alone.
            bbox_map = float(test_stats['coco_eval_bbox'][0])
            is_best = math.isfinite(bbox_map) and bbox_map >= 0 and \
                bbox_map > self.best_stat['coco_eval_bbox']
            if is_best:
                self.best_stat = {'epoch': epoch, 'coco_eval_bbox': bbox_map}
            print('best_stat: ', self.best_stat)

            if self.output_dir and dist.is_main_process():
                checkpoint_paths = [self.output_dir / 'checkpoint.pth']
                if (epoch + 1) % args.checkpoint_step == 0:
                    checkpoint_paths.append(self.output_dir / f'checkpoint{epoch:04}.pth')
                if is_best:
                    checkpoint_paths.append(self.output_dir / 'best.pth')
                # Save after evaluation so every checkpoint includes the latest best score.
                state = self.state_dict(epoch)
                for checkpoint_path in checkpoint_paths:
                    dist.save_on_master(state, checkpoint_path)
                if is_best:
                    print(f'Saved best.pth: epoch={epoch}, bbox mAP={bbox_map:.6f}')


            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,
                        'best_epoch': self.best_stat['epoch'],
                        'best_coco_eval_bbox': self.best_stat['coco_eval_bbox'],
                        'n_parameters': n_parameters}

            if self.output_dir and dist.is_main_process():
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

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))


    def val(self, ):
        self.eval()

        base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)
        
        module = self.ema.module if self.ema else self.model
        test_stats, coco_evaluator = evaluate(module, self.criterion, self.postprocessor,
                self.val_dataloader, base_ds, self.device, self.output_dir)
                
        if self.output_dir:
            dist.save_on_master(coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth")
        
        return
