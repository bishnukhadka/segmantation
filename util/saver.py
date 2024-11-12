import os
import shutil
import torch
from collections import OrderedDict
import glob

class Saver(object):

    def __init__(self, args):
        self.args = args
        # self.directory = os.path.join('run', args.dataset, args.checkname)
        # run/train/project/name
        self.directory = os.path.join('run', args.project, args.checkname )
        self.best_model = None
        self.best_model_path=None
        
        self.runs = sorted(glob.glob(os.path.join(self.directory, 'experiment_*')))
        run_id = int(self.runs[-1].split('_')[-1]) + 1 if self.runs else 0

        self.experiment_dir = os.path.join(self.directory, 'experiment_{}'.format(str(run_id)))
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        """Saves checkpoint to disk"""
        filename = os.path.join(self.experiment_dir, filename)
        torch.save(state, filename)
        if is_best:
            best_pred = state['best_pred']
            # save best.pt
            # model_name = "epoch"+str(state['epoch'])+"score"+formatted_pred+'.pt'
            model_name = 'best.pt'
            model_path = os.path.join(self.experiment_dir, 'weights')
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            model_path = os.path.join(model_path, model_name)
            if self.best_model:
                # Save only the underlying model to avoid pickling issues with DataParallel
                # If self.best_model is wrapped in DataParallel, save self.best_model.module instead
                torch.save(self.best_model.module if isinstance(self.best_model, torch.nn.DataParallel) else self.best_model, model_path)
                self.best_model_path = model_path
            else: 
                print("Model couldn't be saved due to self.best_model being None")


            with open(os.path.join(self.experiment_dir, 'best_pred.txt'), 'w') as f:
                f.write(str(best_pred))
            if self.runs:
                previous_miou = [0.0]
                for run in self.runs:
                    run_id = run.split('_')[-1]
                    path = os.path.join(self.directory, 'experiment_{}'.format(str(run_id)), 'best_pred.txt')

                    if os.path.exists(path):
                        with open(path, 'r') as f:
                            miou = float(f.readline())
                            previous_miou.append(miou)
                    else:
                        continue
                max_miou = max(previous_miou)
                if best_pred > max_miou:
                    shutil.copyfile(filename, os.path.join(self.directory, 'model_best.pth.tar'))
            else:
                shutil.copyfile(filename, os.path.join(self.directory, 'model_best.pth.tar'))

    def save_experiment_config(self):
        logfile = os.path.join(self.experiment_dir, 'parameters.txt')
        log_file = open(logfile, 'w')
        p = OrderedDict()
        p['datset'] = self.args.dataset
        p['backbone'] = self.args.backbone #need to add if it is resnet101 or other not just resnet
        p['out_stride'] = self.args.out_stride
        p['lr'] = self.args.lr
        p['lr_scheduler'] = self.args.lr_scheduler
        p['loss_type'] = self.args.loss_type
        p['epoch'] = self.args.epochs
        p['img'] = self.args.img
        p['crop_size'] = self.args.crop_size

        for key, val in p.items():
            log_file.write(key + ':' + str(val) + '\n')
        log_file.close()