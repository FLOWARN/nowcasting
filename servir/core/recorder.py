import os
import numpy as np
import torch
from servir.utils.main_utils import  print_log


class Recorder:
    def __init__(self, verbose=True, delta=0, patience=10):
        self.verbose = verbose
        self.best_score = None
        self.val_loss_min = np.Inf
        self.delta = delta
        self.torlerance = patience
        self.patience = patience
        self.stop = False

    def __call__(self, val_loss):

        update_model = False

        score = -val_loss

        if (self.best_score is None) or (score >= self.best_score - self.delta):

            update_model = True

            if self.verbose:
                print_log(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). best model updated.\n')

            # update best score and minimum validation loss
            self.best_score = score
            self.val_loss_min = val_loss

            # reset patience
            self.patience = self.torlerance
        
        else:
            # update patience
            self.patience -= 1
            if self.patience <= 0:
                self.stop = True
                if self.verbose:
                    print_log(f'Early stopping. Validation loss did not decrease for the last {self.torlerance} epochs.\n')
            else:
                if self.verbose:
                    print_log(f'Validation loss higher than current lowest {self.val_loss_min}, best model not updated.\n')

        return update_model, self.stop
