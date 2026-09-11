from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR


class WarmupCosineLR(SequentialLR):
    def __init__(
            self, 
            optimizer, 
            warmup_epochs: int, 
            max_epochs: int, 
            start_factor: float = 0.1, 
            eta_min: float = 1e-6, 
            last_epoch: int = -1
            ):
        
        warmup = LinearLR(
            optimizer, 
            start_factor=start_factor, 
            total_iters=warmup_epochs
        )
        
        cosine = CosineAnnealingLR(
            optimizer, 
            T_max=(max_epochs - warmup_epochs), 
            eta_min=eta_min
        )
        
        super().__init__(
            optimizer, 
            schedulers=[warmup, cosine], 
            milestones=[warmup_epochs], 
            last_epoch=last_epoch
        )