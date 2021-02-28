from ...fastai1.basics import *


class HyperParameterTuner:
    def __init__(self, learn: Learner):
        self.learn = learn

    def find_optimized_lr_pre(self):
        self.learn.lr_find()
        self.learn.recorder.plot(suggestion=True)
        lr = self.learn.recorder.min_grad_lr
        return lr

    def find_optimized_lr(self, lr_diff: int = 5, loss_threshold: float = .05, adjust_value: float = 1,
                          plot: bool = False) -> float:
        # Run the Learning Rate Finder
        self.learn.lr_find()

        # Get loss values and their corresponding gradients, and get lr values
        losses = np.array(self.learn.recorder.losses)
        assert (lr_diff < len(losses))
        loss_grad = np.gradient(losses)
        lrs = self.learn.recorder.lrs

        # Search for index in gradients where loss is lowest before the loss spike
        # Initialize right and left idx using the lr_diff as a spacing unit
        # Set the local min lr as -1 to signify if threshold is too low
        local_min_lr = -1
        r_idx = -1
        l_idx = r_idx - lr_diff
        while (l_idx >= -len(losses)) and (abs(loss_grad[r_idx] - loss_grad[l_idx]) > loss_threshold):
            local_min_lr = lrs[l_idx]
            r_idx -= 1
            l_idx -= 1

        lr_to_use = local_min_lr * adjust_value

        if plot:
            ## Default set to False
            # plots the gradients of the losses in respect to the learning rate change
            plt.plot(loss_grad)
            plt.plot(len(losses) + l_idx, loss_grad[l_idx], markersize=10, marker='o', color='red')
            plt.ylabel("Loss")
            plt.xlabel("Index of LRs")
            plt.show()

            plt.plot(np.log10(lrs), losses)
            plt.ylabel("Loss")
            plt.xlabel("Log 10 Transform of Learning Rate")
            loss_coord = np.interp(np.log10(lr_to_use), np.log10(lrs), losses)
            plt.plot(np.log10(lr_to_use), loss_coord, markersize=10, marker='o', color='red')
            plt.show()

        return lr_to_use
