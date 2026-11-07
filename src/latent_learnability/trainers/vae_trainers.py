from .base import Trainer
import torch
from abc import ABC, abstractmethod
from typing import Optional, Tuple
import os


class LabeledSampleable(ABC):
    """
    Distribution which can be sampled from
    """
    @abstractmethod
    def sample(self, num_samples: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            - num_samples: the desired number of samples
        Returns:
            - samples: b d
            - labels: b
        """
        pass

class MNISTVAETrainer(Trainer):
    def __init__(self, mnist_sampleable: LabeledSampleable, batch_size: int = 64, **kwargs):
        super().__init__(**kwargs)
        self.mnist = mnist_sampleable
        self.batch_size = batch_size

    # def get_train_loss(self):
    #     """ See display 85 from the text """
    #     x, y = self.mnist.sample(self.batch_size)
    #     z, z_mean, z_std, x_mean, x_std = self.model(x)
    #     return self.model.compute_loss(z, z_mean, z_std, x_mean, x_std, x)

    def get_train_loss(self):
        """ See display 85 from the text """
        x, y = self.mnist.sample(self.batch_size)
        z, z_mean, z_std, x_mean, x_std = self.model(x)

        loss, self.last_metrics = self.model.compute_loss(
            z, z_mean, z_std, x_mean, x_std, x
        )
        return loss

    @torch.no_grad()
    def checkpoint(self, step: int):
        # Save model
        torch.save(self.model.state_dict(), os.path.join(self.output_dir, f'step_{step:06d}_model.pt'))
        torch.save(self.opt.state_dict(), os.path.join(self.output_dir, f'step_{step:06d}_opt.pt'))
        torch.save(self.metrics_history, os.path.join(self.output_dir, f'step_{step:06d}_metrics.pt'))


        # Save output visualization, using x_mean as reconstruction
        # b = 10
        # x, _ = self.mnist.sample(b)
        # _, _, _, x_mean, _ = self.model(x)
        # x_all = torch.cat([x, x_mean], dim=0)
        # grid = make_grid(x_all, nrow=b, normalize=True, value_range=(0, 1))
        # plt.imshow(grid.permute(1, 2, 0).cpu(), cmap="gray")
        # plt.axis("off")
        # plt.title("VAE Reconstruction")
        # plt.savefig(os.path.join(self.output_dir, f'step_{step:06d}_output.png'))
        # plt.close()
