from latent_learnability.data_sources.mnist import MNISTSampler
import torch


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mnist = MNISTSampler().to(device)
    xx, y = mnist.sample(5)
    print(xx.shape, y.shape)