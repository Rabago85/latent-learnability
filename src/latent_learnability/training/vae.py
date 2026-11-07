import torch
from latent_learnability.data_sources.mnist import MNISTSampler
from latent_learnability.models.vae.model import VAE
from latent_learnability.trainers.vae_trainers import MNISTVAETrainer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_steps = 5000
run_name = "raw_kl_mog"
beta = 0.005 # [0.000488, 0.001, 0.005]
mnist = MNISTSampler().to(device)
vae = VAE(
    data_channels=1,
    hidden_channels=[16, 32, 64, 128],
    k=43,
    kl_type = run_name,
    beta=beta,  # NOT settled - see README, tried 10.0 / 0.1 / 0.0 so far
    prior_override = None
).to(device)

trainer = MNISTVAETrainer(
    mnist_sampleable=mnist,
    batch_size=64,
)
losses, steps = trainer.train(
    model=vae,
    run_name = run_name + "_GN_43_" + str(beta),
    num_steps=num_steps,
    lr=1e-3,
    warmup_steps=500,
    ckpt_every=250,#250
)
plt.plot(steps, losses)
plt.xlabel("Step")
plt.ylabel("Loss")
plt.show()