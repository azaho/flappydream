# flappydream

To reproduce results, follow the steps:

1. Run extract_rollouts.py to extract training rollouts
2. Run extract_rollouts_test.py to extract test rollouts
3. Run train_vae.py to train the AE on training rollouts


Important note: even though the code uses the word "VAE", the loss is instead the default autoencoder loss albeit with noisy sampling.