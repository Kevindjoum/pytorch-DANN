import torch
import train
import mnist
import mnistm
import model


def main():
    source_train_loader = mnist.mnist_train_loader
    target_train_loader = mnistm.mnistm_train_loader

    # Use GPU if available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    encoder = model.Extractor().to(device)
    classifier = model.Classifier().to(device)
    discriminator = model.Discriminator().to(device)

    train.source_only(encoder, classifier, source_train_loader, target_train_loader, device)
    train.dann(encoder, classifier, discriminator, source_train_loader, target_train_loader, device)


if __name__ == "__main__":
    main()
