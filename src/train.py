import argparse
import os
import torch
from torch import optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from .dataset import GTSRBDataset
from .model import Net
from .utils import save_checkpoint, accuracy


def parse_args():
    parser = argparse.ArgumentParser(description='Train a GTSRB classifier')
    parser.add_argument('--csv', default='data/GTSRB/Train.csv')
    parser.add_argument('--root', default='data/GTSRB')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--save-dir', default='models')
    parser.add_argument('--use-roi', action='store_true', help='Crop images using ROI coords if present in CSV')
    return parser.parse_args()


def main():
    args = parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])

    dataset = GTSRBDataset(args.csv, args.root, transform=transform, use_roi=args.use_roi)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = Net(num_classes=43).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        avg_loss = running_loss / len(dataset)
        print(f"Epoch {epoch} - Loss: {avg_loss:.4f}")

        checkpoint_path = os.path.join(args.save_dir, f"checkpoint_epoch_{epoch}.pth")
        os.makedirs(args.save_dir, exist_ok=True)
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }, filename=checkpoint_path)

        # optionally evaluate on validation set here

    # final save
    final_path = os.path.join(args.save_dir, 'best_model.pth')
    save_checkpoint({'epoch': args.epochs, 'state_dict': model.state_dict()}, filename=final_path)


if __name__ == '__main__':
    main()
