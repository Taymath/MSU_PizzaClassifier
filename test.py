import torch
import pandas as pd
from omegaconf import OmegaConf
from data.dataset import create_test_dataloader_with_labels
from models.models import ConvNet

def predict_and_save(model, test_loader, device, submission_csv_path="D:\\MSU\\Coding\\PYTHON\\7 term\\main_pizza_classifier\\data\\submission.csv"):
    model.eval()
    correct = 0
    total = 0

    updated_ids = []
    updated_preds = []

    with torch.no_grad():
        for images, img_ids, true_labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu()

            updated_ids.extend(img_ids.numpy())
            updated_preds.extend(preds.numpy())

            correct += (preds == true_labels).sum().item()
            total += len(true_labels)

    accuracy = 100.0 * correct / total
    print(f"Test accuracy: {accuracy:.2f}%")

    submission_df = pd.read_csv(submission_csv_path)

    for idx, pred_label in zip(updated_ids, updated_preds):
        submission_df.loc[submission_df['id'] == idx, 'label'] = pred_label

    submission_df.to_csv(submission_csv_path, index=False)
    print(f"File updated {submission_csv_path}.")


def main(config_path="D:\\MSU\\Coding\\PYTHON\\7 term\\main_pizza_classifier\\configs\\config.yaml", model_path="model_best.pth"):
    cfg = OmegaConf.load(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    classes_csv = pd.read_csv(cfg.data.classes_csv)
    num_classes = len(classes_csv)
    print("num_classes =", num_classes)

    test_loader = create_test_dataloader_with_labels(
        test_img_dir=cfg.data.test_img_dir,
        submission_csv=cfg.data.test_csv,
        batch_size=cfg.data.batch_size,
        image_size=cfg.data.image_size,
        num_workers=cfg.data.num_workers
    )

    model = ConvNet(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Model downloaded from {model_path}")

    predict_and_save(model, test_loader, device, submission_csv_path=cfg.data.test_csv)
    print("Done!")


if __name__ == "__main__":
    import sys
    config_path = "D:\\MSU\\Coding\\PYTHON\\7 term\\main_pizza_classifier\\configs\\config.yaml"
    model_path = "model_best.pth"


    if "--config" in sys.argv:
        idx = sys.argv.index("--config")
        config_path = sys.argv[idx + 1]
    if "--model" in sys.argv:
        idx = sys.argv.index("--model")
        model_path = sys.argv[idx + 1]

    main(config_path, model_path)
