import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import ToTensor, Resize, Compose, Normalize
from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_fpn,
    FasterRCNN_MobileNet_V3_Large_FPN_Weights
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.cuda.amp import autocast, GradScaler
from PIL import Image
import json
import os
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class DeepScoresDataset(Dataset):
    def __init__(self, imgs_dir, ann_file, transforms=None):
        print("Initializing dataset...")
        self.imgs_dir = imgs_dir
        coco = json.load(open(ann_file))

        # remap category IDs 
        self.cat2idx = {c['id']: i + 1 for i, c in enumerate(coco.get('categories', []))}
        self.num_categories = len(self.cat2idx)

        self.images = {img['id']: img.get('file_name', img.get('filename'))
                       for img in coco.get('images', [])}
        self.anns_per_img = {}
        for ann in coco.get('annotations', []):
            self.anns_per_img.setdefault(ann['image_id'], []).append(ann)

        self.ids = list(self.images.keys())
        self.transforms = transforms or (lambda x: x)
        print(f"Loaded {len(self.ids)} images.")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_path = os.path.join(self.imgs_dir, self.images[img_id])
        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size

        anns = self.anns_per_img.get(img_id, [])
        boxes, labels = [], []
        for ann in anns:
            x, y, w, h = ann['bbox']
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(self.cat2idx[ann['category_id']])

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        img = self.transforms(img)
        _, new_h, new_w = img.shape

        scale_x, scale_y = new_w / orig_w, new_h / orig_h
        boxes *= torch.tensor([scale_x, scale_y, scale_x, scale_y])

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([img_id])
        }
        return img, target


def collate_fn(batch):
    return tuple(zip(*batch))


def visualize_predictions(model, dataloader, device,
                          save_dir="vis_preds", num_images=5, score_thresh=0.5):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    print(f"Visualizing predictions (threshold={score_thresh})...")

    count = 0
    with torch.no_grad():
        for imgs, targets in dataloader:
            imgs = [img.to(device) for img in imgs]
            preds = model(imgs)

            for img_tensor, pred, tgt in zip(imgs, preds, targets):
                img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
                fig, ax = plt.subplots(1)
                ax.imshow(img_np)

                for box, score, label in zip(pred["boxes"],
                                             pred["scores"],
                                             pred["labels"]):
                    if score < score_thresh:
                        continue
                    x1, y1, x2, y2 = box.cpu()
                    rect = patches.Rectangle(
                        (x1, y1), x2 - x1, y2 - y1,
                        linewidth=2, edgecolor="lime", facecolor="none"
                    )
                    ax.add_patch(rect)
                    ax.text(x1, y1, f"{label.item()}:{score:.2f}",
                            color="lime", fontsize=8)

                img_id = int(tgt["image_id"].item())
                save_path = os.path.join(save_dir, f"image_{img_id}.png")
                plt.axis("off")
                plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
                plt.close(fig)

                count += 1
                if count >= num_images:
                    return


def main():
    img_dir = "ds2_dense/images"
    train_json = "ds2_dense/deepscores_train_coco.json"
    test_json = "ds2_dense/deepscores_test_coco.json"

    # ImageNet normalization
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transforms = Compose([
        Resize((512, 512)),
        ToTensor(),
        Normalize(mean=mean, std=std),
    ])


    full_ds = DeepScoresDataset(img_dir, train_json, transforms=transforms)
    idx2cat = {v: k for k, v in full_ds.cat2idx.items()}
    num_classes = full_ds.num_categories + 1  # include background class

    total = len(full_ds)
    val_size = int(0.2 * total)
    train_size = total - val_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size])
    train_loader = DataLoader(
        train_ds, batch_size=8, shuffle=True,
        collate_fn=collate_fn, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=8, shuffle=False,
        collate_fn=collate_fn, num_workers=0, pin_memory=True
    )

    model = fasterrcnn_mobilenet_v3_large_fpn(
        weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT,
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    backbone_params = list(model.backbone.parameters())
    for param in backbone_params[:-20]:
        param.requires_grad = False
    for param in backbone_params[-20:]:
        param.requires_grad = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=0.007, momentum=0.9, weight_decay=0.0005
    )
    scaler = GradScaler()

    num_epochs = 40
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0
        print(f"Epoch {epoch} training...")
        for i, (imgs, targets) in enumerate(train_loader):
            imgs = [img.to(device) for img in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            optimizer.zero_grad()
            with autocast():
                losses = model(imgs, targets)
                loss = sum(losses.values())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            if i % 10 == 0:
                print(f"  Batch {i}: Loss = {loss.item():.4f}")

        val_loss = 0.0
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs = [img.to(device) for img in imgs]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                with autocast():
                    losses = model(imgs, targets)
                val_loss += sum(losses.values()).item()

        print(f"Epoch {epoch}/{num_epochs} - "
              f"Train Loss: {train_loss/len(train_loader):.4f} - "
              f"Val Loss: {val_loss/len(val_loader):.4f}")


    test_ds = DeepScoresDataset(img_dir, test_json, transforms=transforms)
    test_loader = DataLoader(
        test_ds, batch_size=8, shuffle=False,
        collate_fn=collate_fn, num_workers=0, pin_memory=True
    )

    model.eval()
    coco_gt = COCO(test_json)
    coco_results = []
    with torch.no_grad(), autocast():
        for imgs, targets in test_loader:
            imgs = [img.to(device) for img in imgs]
            preds = model(imgs)
            for tgt, pred in zip(targets, preds):
                image_id = int(tgt['image_id'].item())
                img_info = coco_gt.loadImgs(image_id)[0]
                orig_w, orig_h = img_info['width'], img_info['height']
                scale_x, scale_y = orig_w / 512, orig_h / 512
                for box, score, label in zip(pred['boxes'],
                                             pred['scores'],
                                             pred['labels']):
                    x1, y1, x2, y2 = box.cpu().tolist()
                    x1 *= scale_x; y1 *= scale_y
                    x2 *= scale_x; y2 *= scale_y
                    orig_cat = idx2cat[label.item()]
                    coco_results.append({
                        "image_id": image_id,
                        "category_id": orig_cat,
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": float(score)
                    })

    with open('ds2_dense/test_preds.json', 'w') as f:
        json.dump(coco_results, f)

    coco_dt = coco_gt.loadRes('ds2_dense/test_preds.json')
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()



if __name__ == '__main__':
    main()
¯