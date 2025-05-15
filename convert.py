import json

with open("ds2_dense/deepscores_train.json") as f:
    data = json.load(f)
coco_images = []
for img in data["images"]:
    coco_images.append({
        "id": img["id"],
        "file_name": img["filename"],
        "width": img["width"],
        "height": img["height"]
    })

coco_annotations = []
ann_id = 1
for ann_key, ann in data["annotations"].items():
    if not isinstance(ann, dict): continue
    image_id = int(ann["img_id"])
    cat_ids = ann["cat_id"]
    if isinstance(cat_ids, list):
        cat_id = int(cat_ids[0])  
    else:
        cat_id = int(cat_ids)

    x1, y1, x2, y2 = ann["a_bbox"]
    w = x2 - x1
    h = y2 - y1
    coco_annotations.append({
        "id": ann_id,
        "image_id": image_id,
        "category_id": cat_id,
        "bbox": [x1, y1, w, h],
        "area": w * h,
        "iscrowd": 0
    })
    ann_id += 1

coco_categories = []
for cat_id, cat in data["categories"].items():
    try:
        coco_categories.append({
            "id": int(cat_id),
            "name": cat["name"]
        })
    except:
        continue

coco_dict = {
    "info": data.get("info", {}),
    "images": coco_images,
    "annotations": coco_annotations,
    "categories": coco_categories
}

with open("ds2_dense/deepscores_train_coco.json", "w") as f:
    json.dump(coco_dict, f)
