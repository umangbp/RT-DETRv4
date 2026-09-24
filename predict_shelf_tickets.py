import argparse
import json
from pathlib import Path

import torch
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F
from PIL import Image, ImageDraw

from engine.core import YAMLConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--threshold", type=float, default=0.40)
    parser.add_argument(
        "--config",
        default="configs/rtv4/rtv4_hgnetv2_m_shelf_ticket.yml",
    )
    parser.add_argument(
        "--checkpoint",
        default="outputs/rtv4_hgnetv2_m_shelf_ticket_bootstrap_v1/best_stg2.pth",
    )
    parser.add_argument(
        "--output",
        default="outputs/bootstrap_v1_predictions",
    )
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    cfg = YAMLConfig(args.config)

    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    model = cfg.model.to(device)
    postprocessor = cfg.postprocessor.to(device)

    ckpt = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    if "ema" in ckpt and "module" in ckpt["ema"]:
        state = ckpt["ema"]["module"]
        source = "EMA"
    else:
        state = ckpt["model"]
        source = "model"

    model.load_state_dict(state, strict=True)
    model.eval()
    postprocessor.eval()

    print(f"Device: {device}")
    print(f"Weights: {source}")
    print(
        f"Checkpoint last_epoch metadata: "
        f"{ckpt.get('last_epoch')}"
    )

    input_path = Path(args.input)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input path does not exist: {input_path}"
        )

    if input_path.is_dir():
        image_paths = sorted(
            p
            for p in input_path.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
    else:
        if input_path.suffix.lower() not in {
            ".jpg",
            ".jpeg",
            ".png",
        }:
            raise RuntimeError(
                f"Unsupported image file: {input_path}"
            )

        image_paths = [input_path]

    if not image_paths:
        raise RuntimeError(
            f"No images found in {input_path}"
        )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Images: {len(image_paths)}")
    print(f"Threshold: {args.threshold:.2f}")
    print()

    # Matches the validation transform:
    # Resize size: [640, 640]
    resize = T.Resize((640, 640))

    all_predictions = []

    for path in image_paths:
        image = Image.open(path).convert("RGB")
        orig_w, orig_h = image.size

        x = resize(image)
        x = F.pil_to_tensor(x).float() / 255.0
        x = x.unsqueeze(0).to(device)

        with torch.inference_mode():
            outputs = model(x)

            # This repository stores orig_size as [width, height].
            # The postprocessor repeats this to [W, H, W, H]
            # for scaling XYXY boxes back to original pixels.
            orig_size = torch.tensor(
                [[orig_w, orig_h]],
                dtype=torch.float32,
                device=device,
            )

            result = postprocessor(
                outputs,
                orig_size,
            )[0]

        scores = result["scores"].cpu()
        labels = result["labels"].cpu()
        boxes = result["boxes"].cpu()

        draw = ImageDraw.Draw(image)

        kept = 0
        image_predictions = []

        for score, label, box in zip(
            scores,
            labels,
            boxes,
        ):
            score = float(score)

            if score < args.threshold:
                continue

            x1, y1, x2, y2 = map(float, box)

            # Clamp to original image boundaries.
            x1 = max(0, min(x1, orig_w - 1))
            y1 = max(0, min(y1, orig_h - 1))
            x2 = max(0, min(x2, orig_w - 1))
            y2 = max(0, min(y2, orig_h - 1))

            # Ignore invalid boxes after clamping.
            if x2 <= x1 or y2 <= y1:
                continue

            image_predictions.append(
                {
                    "label": "shelf_ticket",
                    "label_id": int(label),
                    "score": score,
                    "bbox_xyxy": [
                        x1,
                        y1,
                        x2,
                        y2,
                    ],
                }
            )

            draw.rectangle(
                (x1, y1, x2, y2),
                outline="red",
                width=5,
            )

            draw.text(
                (x1 + 4, y1 + 4),
                f"{score:.3f}",
                fill="red",
            )

            kept += 1

        all_predictions.append(
            {
                "file_name": path.name,
                "width": orig_w,
                "height": orig_h,
                "predictions": image_predictions,
            }
        )

        output_path = output_dir / path.name
        image.save(
            output_path,
            quality=95,
        )

        max_score = (
            float(scores.max())
            if len(scores)
            else 0.0
        )

        print(
            f"{path.name}: "
            f"{kept} boxes >= "
            f"{args.threshold:.2f}, "
            f"max={max_score:.4f}"
        )

    json_path = output_dir / "predictions.json"

    with open(json_path, "w") as f:
        json.dump(
            all_predictions,
            f,
            indent=2,
        )

    print()
    print(f"Predictions JSON: {json_path}")
    print(f"Saved: {output_dir}")


if __name__ == "__main__":
    main()
