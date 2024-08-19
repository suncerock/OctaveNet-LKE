import os
import yaml

import pandas as pd
import torch

from data import get_split_list, get_SWD_dataloader, ALL_SONGS, ALL_VERSIONS, INDEX_TO_KEY
from model import LKEModel


def inference(
    model_dir,
    output_dir,
    
    dataset_path="/ssddata2/yiwei/Schubert_Winterreise_Dataset_v1-2",
    split="song",

    hop_length=0.2,

    device="cuda"
):
    _, _, test_list = get_split_list(split)
    with open(os.path.join(model_dir, "config.yaml")) as f:
        config = yaml.safe_load(f)

    data = []
    for version in ALL_VERSIONS:
        for song in ALL_SONGS:
            data.append(dict(version=version, song=song, acc=0.0))
    df = pd.DataFrame(data)

    os.makedirs(output_dir, exist_ok=True)
    detailed_output_dir = os.path.join(output_dir, "detailed_output")
    os.makedirs(detailed_output_dir, exist_ok=True)

    for i in range(len(test_list)):
        print("Evaluating split: ", i)
        ckpt_dir = os.path.join(model_dir, "split_{:d}".format(i))
        
        sort_key = lambda x:float(x.replace(".pth", "").split("=")[-1])
        filename = sorted(
            [name for name in os.listdir(ckpt_dir) if name.endswith(".pth")],
            key=sort_key)

        if len(filename) > 1:
            print("Multiple checkpoint found! Using {}".format(filename[-1]))
        ckpt_path = os.path.join(ckpt_dir, filename[-1])

        model = LKEModel(**config["model"])
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        model.to(device)
        model.eval()

        test_dataloader = get_SWD_dataloader(
            dataset_path=dataset_path, piece_list=test_list[i], seg_length=-1, seg_hop_length=-1, hop_length=hop_length,
            batch_size=1, shuffle=False, num_workers=0
        )

        for batch_idx, batch in enumerate(test_dataloader):
            with torch.no_grad():
                batch["x"] = batch["x"].to(device)
                batch["y"] = batch["y"].to(device)

                piece = test_list[i][batch_idx]
                song, version = piece.replace("Schubert_D911-", "").split("_")

                y_pred = model.pred_step(batch).squeeze(dim=0)
                y_pred = torch.softmax(y_pred, dim=0)
                prob, output = torch.max(y_pred, dim=0)

                data = {
                    "output": pd.Series(output.detach().cpu().numpy()).apply(INDEX_TO_KEY.__getitem__),
                    "prob": pd.Series(prob.detach().cpu().numpy())
                }

                target = pd.Series(batch["y"][0].detach().cpu().numpy())
                target[target != -1] = target.apply(INDEX_TO_KEY.__getitem__)
                target[target == -1] = "nan"
                data["ann"] = target

                pd.DataFrame(data).to_csv(os.path.join(detailed_output_dir, "{}.csv".format(piece)), index=False)

                target = batch["y"][0]
                
                count = torch.count_nonzero(target != -1).item()
                correct = torch.count_nonzero(output[target != -1] == target[target != -1]).item()
                
                acc = correct / count * 100

                df.loc[(df["song"] == song) & (df["version"] == version), "acc"] = acc

    df.to_csv(os.path.join(output_dir, "summary.csv"), index=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--dataset_path", type=str, default="/ssddata2/yiwei/Schubert_Winterreise_Dataset_v1-2")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--split", type=str, default="song")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    inference(
        model_dir=args.model_dir, 
        output_dir=args.output_dir,
        dataset_path=args.dataset_path,
        split=args.split,
        device=args.device
    )
