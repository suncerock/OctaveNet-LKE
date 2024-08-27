import os
from datetime import datetime

import argparse
import yaml
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from data import get_split_list, get_SWD_dataloader
from model import LKEModel


def get_time():
    return datetime.now().strftime("[%m/%d - %H:%M:%S]")


def train_one_split(
    train_list,
    val_list,
    save_dir,

    config,
):

    train_seg_hop_length = config["data"].pop("train_seg_hop_length")
    val_seg_hop_length = config["data"].pop("val_seg_hop_length")
    train_dataloader = get_SWD_dataloader(
        piece_list=train_list, seg_hop_length=train_seg_hop_length, shuffle=True, **config["data"])
    val_dataloader = get_SWD_dataloader(
        piece_list=val_list, seg_hop_length=val_seg_hop_length, shuffle=False, **config["data"])
    config["data"]["train_seg_hop_length"] = train_seg_hop_length
    config["data"]["val_seg_hop_length"] = val_seg_hop_length

    lr = config["trainer"]["lr"]
    max_epoch = config["trainer"]["max_epoch"]
    max_patience = config["trainer"]["max_patience"]
    device = config["trainer"]["device"]

    model = LKEModel(**config["model"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    num_params = sum([np.prod(x.shape) for x in model.parameters()])
    print(f"{num_params / 1e3:.2f}K parameters!")
    writer = SummaryWriter(log_dir=os.path.join(save_dir, "tensorboard"))

    best_acc = 0
    ckpt_path = None
    patience = max_patience

    for epoch in range(max_epoch):
        print(f"{get_time()} | Epoch: {epoch} Training...")
        model.train()
        for batch_idx, batch in enumerate(train_dataloader):
            num_steps = len(train_dataloader) * epoch + batch_idx + 1
            optimizer.zero_grad()

            batch["x"] = batch["x"].to(device)
            batch["y"] = batch["y"].to(device)

            loss, y_pred = model.train_step(batch)
            loss.backward()
            optimizer.step()

            writer.add_scalars("Loss", dict(train=loss.item()), num_steps)
            writer.add_scalars("lr", dict(
                lr=optimizer.param_groups[0]["lr"]), num_steps)

        print(f"{get_time()} | Epoch: {epoch} Validating...")

        model.eval()
        val_loss = []
        correct, count = 0, 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_dataloader):
                batch["x"] = batch["x"].to(device)
                batch["y"] = batch["y"].to(device)

                loss, y_pred = model.val_step(batch)

                val_loss.append(loss.item())

                count += torch.count_nonzero(batch["y"] != -1).item()
                correct += torch.count_nonzero(
                    y_pred.argmax(dim=1)[batch["y"] != -1] == batch["y"][batch["y"] != -1]
                ).item()

        acc = correct / count * 100
        val_loss = sum(val_loss) / len(val_loss)

        print(f"{get_time()} | Epoch: {epoch} Validation Loss={val_loss:.2f} Accuracy={acc:.2f}")

        writer.add_scalars("Loss", dict(val=val_loss), num_steps)
        writer.add_scalars("Accuracy", dict(val=acc), num_steps)
        writer.flush()

        if acc > best_acc:
            best_acc = acc
            if ckpt_path is not None:
                os.remove(ckpt_path)
            ckpt_path = os.path.join(
                save_dir, f"epoch={epoch:d}_loss={val_loss:.3f}_acc={acc:.2f}.pth")
            torch.save(model.state_dict(), ckpt_path)
            patience = max_patience
        else:
            patience -= 1

        if patience == 0:
            break


def train(config):
    split = config["data"].pop("split")
    train_list, val_list, test_list = get_split_list(split)
    num_split = len(train_list)
    assert len(val_list) == num_split and len(test_list) == num_split

    base_save_dir = config["trainer"].pop("save_dir")

    for i in range(num_split):
        print(f"{get_time()} | Split: {i}")
        train_one_split(
            train_list=train_list[i],
            val_list=val_list[i],
            save_dir=os.path.join(f"{base_save_dir}/split_{i}"),
            config=config
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="config.yaml")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    dataset_path = config["data"]["dataset_path"]
    save_dir = config["trainer"]["save_dir"]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(os.path.join(save_dir, "config.yaml"), 'w', encoding="utf-8") as f:
        yaml.dump(config, f)

    train(config)
