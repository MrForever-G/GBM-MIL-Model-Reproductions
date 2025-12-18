import os
import time
import json
import argparse
from train_transmil import train_transmil


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt_dir", type=str, default="F:\ProjectWork\GBM-MIL-Model-Reproductions\CLAM-GBM-Classification\pt_files")
    parser.add_argument("--train_csv", type=str, default="train_sample_id.csv")
    parser.add_argument("--test_csv", type=str, default="test_sample_id.csv")
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--feat_dim", type=int, default=768)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()

    # timestamp
    time_tag = time.strftime("%Y-%m-%d-%H-%M-%S")
    save_dir = os.path.join("TrainProcess", time_tag)

    args.time_tag = time_tag
    args.save_dir = save_dir
    return args


def main():
    cfg = get_config()

    # stdout header (TwinGCN style)
    print(f"Running time tab: {cfg.time_tag}")
    print(f"Save Dir: {cfg.save_dir}")

    os.makedirs(cfg.save_dir, exist_ok=True)

    # save HParams JSON (TwinGCN style)
    hparam_dict = vars(cfg)
    json_path = os.path.join(cfg.save_dir, f"{cfg.time_tag}.json")
    with open(json_path, "w") as f:
        json.dump(hparam_dict, f, indent=4)

    # start training
    train_transmil(cfg)


if __name__ == "__main__":
    main()
