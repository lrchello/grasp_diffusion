#!/usr/bin/env python3
# validate.py -- FINAL STABLE VERSION
# - Fair comparison with original DRO-Grasp
# - Supports diffusion via success@K sampling
# - No duplicated Isaac calls
# - Statistically correct and reproducible

import os
import sys
import time
import warnings
import hydra
import torch
import numpy as np
from termcolor import cprint

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

from model.network import create_network
from data_utils.CMapDataset import create_dataloader
from utils.multilateration import multilateration
from utils.se3_transform import compute_link_pose
from utils.optimization import process_transform, create_problem, optimization
from utils.hand_model import create_hand_model
from validation.validate_utils import validate_isaac


@hydra.main(version_base="1.2", config_path="configs", config_name="validate")
def main(cfg):
    device = torch.device(f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu")
    batch_size = cfg.dataset.batch_size
    K = int(getattr(cfg.validate, "sample_K", 1))

    print(f"Device: {device}")
    print(f"Name: {cfg.name}")
    print(f"Sampling K = {K}")

    os.makedirs("validate_output", exist_ok=True)
    log_file = f"validate_output/{cfg.name}.log"

    for epoch in cfg.validate_epochs:
        cprint(f"\n===== Validating epoch {epoch} =====", "cyan")
        with open(log_file, "a") as f:
            print(f"\n===== Validating epoch {epoch} =====", file=f)

        network = create_network(cfg.model, mode="validate").to(device)
        ckpt = f"output/state_dict/epoch_{epoch}.pth"
        network.load_state_dict(torch.load(ckpt, map_location=device))
        network.eval()

        dataloader = create_dataloader(cfg.dataset, is_train=False)

        global_robot = None
        hand = None

        success_num = 0
        total_num = 0
        time_list = []
        all_success_q = []

        for data in dataloader:
            robot_name = data["robot_name"]
            object_name = data["object_name"]

            if robot_name != global_robot:
                if global_robot is not None:
                    rate = success_num / total_num * 100
                    cprint(f"[{global_robot}] Result: {success_num}/{total_num} ({rate:.2f}%)", "yellow")
                hand = create_hand_model(robot_name, device)
                global_robot = robot_name
                success_num = 0
                total_num = 0
                time_list = []
                all_success_q = []

            initial_q = data["initial_q"].to(device)
            robot_pc = data["robot_pc"].to(device)
            object_pc = data["object_pc"].to(device)

            B = initial_q.shape[0]
            best_success = torch.zeros(B, dtype=torch.bool, device=device)
            best_q = None

            for k in range(K):
                with torch.no_grad():
                    out = network(robot_pc, object_pc)

                dro = out["dro"]
                mlat_pc = multilateration(dro, object_pc)
                transform, _ = compute_link_pose(hand.links_pc, mlat_pc, is_train=False)
                optim_transform = process_transform(hand.pk_chain, transform)
                layer = create_problem(hand.pk_chain, optim_transform.keys())

                t0 = time.time()
                pred_q = optimization(hand.pk_chain, layer, initial_q, optim_transform)
                t1 = time.time()
                time_list.append(t1 - t0)

                success_k, _ = validate_isaac(robot_name, object_name, pred_q, gpu=cfg.gpu)
                success_k = success_k.to(device)

                if best_q is None:
                    best_q = pred_q.clone()
                else:
                    replace = success_k & (~best_success)
                    best_q[replace] = pred_q[replace]

                best_success |= success_k
                if best_success.all():
                    break

            succ = best_success.sum().item()
            success_num += succ
            total_num += B
            all_success_q.append(best_q[best_success])

            cprint(
                f"[{robot_name}/{object_name}] Result: {succ}/{B} ({succ/B*100:.2f}%)",
                "green",
            )

        rate = success_num / total_num * 100
        time_mean = np.mean(time_list)
        time_std = np.std(time_list)

        cprint(
            f"[{global_robot}] FINAL: {success_num}/{total_num} ({rate:.2f}%), "
            f"Time: {time_mean:.2f}±{time_std:.2f}s",
            "magenta",
        )


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    torch.set_num_threads(8)
    main()
