import torch
from pathlib import Path
import os

def save_model(model: torch.nn.Module, target_dir: str, model_name: str):
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(exist_ok=True,parents=True)

    assert model_name.endswith(".pth") or model_name.endswith(".pt")

    count_saved_model_files = 0
    for i in os.listdir(target_dir_path):
        if i.endswith("pt") or i.endswith("pth"):
            count_saved_model_files += 1
            
    model_save_path = target_dir_path / (f"{count_saved_model_files}_" + model_name)
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(),f=model_save_path)