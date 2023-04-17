from pathlib import Path
from livepipeline import init_camera_live_pipeline

model_path = Path("TACTUS-live\\tactus_live\data\model\pickle.json")
rstp_url = ".\Video_for_hilda\\groundPOV.MP4"
computing_device = "cuda:0"
flag_save = False

init_camera_live_pipeline(model_path, rstp_url, computing_device, flag_save)
