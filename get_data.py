from huggingface_hub import snapshot_download

local_dir = "./data"

snapshot_download(
    repo_id="doannvptit/cv_pascal_voc", 
    repo_type="dataset", 
    local_dir=local_dir,
    local_dir_use_symlinks=False  # Quan trọng: False để tải file thực tế thay vì link ảo
)