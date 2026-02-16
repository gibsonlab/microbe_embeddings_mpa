from typing import *
import torch


def parse_cuda_device_ids(cuda_device_ids: str) -> List[torch.device]:
    cuda_device_ids = [int(x) for x in cuda_device_ids.split(",") if len(x) > 0]
    if len(cuda_device_ids) == 0:
        print(f"At least one CUDA device ID must be specified. Got: {cuda_device_ids}")
        exit(1)

    cuda_devices = []
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"Total CUDA devices available: {device_count}")

        for device_id in cuda_device_ids:
            if device_id < device_count:
                print(f"CUDA device :{device_id} exists and is available.")
                # You can now create a device object for it
                device = torch.device(f"cuda:{device_id}")
                cuda_devices.append(device)
            else:
                print(f"CUDA device :{device_id} does not exist. Only devices 0 to {device_count - 1} are available.")
                exit(1)
    else:
        print("CUDA is not available on this system.")

    assert len(cuda_devices) > 0, "Unexpected error: parsed zero CUDA devices."
    return cuda_devices
