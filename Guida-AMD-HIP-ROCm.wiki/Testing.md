# This section is dedicated to testing software to see if everything is working properly. If anyone tests anything please return your results.

* Installing Bitsandbytes through this repository (pip install https://github.com/wasd-tech/Guida-AMD-HIP-ROCm/releases/download/ROCm/bitsandbytes-0.43.3.dev0-cp312-cp312-linux_x86_64.whl)

* Use [fluxgym-rocm](https://github.com/wasd-tech/fluxgym-rocm) to train FLUX LoRA:

    * It gives strange behavior, It seems to work, but it uses a lot of RAM. The only way to use it is to pass UNET safetensors that are already in FP8 format ro limit the overflow of RAM. Maybe is related to an issue concerning too high vram usage for ROCm.