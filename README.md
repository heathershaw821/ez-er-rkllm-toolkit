# EZ-ER-RKLLM-Toolkit2

## Backstory

I got tired of manually downloading models from HuggingFace using git-lfs, authenticating every time, waiting for that finish, and then FINALLY manually inputting the model source and destination into a Python script, and then wait for THAT to finish inside of a Docker container before moving onto the next one and starting the process all over again.

As a result, I wrote these two scripts to download models from HuggingFace, convert them, pull all .json files from the original repo, and then insert a block of text into the model card (see below) before finally uploading everything to HuggingFace.

Original repo is here: [RKLLM](https://github.com/airockchip/rknn-llm)

Initial testing was done using Pelochus' EZ RKNN-LLM container found in this repo: [ezrknn-llm](https://github.com/Pelochus/ezrknn-llm/)

For more information, and useful links, please check out the [RockchipNPU subreddit](https://reddit.com/r/RockchipNPU)

Conversion tests done on consumer grade hardware:

- AMD Ryzen 3 1200 Quad-Core Processor
- GA-AX370-Gaming K5 Motherboard
- 2 x G.SKILL Ripjaws V Series DDR4 RAM 32GB, 64GB total
- NVIDIA GeForce GTX 780 (not used in this experiment)

Also performed conversion using an Intel X5650 CPU, which uses DDR3 RAM and does not support AVX.

## How to use

There is only an interactive script included here. I have also included version 1.1.0 of RKLLM-Toolkit, since it contains all of the dependencies required to run these scripts (except for inquirer.)

To get started, clone this repository:

```bash
git clone https://github.com/heathershaw821/ez-er-rkllm.git
```

To do a one-shot conversion in an interactive shell:

### CPU

```bash
cd docker
docker build -t $(whoami)/rkllm-interactive . && docker run -it --rm $(whoami)/rkllm-interactive
```

### GPU _(CUDA)_

your system will need a compatible nvidia card, drivers, and [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed.

First check what version of CUDA your system is running with:

```bash
nvidia-smi
```

Then look up an appropriate base image on https://hub.docker.com/r/nvidia/cuda/tags with a matching `major.minor` version.

For this example, my system reports CUDA version `12.4`, and the latest compatible version is `12.4.1`. Based on this I pick image `cuda:12.4.1-devel-ubuntu22.04`

```bash
cd docker
docker build -t $(whoami)/rkllm-interactive --build-arg BASE_IMAGE=nvidia/cuda:12.4.1-devel-ubuntu22.04 . && docker run -it --gpus all --rm $(whoami)/rkllm-interactive
```

The script should detect CUDA is installed inside the container and attempt to use it.

### Caching

You may want to store the downloaded models, or the generated files, or both.

To do so mount a folder to `/root/toolkit/models` in the container, and when asked in the interactive script, tell it not to cleanup the files.

```bash
docker run -it --gpus all --volume ./.cache:/root/toolkit/models --rm $(whoami)/rkllm-interactive
```

## Changing the model card template

Of course, feel free to adjust the model card template under the HubHelpers class, which is available in both:

```python
    def build_card(self, export_path):
        """
        Inserts text into the README.md file of the original model, after the model data.
        Using the HF built-in functions kept omitting the card's model data,
        so gonna do this old school.
        """
        self.model_name = self.model_id.split("/", 1)[1]
        self.card_in = ModelCard.load(self.model_id)
        self.card_out = export_path + "README.md"
        self.template = f'---\n' + \
            f'{self.card_in.data.to_yaml()}\n' + \
            f'---\n' + \
            f'# {self.model_name}-{self.platform.upper()}-{self.rkllm_version}\n\n' + \
            f'This version of {self.model_name} has been converted to run on the {self.platform.upper()} NPU using {self.qtype} quantization.\n\n' + \
            f'This model has been optimized with the following LoRA: {self.lora_id}\n\n' + \
            f'Compatible with RKLLM version: {self.rkllm_version}\n\n' + \
            f'###Useful links:\n' + \
            f'[Official RKLLM GitHub](https://github.com/airockchip/rknn-llm) \n\n' + \
            f'[RockhipNPU Reddit](https://reddit.com/r/RockchipNPU) \n\n' + \
            f'[EZRKNN-LLM](https://github.com/Pelochus/ezrknn-llm/) \n\n' + \
            f'Pretty much anything by these folks: [marty1885][https://github.com/marty1885] and [happyme531](https://huggingface.co/happyme531) \n\n' + \
            f'Converted using https://github.com/heathershaw821/ez-er-rkllm-toolkit2 \n\n' + \
            f'# Original Model Card for base model, {self.model_name}, below:\n\n' + \
            f'{self.card_in.text}'
        try:
            ModelCard.save(self.template, self.card_out)
        except RuntimeError as e:
            print(f"Runtime Error: {e}")
        except RuntimeWarning as w:
            print(f"Runtime Warning: {w}")
        else:
            print(f"Model card successfully exported to {self.card_out}!")
            c = open(self.card_out, 'r')
            print(c.read())
            c.close()
```

## Utilization

Model conversion utilizes anywhere from 2-4x the size of the original model, which means that you need an equal amount of memory. I compensated for this with swap files of varying size. Since I just leave the process running overnight (I have low upload speeds,) the performance hit from using swap files vs partitions doesn't bother me much. If performance is critical, I would recommend at least 192GB - 512GB of DDR4 RAM with a lot of cores to handle especially large models. For evaluation and chat simulation, a CPU with AVX\* support is also recommended.

## Compatibility and Testing

Models converted using the Python 3.10 and RKLLM v1.1.1 packages do appear to be backwards compatible with the v1.1.0 runtime! So far, only [Llama 3.2 3B Instruct](https://huggingface.co/c01zaut/Llama-3.2-3B-Instruct-rk3588-1.1.1/blob/main/Llama-3.2-3B-Instruct-rk3588-w8a8_g128-opt-0-hybrid-ratio-1.0.rkllm) has been tested. Check out [u/DimensionUnlucky4046](https://www.reddit.com/user/DimensionUnlucky4046/)'s pipeline in this [Reddit thread](https://www.reddit.com/r/RockchipNPU/comments/1gi2web/llama3_for_rk3588_available/)

## To do

- Test with LoRA
- Test with full RAG pipeline
- Test with multimodal models
