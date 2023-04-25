# MACTA

> [**MACTA: A Multi-agent Reinforcement Learning Approach for Cache Timing Attacks and Detection**](https://openreview.net/forum?id=CDlHZ78-Xzi)\
> Jiaxun Cui, Xiaomeng Yang*, Mulong Luo*, Geunbae Lee*, Peter Stone, Hsien-Hsin S. Lee, Benjamin Lee, Edward Suh, Wenjie Xiong^, Yuandong Tian^\
> The Eleventh International Conference on Learning Representations (_ICLR 2023_)\
> \*Equal Second-author Contribution, ^Equal Supervising

[Paper](https://openreview.net/pdf?id=CDlHZ78-Xzi) | [GitHub](https://github.com/facebookresearch/macta) | [Bibtex](#citation)

## Dependencies

* MACTA is based on [AutoCAT](https://github.com/facebookresearch/AutoCAT).
* The Cache Gym Environment we used is based on an [open source CacheSimulator](https://github.com/auxiliary/CacheSimulator) from [auxiliary](https://github.com/auxiliary).
* MACTA uses [RLMeta](https://github.com/facebookresearch/rlmeta) as the RL framework.
* MACTA uses [SPEC 2017](https://www.spec.org/cpu2017/) to generate benign traces. Please check the license and follow the [instructions here](https://code.vt.edu/bearhw-public/rl-mem-trace) to generate benign traces.

## Installation

Create conda environment with python 3.7+.

```
conda create -n macta python=3.7
conda activate macta
```

Please follow the [PyTorch Get Started](https://pytorch.org/get-started/locally/) website to install PyTorch with proper CUDA version. One example is listed below, the other PyTorch versions may also work.

```
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=10.2 -c pytorch
```

The enviroment needs [moolib](https://github.com/facebookresearch/moolib) as the RPC backend for distributed RL training. Please follow the moolib installation instructions.
We recommend building moolib from source with the following steps.

```
git clone https://github.com/facebookresearch/moolib
cd moolib
git submodule sync && git submodule update --init --recursive
pip install .
```

There are several breaking backward compatibility changes in [OpenAI Gym](https://github.com/openai/gym) after 0.26.0. Please see the [release note](https://github.com/openai/gym/releases/tag/0.26.0) for details. We are using the old step APIs so please install OpenAI Gym 0.25.2.

```
pip install gym\[atari,accept-rom-license\]==0.25.2
```

The RL trainer is based on [RLMeta](https://github.com/facebookresearch/rlmeta).

The commit number of RLMeta we are using is [dc36de7983f8c9df70b957551efb8573126debee](https://github.com/facebookresearch/rlmeta/commit/dc36de7983f8c9df70b957551efb8573126debee).

Please install RLMeta by the following steps.

```
git clone https://github.com/facebookresearch/rlmeta
cd rlmeta
git checkout dc36de7983f8c9df70b957551efb8573126debee
git submodule sync && git submodule update --init --recursive
pip install -e .
```

Then check out MACTA and install the other requirements.

```
git clone https://github.com/facebookresearch/macta
cd macta
git submodule sync && git submodule update --init --recursive
pip install -r requirements.txt

```

## Quick Start with Pre-trained Models
We provide pretrained models of all methods, checkout the `checkpoints/`.
To run our pretrained model, simply modify the absolute path to the checkpoints in `src/rlemta/macta/config/sample_multiagent.yaml` and run
```
cd src/rlmeta/macta
conda activate macta
python sample_multiagent.py
```

## Benign Trace Generation
If you want to use [SPEC 2017](https://www.spec.org/cpu2017/), please make sure you have the license to use it and follow the [instructions here](https://code.vt.edu/bearhw-public/rl-mem-trace) to generate the traces. To use the traces, specify the path to the trace files in the configs. You can also test some open-source datasets as well.

## Training
To train MACTA, please modify the `attacker_checkpoint` and `trace_files` config in `src/rlmeta/macta/config/macta.yaml` to the absolute path of the files. Then run the following commands.
```
conda activate macta
cd src/rlmeta/macta
python train/train_macta.py
```

## Evaluation
Please specify the agents and evaluation parameters the config in `src/rlmeta/macta/config/sample_multiagent.yaml`
```
cd src/rlmeta/macta
conda activate macta
python sample_multiagent.py
```

## Citation
```bibtex
@inproceedings{
  cui2023macta,
  title={{MACTA}: A Multi-agent Reinforcement Learning Approach for Cache Timing Attacks and Detection},
  author={Jiaxun Cui and Xiaomeng Yang and Mulong Luo and Geunbae Lee and Peter Stone and Hsien-Hsin S. Lee and Benjamin Lee and G. Edward Suh and Wenjie Xiong and Yuandong Tian},
  booktitle={The Eleventh International Conference on Learning Representations },
  year={2023},
  url={https://openreview.net/forum?id=CDlHZ78-Xzi}
}
```

### License

This project is under the GNU General Public License (GPL) v2.0 license. See [LICENSE](LICENSE) for details.
