# recency-bias
Code and instructions for replicating experiments from ["Linear Recency Bias During Training Improves Transformers’ Fit to Reading Times" (Clark et al., COLING 2025)](https://arxiv.org/abs/2409.11250).

Feel free to contact [Christian Clark](mailto:clark.3664@osu.edu) with any questions.

## Pythia-based LM training

The steps for tranining Pythia-style LMs are based on the [slm\_surprisal](https://github.com/byungdoh/slm_surprisal/tree/main) repository.

1. Clone the [GPT-NeoX](https://github.com/EleutherAI/gpt-neox) repository, and revert it to the commit that was used in this work:
```
$ git clone https://github.com/EleutherAI/gpt-neox.git
$ cd gpt-neox
$ git reset --hard 038b01
```

2. Install the dependencies following the README of the resulting GPT-NeoX repository.

3. Copy all files under the `gpt-neox` directory of THIS repository to the GPT-NeoX repository. This will overwrite some files and add some new files.

4. Prepare the training data (1,000 batches from the Pile in this work) following the instructions outlined in the [EleutherAI Pythia](https://github.com/EleutherAI/pythia) repository. This should be a numpy array of size `(1000*1024, 2049)` saved in a `.bin` file under `gpt-neox/data` that can be loaded with the following line:
```
np.memmap(data_prefix+".bin", dtype=np.uint16, mode="r", order="C", shape=(1000*1024, 2049))
```

5. Run the command `python deepy.py train_slms.py CONFIG_FILE` (e.g. `python deepy.py train_slms.py configs/pythia-2-4-256-1k.yml`) under the GPT-NeoX repository to launch LM training.
Refer to the README of the GPT-NeoX repository for an explanation of each argument in the configuration.
See also gpt-neox/slurm.sh (in this repository) for an example slurm script.

## Surprisal estimation on psycholinguistic corpora

[LLM Surprisal](https://github.com/byungdoh/llm_surprisal)

## Linear mixed-effects regression

[Modelblocks](https://github.com/modelblocks/modelblocks-release)
