# recency-bias
Code and instructions for replicating experiments from ["Linear Recency Bias During Training Improves Transformers’ Fit to Reading Times" (Clark et al., COLING 2025)](https://aclanthology.org/2025.coling-main.517/).

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

6. Convert the model to Hugging Face format.
The `tools` directory has separate scripts to run depending on whether the LM uses no recency bias (i.e., vanilla Transformer), de Varda and Marelli (2024) bias, or ALiBi.
Example commands:
```
 python tools/convert_sequential_to_hf.py --input_dir <ORIGINAL-LM-PATH>/global_step10000 --config_file configs/pythia-2-4-256-1k.yml --output_dir <HF-LM-PATH>/global_step10000
 ```
```
 python tools/convert_sequential_to_hf_alibi.py --input_dir <ORIGINAL-LM-PATH>/global_step10000 --config_file configs/pythia-2-4-256-1k-alibi.yml --output_dir <HF-LM-PATH>/global_step10000
 ```
```
 python tools/convert_sequential_to_hf_devarda.py --input_dir <ORIGINAL-LM-PATH>/global_step10000 --config_file configs/pythia-2-4-256-1k-dVM.yml --output_dir <HF-LM-PATH>/global_step10000
 ```
 Substitute your own model paths into `<ORIGINAL-LM-PATH>` and `<HF-LM-PATH>`.

## Surprisal estimation on psycholinguistic corpora

1. Install an editable version of [Transformers](https://github.com/huggingface/transformers), and check out the relevant commit:
```
git clone https://github.com/huggingface/transformers
cd transformers
git checkout d0acc953
pip install -e .
```

2. Copy the files under the `transformers/` directory within `recency-bias` into the relevant locations in the Transformers repo. This will enable use of ALiBi or de Varda and Marelli (2024) bias in Hugging Face–style models.

4. Clone the [llm_surprisal](https://github.com/byungdoh/llm_surprisal) repository, and copy the trained LM (in Hugging Face format) into a subdirectory.

5. Run `get_llm_surprisal.py` on psycholinguistic stimuli:
```
python get_llm_surprisal.py <STIMULI> <HF-LM-PATH> word > <STIMULI>.surprisals
```
Substitute `<STIMULI>` with your own file, formatted with one sentence per line.

## Linear mixed-effects regression

Various tools for mixed-effects regression are available in the [Modelblocks](https://github.com/modelblocks/modelblocks-release) repository, with documentation [here](https://www.asc.ohio-state.edu/schuler.77/overview-mb.pdf).
