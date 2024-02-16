# Understanding the Weakness of Large Language Model Agents within a Complex Android Environment

<a href="https://arxiv.org/abs/2402.06596">Paper Link</a>

## Abstract
Large language models (LLMs) have empowered intelligent agents to execute intricate tasks within `domain-specific software` such as browsers and games. However, when applied to `general-purpose software systems` like operating systems, LLM agents face three primary challenges. Firstly, the `action space is vast and dynamic`, posing difficulties for LLM agents to maintain an up-to-date understanding and deliver accurate responses. Secondly, real-world tasks often require `inter-application cooperation`, demanding farsighted planning from LLM agents. Thirdly, agents need to identify optimal solutions `aligning with user constraints`, such as security concerns and preferences.
These challenges motivate AndroidArena, an environment and benchmark designed to evaluate LLM agents on a modern operating system. To address high-cost of manpower, we design a scalable and semi-automated method to construct the benchmark.
In the task evaluation, AndroidArena incorporates accurate and adaptive metrics to address the issue of non-unique solutions. Our findings reveal that even state-of-the-art LLM agents struggle in cross-APP scenarios and adhering to specific constraints. Additionally, we identify a lack of four key capabilities, i.e., understanding, reasoning, exploration, and reflection, as primary reasons for the failure of LLM agents. Furthermore, we provide empirical analysis on the failure of reflection, and improve the success rate by 27% with our proposed exploration strategy. This work is the first to present valuable insights in understanding fine-grained weakness of LLM agents, and offers a path forward for future research in this area.

## Demo
Task: `Get directions from my current location to "Microsoft SVC Building".`


https://github.com/AndroidArenaAgent/AndroidArena/assets/158838805/e7395b3b-4272-45e2-8492-93572ad722ec



## Dependencies:

### Python
- Python 3.10
- `pip install -r requirements.txt`

### Emulator Installation
Please follow [Android Emulator Installation Guide](./android_env2/README.md) to install the Android Emulator.

### Environment Steup
1. Please setup up your Google account first.
2. Run setup scripts:
    - for single-APP evaluation: `python scripts/env_setup.py`
    - for cross-APP evaluation: `python scripts/env_setup_crossapp.py`

## Benchmark
The task instructions are located in the `tasks` folder, where tasks for each APP are organized in YAML files. The `constrain.yaml` and `cross-app.yaml` files contain cross-APP and constrained tasks, respectively. We offer only task instructions at this time, with the exception of `calendar.yaml` provided as an example. Annotated action sequences will be released later.

## Run

### Execute tasks

`python run_lm_agent.py --model_provider=<model_provider> --model_name=<model_name> --agent_type=<agent_type> --test_app=<app_name>`

For example:

`python run_lm_agent.py --model_name=gpt-4 --agent_type=react --test_app=calendar`

### Evaluation

The evaluation script is in `run_evaluator.py`.



## Citation
If you find our environment or benchmark useful, please cite our paper:

```
@article{xing2024understanding,
  title={Understanding the Weakness of Large Language Model Agents within a Complex Android Environment},
  author={Xing, Mingzhe and Zhang, Rongkai and Xue, Hui and Chen, Qi and Yang, Fan and Xiao, Zhen},
  journal={arXiv preprint arXiv:2402.06596},
  year={2024}
}
```
