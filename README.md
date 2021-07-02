# Pragmatics for SCII based on NDQ


## Note
 This is the codebase for the SCII case study of NeurIPS2020 spotlight paper [**Incorporating Pragmatic Reasoning Communication into Emergent Language**](https://fringsoo.github.io/pragmatic_in2_emergent_papersite/), authored by Yipeng Kang, [Tonghan Wang](https://tonghanwang.github.io/) and [Gerard de Melo](http://gerard.demelo.org/). It is based on [NDQ](https://github.com/TonghanWang/NDQ),  [PyMARL](https://github.com/oxwhirl/pymarl) and [SMAC](https://github.com/oxwhirl/smac), which are open-sourced.

Tested Environment: AWS g4dn.xlarge GPU (NVIDIA T4 Tensor Core GPU) instance Ubuntu 16.04 (with CUDA cuDNN, NVIDIA-Docker) 

To install necessary dependencies, build the Dockerfile using 
```shell
cd docker
bash build.sh
```

Set up StarCraft II and SMAC:
```shell
bash install_sc2.sh
```
This will download SC2 into the 3rdparty folder and copy the required maps.

Config in sc2.yaml to use the default map.
```shell
map_name: "1o_2r_vs_4r" 
```


For long-term training:

Config in defaulty.yaml
```shell
evaluate: "False" 
```
Config in categorical_qmix.yaml, to use the pretrained model. 
Or comment out this item to train from scratch.
```shell
checkpoint_path: "results/models/cate_qmix_smac_parallel__2020-05-26_16-14-23__1o_2r_vs_4r/" 
```

For short-term basseline and pragmatics testing:
Config in defaulty.yaml
```shell
evaluate: "True" 
```
Config "checkpoint_path" to your model path.

Run:
```shell
bash run_interactive.sh $GPU python3 src/main.py --config=categorical_qmix --env-config=sc2
```

The testing result metrics (took about 10-20 minutes to run):
| drop rate  | baseline |pragmatics| 
| ---------- | -------- |----------|
| 0%  |  80.3% | 79.8% |
| 30% |  77.8% | 79.7% |
| 60% |  67.5% | 70.9% |
| 80% |  56.3% | 63.1% |
| 90% |  49.1% | 56.6% |
| 95% |  48.4% | 53.1% |
| 98% |  47.2% | 50.0% |
| 100%|  40.6% | 50.6% |

<!-- 
# NDQ: Learning Nearly Decomposable Value Functions with Communication Minimization

## Note
 This codebase accompanies paper [Learning Nearly Decomposable Value Functions with Communication Minimization](https://openreview.net/forum?id=HJx-3grYDB&noteId=HJx-3grYDB), 
 and is based on  [PyMARL](https://github.com/oxwhirl/pymarl) and [SMAC](https://github.com/oxwhirl/smac) codebases which are open-sourced.

The implementation of the following methods can also be found in this codebase, which are finished by the authors of [PyMARL](https://github.com/oxwhirl/pymarl):

- [**QMIX**: QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1803.11485)
- [**COMA**: Counterfactual Multi-Agent Policy Gradients](https://arxiv.org/abs/1705.08926)
- [**VDN**: Value-Decomposition Networks For Cooperative Multi-Agent Learning](https://arxiv.org/abs/1706.05296) 
- [**IQL**: Independent Q-Learning](https://arxiv.org/abs/1511.08779)

Build the Dockerfile using 
```shell
cd docker
bash build.sh
```

Set up StarCraft II and SMAC:
```shell
bash install_sc2.sh
```

This will download SC2 into the 3rdparty folder and copy the maps necessary to run over.

The requirements.txt file can be used to install the necessary packages into a virtual environment (not recomended).

## Run an experiment 

The following command train NDQ on the didactic task `hallway`.

```shell
python3 src/main.py 
--config=categorical_qmix
--env-config=join1
with
env_args.n_agents=2
env_args.state_numbers=[6,6]
obs_last_action=False
comm_embed_dim=3
c_beta=0.1
comm_beta=1e-2
comm_entropy_beta=0.
batch_size_run=16
t_max=2e7
local_results_path=$DATA_PATH
is_cur_mu=True
is_rank_cut_mu=True
runner="parallel_x"
test_interval=100000
```

The config files act as defaults for an algorithm or environment. 

They are all located in `src/config`.
`--config` refers to the config files in `src/config/algs`
`--env-config` refers to the config files in `src/config/envs`

To run experiments using the Docker container:
```shell
bash run.sh $GPU python3 src/main.py --config=qmix_smac --env-config=sc2 with env_args.map_name=2s3z
```

All results will be stored in the `Results` folder.

## Saving and loading learnt models

### Saving models

You can save the learnt models to disk by setting `save_model = True`, which is set to `False` by default. The frequency of saving models can be adjusted using `save_model_interval` configuration. Models will be saved in the result directory, under the folder called *models*. The directory corresponding each run will contain models saved throughout the experiment, each within a folder corresponding to the number of timesteps passed since starting the learning process.

### Loading models

Learnt models can be loaded using the `checkpoint_path` parameter, after which the learning will proceed from the corresponding timestep. 

## Watching StarCraft II replays

`save_replay` option allows saving replays of models which are loaded using `checkpoint_path`. Once the model is successfully loaded, `test_nepisode` number of episodes are run on the test mode and a .SC2Replay file is saved in the Replay directory of StarCraft II. Please make sure to use the episode runner if you wish to save a replay, i.e., `runner=episode`. The name of the saved replay file starts with the given `env_args.save_replay_prefix` (map_name if empty), followed by the current timestamp. 

The saved replays can be watched by double-clicking on them or using the following command:

```shell
python -m pysc2.bin.play --norender --rgb_minimap_size 0 --replay NAME.SC2Replay
```

**Note:** Replays cannot be watched using the Linux version of StarCraft II. Please use either the Mac or Windows version of the StarCraft II client. -->
