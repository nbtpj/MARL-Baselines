# MARL Algorithms

One-file implementation (inspired by cleanRL) of different marl algorithms. 

## Run each

- Check the availability of the environment in the [`./register_env.py`](./register_env.py) (It provide a mapping from environment id (str) to initialize function (callable))
- Check the availability of the environment in the [`./hparams.py`](./hparams.py) (It provide a mapping from (environment id, algorithm name) (tuple of (str, \[file name without .py\])) to hyperparameter dict)

- The training loop is configed and run at each file. For example to run qmix on environment name `smac:mm`
```bash
python qmix.py --env smac:mm
```
## Scale

the implementation is designed for scale (via `ray`). Refer to the example at [`./ray_at_scale.py`](./ray_at_scale.py) for hparams tuning, concurrent experiment running or accelerating on large-resource machine.