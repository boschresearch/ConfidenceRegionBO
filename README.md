# Confidence Region Bayesian Optimization

This is the companion code for the benchmarking study reported in the paper
*Cautious Bayesian Optimization for Scalable and Efficient Policy Search*
by Lukas Fröhlich et al., L4DC 2021. The paper can be found 
[here](https://arxiv.org/abs/2011.09445). The code allows the users to
reproduce and extend the results reported in the study. Please cite the
above paper when reporting, reproducing or extending the results.

## Purpose of the project

This software is a research prototype, solely developed for and published as
part of the publication cited above. It will neither be maintained nor monitored
in any way.

## Installation guide

In the root directory of the repository execute the following commands:
```shell
conda env create -f environment.yml
conda activate crbo
pip install -e .
```

## Basic functionality

Note: This implementation assumes that you want to minimize a function. 

We have prepared a jupyter notebook to run CRBO on the CartPole task.
Before we start, make sure to download the pre-trained RL agents.
For your convenience, just execute the following commands:

```shell
# makes the script executable
chmod +x download_agents.sh
# downloads the files (only a few kB) and stores to disk
./download_agents.sh
```

Now, to open the notebook execute the following commands:

```shell
jupyter notebook run_experiment.ipynb
```

If the browser window does not open automatically, copy the generated link and
paste it manually into your browser.

## License

Confidence Region Bayesian Optimization is open-sourced under the AGPL-3.0
license. See the [LICENSE](LICENSE) file for details.

