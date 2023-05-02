# LayerCake: Efficient Inference Serving with Cloud and Mobile Resources

This repository contains code for the paper LayerCake: Efficient Inference Serving with Cloud and Mobile Resources, presented at [CCGRID'23](https://ccgrid2023.iisc.ac.in/)

## Overview

Many mobile applications are now integrating deep learning models into their core functionality. 
These functionalities have diverse latency requirements while demanding high-accuracy results. 
Currently, mobile applications statically decide to use either in-cloud inference, relying on a fast and consistent network, or on-device execution, relying on sufficient local resources. 
However, neither mobile networks nor computation resources deliver consistent performance in practice. 
Consequently, mobile inference often experiences variable performance or struggles to meet performance goals, when inference execution decisions are not made dynamically.

In this paper, we introduce LayerCake, a deep-learning inference framework that dynamically selects the best model and location for executing inferences. 
LayerCake accomplishes this by tracking model state and availability, both locally and remotely, as well as the network bandwidth, allowing for accurate estimations of model response time.
By doing so, LayerCake achieves latency targets in up to 96.4% of cases, which is an improvement of 16.7% over similar systems, while decreasing the cost of cloud-based resources by over 68.33% than in-cloud inference

## Structure

- `src-client`
  - Code for Android Client
- `src-server`
  - Code for information server

## Citation

```
@INPROCEEDINGS{Ogden2023a,
  author={Ogden, Samuel S. and Guo, Tian},
  booktitle={2023 23rd IEEE International Symposium on Cluster, Cloud and Internet Computing (CCGrid)}, 
  title={LayerCake: Efficient Inference Serving with Cloud and Mobile Resources}, 
  year={2023}
}
```

## Acknowledgements

We thank the anonymous reviewers for their constructive reviews. 
This work is partly supported by NSF Grants 1755659, 1815619, 2105564, 2236987, and VMWare.