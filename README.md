<div align="center">

# Zero or Infinite Data? Knowledge Synchronized Machine Learning Emulation

</div>

## Abstract

Even when the mathematical model is known in many applications in computational science and engineering, uncertainties are unavoidable. They are caused by initial conditions, boundary conditions, and so on. As a result, repeated evaluations of a costly model governed by partial differential equations (PDEs) are required, making the computation prohibitively expensive. Recently, neural networks have been used as fast alternatives for propagating and quantifying uncertainties. Notably, a large amount of high-quality training data is required to train a reliable neural networks-based emulator. Such ground truth data is frequently gathered in advance by running the numerical solvers that these neural emulators are intended to replace. But, if the underlying PDEs’ form is available, do we really need training data? In this paper, we present a principled training framework derived from rigorous and trustworthy scientific simulation schemes. Unlike traditional neural emulator approaches, the proposed emulator does not necessitate the use of a classical numerical solver to collect training data. Rather than emulating dynamics directly, it emulates how a specific numerical solver solves PDEs. The numerical case study demonstrates that the proposed emulator performed well in a variety of testing scenarios.

For more information, please refer to the following:

- Xihaier Luo and Wei Xu and Yihui Ren and Shinjae Yoo and Balu Nadiga and Ahsan Kareem. "[Zero or Infinite Data? Knowledge Synchronized Machine Learning Emulation](https://openreview.net/forum?id=-VxUZp0Zkdg)." (NeurIPS 2022: Workshop on AI for Science Progress and Promises).
