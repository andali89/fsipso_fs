# FSIPSO Implementation

This is an implementation of a forward search inspired PSO algorihtm (FSIPSO) 
for feature selection (FS) for high-dimensional data. FSIPSO uses a forward search 
scheme to dynamically expand the search space of FS problems. The mutation operation
is also used in FSIPSO to improve the search performance. For a detailed description 
of the method please refer to 


> Li, A.-D., Xue, B., & Zhang, M. (2021). A Forward Search Inspired Particle Swarm Optimization Algorithm for Feature Selection in Classification. IEEE Congress on Evolutionary Computation, CEC 2021, Kraków, Poland, June 28 - July 1, 2021, 786–793. 
[[BibTeX](https://andali89.github.io/homepage/bibfiles/Li2021FPSO.bib)] [https://doi.org/10.1109/CEC45853.2021.9504949](https://doi.org/10.1109/CEC45853.2021.9504949)

The source code is in the [src](./src/) folder. An illustration to run the FSIPSO algorithm is implemented in [./src/fs/RunFSIPSO.java](./src/fs/RunFSIPSO.java). Please note that code requires the jar file ([./lib/weka.jar](./lib/weka.jar)) of [Weka](https://www.cs.waikato.ac.nz/ml/weka/), Machine Learning Software in Java. Please make sure it is added in the libiray.  