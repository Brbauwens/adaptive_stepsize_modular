This code is to inspect various learning rate schedulers that try to adapt learning rates to individual minibatches. 
A well known theoretical technique is the Polyak Stepsize, for which Polyak proved convergence speeds in his book from 1987 (but the idea is already in a paper of him from 1969). 

In practice, this stepsize is unstable, and people proposed many fixes and hope to make it practical. 
Here is some software to conveniently check these approaches. I haven't checked to much yet, I hope to add more soon. 


The file 'trainer.py' contains the Trainer class. 
It runs a standard pytorch training pipeline and allows to add a scheduler. 
Here a scheduler is any object that either has a 'next' function or 'next_batch' function or both. 
The 'next' function is called at the end of each epoch, the next_batch is called right before each gradient step and it updates the learning rate of the optimizer. 

In tools/recorder.py there is a Recorder object that carefully monitors the training process. It has an elaborate 'recplot' function. 

In tools/load_data.py there is a bit of messy code to load various datasets. 
For debugging, there is a way to modify to artificially make the datasets smaller. 

To see a demo of the code, run for example:

python experiments.py 2
