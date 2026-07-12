This code is to inspect various learning rate schedulers that try to adapt learning rates to individual minibatches. 
A well known theoretical technique is the Polyak Stepsize, for which Polyak proved convergence speeds in his book from 1987 (but the idea was already in his paper from 1969). 

In practice, this stepsize is unstable, and people proposed many fixes in the hope of making it practical. 
Here is a software-framework to conveniently check some of these approaches. Also, a new optimisation method net-line is included.

The file 'trainer.py' contains the Trainer class. 
It runs a standard pytorch training pipeline and allows to add a scheduler. 
The scheduler is any object that has function 'next', function 'next_batch' or both. 
The 'next' function is called at the end of each epoch, the next_batch is called right before each gradient step and it updates the learning rate of the optimizer. 

In tools/recorder.py there is a Recorder object that monitors and logs the training process. It has an elaborate 'recplot' function. 

In tools/load_data.py there is a bit of messy code to load various datasets. 
For debugging, there is an artificial fix to make the datasets smaller. 

To see a demo of the code, run:

python experiments.py 2
