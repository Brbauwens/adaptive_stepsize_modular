The trainer.py contains a Trainer object. 
The idea is that it is easy to implement various schedulers, which are objects which have 'next' function or 'next_batch' function
(the first function is called at the end of each epoch, the other right before each gradient step). 

In tools/recorder.py there is a Recorder object that carefully monitors the training process and has an elaborate 'recplot' function. 
There is a commented code to store some logfiles. 

In tools/load_data.py there is a bit of messy code to load various datasets. 
For debugging, there is a way to modify to artificially make the datasets smaller. 

To see a demo of the code, run 

python experiments.py 2

There are a few more experiments. Unfortunately, I did not implement netline and lookahead yet. 
