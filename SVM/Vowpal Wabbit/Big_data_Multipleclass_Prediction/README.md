## VowpalWabbit
Vowpal Wabbit is a machine learning system which pushes the frontier of machine learning with techniques such as online, hashing, allreduce, reductions, learning2search, active, and interactive learning.

There are several features that (in combination) can be powerful.

1.Input Format:
 The input format for the learning algorithm is substantially more flexible than might be expected. Examples can have features consisting of free form text, which is interpreted in a bag-of-words way. There can even be multiple sets of free form text in different namespaces.
2.Speed:
 The learning algorithm is pretty fast---similar to the few other online algorithm implementations out there. As one datapoint, it can be effectively applied on learning problems with a sparse terafeature (i.e. 1012 sparse features). As another example, it's about a factor of 3 faster than Leon Bottou's svmsgd on the RCV1 example in wall clock execution time.
3.Scalability:
 This is not the same as fast. Instead, the important characteristic here is that the memory footprint of the program is bounded independent of data. This means the training set is not loaded into main memory before learning starts. In addition, the size of the set of features is bounded independent of the amount of training data using the hashing trick.
4.Feature Pairing:
 Subsets of features can be internally paired so that the algorithm is linear in the cross-product of the subsets. This is useful for ranking problems. David Grangier seems to have a similar trick in the PAMIR code. The alternative of explicitly expanding the features before feeding them into the learning algorithm can be both computation and space intensive, depending on how it's handled.

#### prerequisites
1.VowpalWabbit, ref :https://github.com/VowpalWabbit/ for installation and running.
2.numpy
3.pandas

#### Algorithm
1. Convert train and test data into vowpal wabbit data format. 
2. using VW commandline arguments train your data and save the model.
3. using the trained model predict the classes of your test data.
4. check the accuracy.

#### command line arguments
ref : https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Command-Line-Arguments
#### loss functions 
ref : https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Loss-functions

