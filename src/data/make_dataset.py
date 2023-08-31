


"""
The cleanest way to do it is to have a build_dataset.py file that will be called once at the start of the project 
and will create the split into train, dev and test. Optionally, calling build_dataset.py can also download the dataset.
We need to make sure that any randomness involved in build_dataset.py uses a fixed seed so that every call to python build_dataset.py will result in the same output. 

filenames = ['img_000.jpg', 'img_001.jpg', ...]
filenames.sort()  # make sure that the filenames have a fixed order before shuffling
random.seed(230)
random.shuffle(filenames) # shuffles the ordering of filenames (deterministic given the chosen seed)

split_1 = int(0.8 * len(filenames))
split_2 = int(0.9 * len(filenames))
train_filenames = filenames[:split_1]
dev_filenames = filenames[split_1:split_2]
test_filenames = filenames[split_2:]
The call to filenames.sort() makes sure that if you build filenames in a different way, the output is still the same.


"""