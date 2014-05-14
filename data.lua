
trainPath = '/Users/sainaa/data/cifar-10/CIFAR_CN_train.t7'
testPath = '/Users/sainaa/data/cifar-10/CIFAR_CN_train.t7'

trainFile = torch.load(trainPath)
testFile = torch.load(testPath)

trainData = {
	data = trainFile.datacn,
	labels = trainFile.labels,
	size = trainFile.datacn:size(1)
	}
testData = {
	data = testFile.datacn,
	labels = testFile.labels,
	size = testFile.datacn:size(1)
	}
