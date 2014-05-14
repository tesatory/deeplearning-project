require 'nn'

imgSize = 32
inDim = 3 * imgSize^2
hidDim = 100
outDim = 10

model = nn.Sequential()
model:add(nn.Reshape(inDim))
model:add(nn.Linear(inDim, hidDim))
model:add(nn.Threshold())
model:add(nn.Linear(hidDim, outDim))
model:add(nn.LogSoftMax())
print(model)

criterion = nn.ClassNLLCriterion()
print(criterion)
