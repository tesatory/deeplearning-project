require 'optim'

parameters, gradParameters = model:getParameters()

optimState = {
	learningRate = 1,
	weightDecay = 0,
	momentum = 0.9,
	learningRateDecay = 1e-7
}

batchSize = 100

-- classes
classes = {'1','2','3','4','5','6','7','8','9','0'}
confusion = optim.ConfusionMatrix(classes)

function train(trainData)
	epoch = epoch or 1
	for t = 1,trainData.size,batchSize do
		xlua.progress(t, trainData.size)
		local inputs = {}
		local targets = {}
		for i = t,t+99 do
			local x = trainData.data[i]
			local y = trainData.label[i] + 1
			table.insert(inputs, x)
			table.insert(targets, y)
		end

		local feval = function(x)
			if x ~= parameters then
				parameters:copy(x)
			end
			gradParameters:zero()
			local f = 0
			for i =1,#inputs do
				local output = model:forward(inputs[i])
				local err = criterion:forward(output, targets[i])
				f = f + err
				local df_do = criterion:backward(output, targets[i])
				model:backward(inputs[i], df_do)
				confusion:add(output, targets[i])
			end
			gradParameters:div(#inputs)
			f = f/#inputs
			return f, gradParameters
		end

		optim.sgd(feval, parameters, optimState)
	end
	confusion:updateValids()
	print('epoch=' .. epoch .. ' train-error=' .. (1-confusion.totalValid))
	confusion:zero()
	epoch = epoch + 1
end
