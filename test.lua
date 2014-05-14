function test(testData)
	for t = 1,testData.size do
		-- xlua.progress(t, testData.size)
		local input = testData.data[t]
		local target = testData.labels[t]
		local pred = model:forward(input)
		confusion:add(pred, target)
	end
	confusion:updateValids()
	print('testing test-error=' .. (1-confusion.totalValid))
	confusion:zero()
end