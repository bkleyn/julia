using Flux.Tracker

cd(@__DIR__)

isfile("auto-mpg.data") ||
  download("https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data",
           "auto-mpg.data")

rawdata = readdlm("auto-mpg.data", '\t')

# The last feature is our target -- the price of the house.

x = rawdata[1:13,:]
y = rawdata[14:14,:]

# Normalise the data
x = (x .- mean(x,2)) ./ std(x,2)

# The model

W = param(randn(1,13)/10)
b = param([0.])

predict(x) = W*x .+ b
meansquarederror(ŷ, y) = sum((ŷ .- y).^2)/size(y, 2)
loss(x, y) = meansquarederror(predict(x), y)

function update!(ps, η = .1)
  for w in ps
    w.data .-= w.grad .* η
    w.grad .= 0
  end
end

for i = 1:10
  back!(loss(x, y))
  update!((W, b))
  @show loss(x, y)
end

predict(x[:,1]) / y[1]
