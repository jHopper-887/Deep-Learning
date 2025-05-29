reLU(x) = max(0, x)
sigmoid(x) = 1 / (1 + exp(-x))

mse(x, y) = (x - y)^2

output = [4.0, 5.0]

function perceptron(weights::AbstractMatrix, inputs::AbstractVector, bias::AbstractVector, activation_func::Function)
    weighted_sum = weights * inputs .+ bias
    return activation_func.(weighted_sum)
end

function cost(predicted::AbstractVector, actual::AbstractVector)
    return sum(mse.(predicted, actual)) / length(actual)
end

function train_neuron!(weights::AbstractMatrix, inputs::AbstractVector, target::AbstractVector, 
                      outputs::AbstractVector, learning_rate::Float64=0.01)
    error = outputs - target
    grad = error .* (outputs .> 0) * inputs'
    weights .-= learning_rate * grad
    return weights
end

weights = [1.4 2.2; 3.5 4.2]
inputs = [2.2, 4.3]
bias = [2.0, 3.0]

for epoch in 0:200
    result = perceptron(weights, inputs, bias, reLU)
    current_cost = cost(result, output)
    println("Epoch $epoch Cost: $current_cost Output: $result")
    global weights= train_neuron!(weights, inputs, output, result, 0.01)
end

@show perceptron(weights, inputs, bias, reLU) == output
@show output
@show weights

