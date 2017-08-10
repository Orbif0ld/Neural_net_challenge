require 'set'
require 'byebug'

=begin

This file implemets a neural net.

In this implementation, neural nets are composed of

- Inputs
- Neurons
- Performance testers attached to outputs

Neurons in this code use 1.0/(1.0 + e**(-z)) as a threshold function, where z
is weights*inputs to the neuron. Each neuron has one special input with value -1.
The corresponding weight shifts the threshold function.

The network also relies on the Weight class. Each instance of that class
is a weight in the net that can be set or differentiated with respect to.

Performance testers are attached to all outputs of the network.
They provide a function that has a maximum when the net returns
the desired values. Weight values are improved by hill-climbing
on this function. (Basically calculating the gradient and following it.)

The Network class wraps up the neural network and provides
the output and train methods.


The motiviation behind this implementation was to provide a "language"
for building neural nets. I image that an implmenetation with loops instead
of recursions and arrays instead of objects may be faster.

This implementation could use a couple more upgrades:
- provide utility methods that reduce the amount of typing necessary
  to construct a net
- at initialization, determine which neurons depend (indirectly) on which
  weights. this could be used to speed up differentiation.
- add methods for easier testing of the network

=end

module Setable

  # sets a new @value and returns self
  def set_value(val)
    raise ArgumentError, 'Argument is not numeric'  unless val.is_a?(Numeric) 
    @value = val
    return self
  end

  # returns @value
  def get_value()
    @value
  end
end

# elements of neural nets

# represents a weight in the net
# it's important that each weight is represented by it's own instance
# of the weight class even if the value is the same
class Weight
  include Setable

  def initialize(val=0)
    @value = val
    @next_value = val
  end

  # sets a new value for this weight, which becomes
  # the new active weight value when Weight#update is
  # called
  def set_next_value(val)
    @next_value = val
  end

  # makes the previously set next value
  # the active value for this weight
  def update
    @value = @next_value
  end
end

# represents and input to the net
class Input
  include Setable

  attr_reader :constant

  # creates a new input
  # val must be a number
  def initialize(val=-1, constant=false)
    raise RuntimeError unless val.is_a?(Numeric)
    @value = val
    @constant = constant
  end

  # computes the output of this element
  # returns a number
  def output
    return @value
  end

  # computes the derivative of the this element
  # with respect to weight
  # weight must be an instance of Weight
  # returns a number
  def derivative(weight)
    return 0
  end
end

# represents a neuron in the net
# another way to think of it is as
# a sub-network, starting at this neuron
# and going back to the inputs
class Neuron

  attr_reader :weights, :inputs

  # creates a new neuron
  # input_list is a list of neural net elements
  # weight_list is a list of weights
  # input_list[i] belongs with weight_list[i]
  def initialize(input_list, weight_list)
    @inputs = input_list
    @weights = weight_list
    @output_computed = false
    @output = nil
    @derivatives = Hash.new
    @connected_weights = find_connected_weights
  end

  # computes the output of this neuron
  # returns a number
  def output
    if @output_computed
      return @output
    else
      z = @inputs.zip(@weights).map{|input, weight|
        weight.get_value * input.output}.reduce(:+)
      @output_computed = true
      @output = sigmoid(z)
    end
  end

  # compute the derivate of the sub-network with
  # respect to weight
  # weight must be an instance of the Weight class
  # returns a number
  def compute_derivative(weight)
    if not @connected_weights.include? weight
      return 0
    end
    z = @inputs.zip(@weights).map {|inpt, wt| inpt.output * wt.get_value}.reduce(:+)
    if @weights.include?(weight)
      i = @weights.find_index(weight)
      return sigmoid(z) * (1 - sigmoid(z)) * @inputs[i].output
    else
      return sigmoid(z) * (1 - sigmoid(z)) *
             @inputs.zip(@weights).map {|inpt, wt| wt.get_value * inpt.derivative(weight)}.
               reduce(:+)
    end
  end

  # generates a set of weights which this neuron depends on indirectly
  # returns a set of weights
  def find_connected_weights
    weights_set = Set.new
    @weights.each {|w| weights_set.add(w)}
    @inputs.each {|i| weights_set.merge(i.find_connected_weights) unless i.instance_of? Input}
    return weights_set
  end

  # returns true if this neuron depends on weight, returns false otherwise
  # paramater weight must be an instance of the Weight class
  def depends_on(weight)
    return @connected_weights.include? weight
  end

  # looks up derivative if a value is cached,
  # otherwise computes the derivative and caches the value
  # (I decided to serparate the caching from the computation here)
  def derivative(weight)
    if @derivatives.has_key?(weight)
      return @derivatives[weight]
    else
      @derivatives[weight] = compute_derivative(weight)
      return @derivatives[weight]
    end
  end

  # resets this neuron's cache
  def reset_cache
    @output_computed = false
    @derivatives = Hash.new
  end

  private
  
  # threshold function
  def sigmoid(z)
    1/(1 + Math::exp(-z))
  end
end

# connected to the output of a final neuron
# calculates the performance function from
# the neuron output
class PerformanceTester

  attr_reader :neuron

  # end_neuron should be an instance of the class Neuron
  def initialize(end_neuron)
    @neuron = end_neuron
    @desired_output = 0
  end

  # sets the value that the network is supposed to return
  # val should be a number
  def set_desired_output(val)
    @desired_output = val
  end

  def output
    -0.5*(@neuron.output - @desired_output)**2
  end

  def derivative(weight)
    (@desired_output - @neuron.output) * @neuron.derivative(weight)
  end
end

# represents a neural network
class Network

  # perfmance_testers is an array of PerformanceTester instances
  # each is connected to one of the terminal (output) neurons of the net 
  def initialize(performance_testers, neurons)
    @performance_testers = performance_testers
    @neurons = neurons
    @weights = @neurons.map {|n| n.weights}.flatten
    inputs_set = Set.new
    @neurons.
      map {|n| n.inputs}.
      flatten.
      select{|i| (i.instance_of? Input)}.
      each {|i| inputs_set.add(i)}
    @inputs = inputs_set.to_a
  end

  # returns the output of the network
  # if the network has one output only
  # returns a number
  # if the network has several outputs,
  # returns an array of numbers
  def output
    reset_cache
    result = @performance_testers.map {|p| p.neuron.output}
    if result.size==1
      return result.first
    else
      return result
    end
  end

  def performance_output
    reset_cache
    @performance_testers.map {|p| p.output}.reduce(:+)
  end

  # computes the derivative with respect to weight
  # of the sum of performance tester outputs
  # returns a number
  def performance_derivative(weight)
    @performance_testers.map {|p| p.derivative(weight)}.reduce(:+)
  end

  # trains the network to produce the desired outputs
  # from the inputs
  # each element of inputs should be an array of input values (numbers)
  # each element of desired_outputs should be an array of desired
  # output values (numbers)
  # inputs[i] must correspond to desired_outputs[i]
  def train(inputs, desired_outputs,
            max_iterations=500,
            step_size=1.0)
    
    data = inputs.zip(desired_outputs)
    
    iterations = 0
    performances = []
    
    while iterations < max_iterations
      data.each {|inpts, outpts|
        # set up the inputs to the network
        #exclude inputs that are marked constant
        @inputs.select{|i| !i.constant}.zip(inpts).each {|i, val| i.set_value(val)}
        # set up the performance testers with expected values
        @performance_testers.zip(outpts).each {|p, d| p.set_desired_output(d)}
        reset_cache
        # calculate improved weights
        @weights.each {|w| w.set_next_value(
          w.get_value + step_size * performance_derivative(w))}
        # use the new weights
        update
        reset_cache
      }
      # if iterations % max_iterations/10 == 0
      #      puts "#{iterations}"
      #end
      if iterations % max_iterations/10 == 0
        performances.push(performance_output) ## TODO: bad, just samples, fix later
      end
      iterations += 1
    end
    return performances
  end

  private

  # sets the weights in the network to their next value
  # use when weights have changed
  def update
    @weights.each {|w| w.update}
  end

  # resets the output caches of all neurons in the network
  def reset_cache
    @neurons.each {|n| n.reset_cache}
  end
end
