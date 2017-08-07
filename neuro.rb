require 'matrix'
require './neural_net.rb' # my implementation of neural nets

# Matrix element wise matrix multiplication 
# Hadamard product (matrices):
# https://en.wikipedia.org/wiki/Hadamard_product_(matrices)
def element_multiplication(m1, m2)
	m3 = Matrix.build(m1.row_count, m1.column_count) {|r, c| m1[r, c] * m2[r, c]}
	return m3
end

# Summation of all values in a matrix
def element_summation(m)
	s = 0
	m.each {|x| s += x}
	return s
end

# a confusion matrix illustrates the accuracy of a classifier:
# values in the diagonal of the classifier are correctly classified.
# https://en.wikipedia.org/wiki/Confusion_matrix
def confusion_matrix(expected, predicted)
	expected = expected.to_a.map {|x| x.index(x.max)}
	predicted = predicted.to_a.map {|x| x.index(x.max)}
	
	n = (expected + predicted).uniq.length
	cm = Matrix.build(n){0}.to_a
	expected.zip(predicted).map {|x, y| cm[x][y]+=1}
	
	return Matrix.rows(cm)
end

#The actual neural network
class NeuralNetwork
	def initialize()
		# For the sake of simplicity feel free to hardcode
		# parameters. The goal is a working feedforward neral
		# network. One hidden layer, one input layer, and one
		# output layer are enough to achieve 99% accuracy on
	  # the data set.
          
          # declare some inputs
          
          i0 = Input.new(-1, true) # this is a constant input that all my neurons use
          @i1 = Input.new
          @i2 = Input.new
          @i3 = Input.new
          @i4 = Input.new

          # declare some weights

          w1A = Weight.new # weight of input 1 into neuron A
          w2A = Weight.new
          w3A = Weight.new
          w4A = Weight.new
          w1B = Weight.new
          w2B = Weight.new
          w3B = Weight.new
          w4B = Weight.new
          
          wA = Weight.new  # threshold weight for neuron A
          wAC = Weight.new # weight of connection from neuron A to neuron C
          wAD = Weight.new
          wB = Weight.new
          wBC = Weight.new
          wBD = Weight.new
          
          wC = Weight.new
          wCE = Weight.new
          wCF = Weight.new
          wCG = Weight.new
          wD = Weight.new
          wDE = Weight.new
          wDF = Weight.new
          wDG = Weight.new

          wE = Weight.new
          wF = Weight.new
          wG = Weight.new

          # declare some neurons
          # first array are inputs to the neuron. second array are the corresponding weights.

          a = Neuron.new([i0,@i1,@i2,@i3,@i4], [wA,w1A,w2A,w3A,w4A])
          b = Neuron.new([i0,@i1,@i2,@i3,@i4], [wB,w1B,w2B,w3B,w4B])
          c = Neuron.new([i0, a, b], [wC, wAC, wBC])
          d = Neuron.new([i0, a, b], [wD, wAD, wBD])
          e = Neuron.new([i0, c, d], [wE, wCE, wDE])
          f = Neuron.new([i0, c, d], [wF, wCF, wDF])
          g = Neuron.new([i0, c, d], [wG, wCG, wDG])

          # connect terminal neurons to performance testers

          pe = PerformanceTester.new(e)
          pf = PerformanceTester.new(f)
          pg = PerformanceTester.new(g)

          # finally declare the network

          @net = Network.new([pe, pf, pg], [a, b, c, d, e, f, g])                    
	end

	##############################################
	def train(x, y)
          x_data = []
          y_data = []
          for i in (0...x.row_count) do
            x_data.push(x.row(i).to_a)
            y_data.push(y.row(i).to_a)
          end
          
          @net.train(x_data, y_data, 1000)
	end

	##############################################
	def predict(x)
          x_data = []
          for i in (0...x.row_count) do
            x_data.push(x.row(i).to_a)
          end
          y_data = []
          x_data.each {|datum|
            @i1.set_value(datum[0])
            @i2.set_value(datum[1])
            @i3.set_value(datum[2])
            @i4.set_value(datum[3])
            y_data.push(@net.output)
          }
          return y_data
	end

	##############################################
	protected

		##############################################
		def propagate(x)
			# applies the input to the network
			# this is the forward propagation step
		end

		##############################################
		def back_propagate(x, y, y_hat)
			# goes backwards and finds the weights
			# that need to be tuned
		end
end
