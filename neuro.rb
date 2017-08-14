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

          print "initializing network... "
          
          i0 = Input.new(-1, true) # this is a constant input that all my neurons use
          @i1 = Input.new
          @i2 = Input.new
          @i3 = Input.new
          @i4 = Input.new

          g = Random.new

          # declare some neurons
          # first array are inputs to the neuron. second array are the corresponding weights.

          a = Neuron.new([i0,@i1,@i2,@i3,@i4], (1..5).map {Weight.new(g.rand(-1..1))})
          b = Neuron.new([i0,@i1,@i2,@i3,@i4], (1..5).map {Weight.new(g.rand(-1..1))})
          k = Neuron.new([i0,@i1,@i2,@i3,@i4], (1..5).map {Weight.new(g.rand(-1..1))})
          l = Neuron.new([i0,@i1,@i2,@i3,@i4], (1..5).map {Weight.new(g.rand(-1..1))})
          n = Neuron.new([i0,@i1,@i2,@i3,@i4], (1..5).map {Weight.new(g.rand(-1..1))})
          m = Neuron.new([i0, a, b, k, l, n], (1..6).map {Weight.new(g.rand(-1..1))})
          o = Neuron.new([i0, a, b, k, l, n], (1..6).map {Weight.new(g.rand(-1..1))})
          c = Neuron.new([i0, a, b, k, l, n], (1..6).map {Weight.new(g.rand(-1..1))})
          d = Neuron.new([i0, a, b, k, l, n], (1..6).map {Weight.new(g.rand(-1..1))})
          h = Neuron.new([i0, a, b, k, l, n], (1..6).map {Weight.new(g.rand(-1..1))})
          e = Neuron.new([i0, c, d, h, m, o], (1..6).map {Weight.new(g.rand(-1..1))})
          f = Neuron.new([i0, c, d, h, m, o], (1..6).map {Weight.new(g.rand(-1..1))})
          g = Neuron.new([i0, c, d, h, m, o], (1..6).map {Weight.new(g.rand(-1..1))})

          # connect terminal neurons to performance testers

          pe = PerformanceTester.new(e)
          pf = PerformanceTester.new(f)
          pg = PerformanceTester.new(g)

          # finally, declare the network

          @net = Network.new([pe, pf, pg],
                             [a, b, c, d, e, f, g, h, k, l, m, n, o])
          puts "done"
	end

	##############################################
	def train(x, y)
          puts "training... "
          x_data = []
          y_data = []
          for i in (0...x.row_count) do
            x_data.push(x.row(i).to_a)
            y_data.push(y.row(i).to_a)
          end
          
          @net.train(x_data, y_data, 450, verbose=true, step_size=0.2)
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
