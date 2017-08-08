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

          g = Random.new

          w1A = Weight.new(g.rand(-1..1)) # weight of input 1 into neuron A
          w2A = Weight.new(g.rand(-1..1))
          w3A = Weight.new(g.rand(-1..1))
          w4A = Weight.new(g.rand(-1..1))
          w1B = Weight.new(g.rand(-1..1))
          w2B = Weight.new(g.rand(-1..1))
          w3B = Weight.new(g.rand(-1..1))
          w4B = Weight.new(g.rand(-1..1))
          
          wA = Weight.new(g.rand(-1..1))  # threshold weight for neuron A
          wAC = Weight.new(g.rand(-1..1)) # weight of connection from neuron A to neuron C
          wAD = Weight.new(g.rand(-1..1))
          wB = Weight.new(g.rand(-1..1))
          wBC = Weight.new(g.rand(-1..1))
          wBD = Weight.new(g.rand(-1..1))
          
          wC = Weight.new(g.rand(-1..1))
          wCE = Weight.new(g.rand(-1..1))
          wCF = Weight.new(g.rand(-1..1))
          wCG = Weight.new(g.rand(-1..1))
          wD = Weight.new(g.rand(-1..1))
          wDE = Weight.new(g.rand(-1..1))
          wDF = Weight.new(g.rand(-1..1))
          wDG = Weight.new(g.rand(-1..1))

          wE = Weight.new(g.rand(-1..1))
          wF = Weight.new(g.rand(-1..1))
          wG = Weight.new(g.rand(-1..1))
          
          wH = Weight.new(g.rand(-1..1))
          wAH = Weight.new(g.rand(-1..1))
          wBH = Weight.new(g.rand(-1..1))
          wHE = Weight.new(g.rand(-1..1))
          wHF = Weight.new(g.rand(-1..1))
          wHG = Weight.new(g.rand(-1..1))

          wK = Weight.new(g.rand(-1..1))
          w1K = Weight.new(g.rand(-1..1))
          w2K = Weight.new(g.rand(-1..1))
          w3K = Weight.new(g.rand(-1..1))
          w4K = Weight.new(g.rand(-1..1))
          wKD = Weight.new(g.rand(-1..1))
          wKH = Weight.new(g.rand(-1..1))
          wKC = Weight.new(g.rand(-1..1))

          wL = Weight.new(g.rand(-1..1))
          wLC = Weight.new(g.rand(-1..1))
          wLD = Weight.new(g.rand(-1..1))
          wLH = Weight.new(g.rand(-1..1))
          w1L = Weight.new(g.rand(-1..1))
          w2L = Weight.new(g.rand(-1..1))
          w3L = Weight.new(g.rand(-1..1))
          w4L = Weight.new(g.rand(-1..1))

          # wN = Weight.new(g.rand(-1..1))
          # wNC = Weight.new(g.rand(-1..1))
          # wND = Weight.new(g.rand(-1..1))
          # wNH = Weight.new(g.rand(-1..1))
          # w1N = Weight.new(g.rand(-1..1))
          # w2N = Weight.new(g.rand(-1..1))
          # w3N = Weight.new(g.rand(-1..1))
          # w4N = Weight.new(g.rand(-1..1))

          # wM = Weight.new(g.rand(-1..1))
          # #wNM = Weight.new(g.rand(-1..1))
          # wAM = Weight.new(g.rand(-1..1))
          # wBM = Weight.new(g.rand(-1..1))
          # wKM = Weight.new(g.rand(-1..1))
          # wLM = Weight.new(g.rand(-1..1))
          # wME = Weight.new(g.rand(-1..1))
          # wMF = Weight.new(g.rand(-1..1))
          # wMG = Weight.new(g.rand(-1..1))

          # declare some neurons
          # first array are inputs to the neuron. second array are the corresponding weights.

          a = Neuron.new([i0,@i1,@i2,@i3,@i4], [wA,w1A,w2A,w3A,w4A])
          b = Neuron.new([i0,@i1,@i2,@i3,@i4], [wB,w1B,w2B,w3B,w4B])
          k = Neuron.new([i0,@i1,@i2,@i3,@i4], [wK,w1K,w2K,w3K,w4K])
          l = Neuron.new([i0,@i1,@i2,@i3,@i4], [wL,w1L,w2L,w3L,w4L])
          #n = Neuron.new([i0,@i1,@i2,@i3,@i4], [wN,w1N,w2N,w3N,w4N])
          #m = Neuron.new([i0, a, b, k, l], [wM, wAM, wBM, wKM, wLM])
          c = Neuron.new([i0, a, b, k, l], [wC, wAC, wBC, wKC, wLC])
          d = Neuron.new([i0, a, k, b, l], [wD, wAD, wKD, wBD, wLD])
          h = Neuron.new([i0, b, k, a, l], [wH, wBH, wKH, wAH, wLH])
          e = Neuron.new([i0, c, d, h], [wE, wCE, wDE, wHE])
          f = Neuron.new([i0, c, d, h], [wF, wCF, wDF, wHF])
          g = Neuron.new([i0, c, d, h], [wG, wCG, wDG, wHG])

          # connect terminal neurons to performance testers

          pe = PerformanceTester.new(e)
          pf = PerformanceTester.new(f)
          pg = PerformanceTester.new(g)

          # finally declare the network

          @net = Network.new([pe, pf, pg], [a, b, c, d, e, f, g, h])                    
	end

	##############################################
	def train(x, y)
          x_data = []
          y_data = []
          for i in (0...x.row_count) do
            x_data.push(x.row(i).to_a)
            y_data.push(y.row(i).to_a)
          end
          
          @net.train(x_data, y_data, 450)
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
