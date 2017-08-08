require './neural_net'
require 'test/unit'

# Note: sometimes tests fail as weights are random and
# network can get stuck in local maxima

class TestNeuralNet < Test::Unit::TestCase

  def test_input_accessors
    i1 = Input.new
    assert_equal(-1, i1.output)
    i1.set_value(2.5)
    assert_equal(2.5, i1.output)
    i2 = Input.new(5.1)
    assert_equal(5.1, i2.output)
  end

  def test_single_neuron_output
    
    i0 = Input.new
    i1 = Input.new(2)
    i2 = Input.new(4)
    
    wn = Weight.new(2)
    w1 = Weight.new(0.5)
    w2 = Weight.new(0.5)
    
    n = Neuron.new([i0, i1, i2], [wn, w1, w2])
    
    expected = 1/(1 + Math::exp(-1))
    assert_equal(expected, n.output)
  end

  def test_three_neuron_output

    i0 = Input.new
    i1 = Input.new
    i2 = Input.new

    # these weights are supposed to make the net an or-gate
    w1A = Weight.new(4.5273)
    w2A = Weight.new(4.5070)
    wA = Weight.new(2.6737)
    w1B = Weight.new(-4.1961)
    w2B = Weight.new(-4.2173)
    wB = Weight.new(-2.4721)
    wAC = Weight.new(7.8100)
    wBC = Weight.new(-7.3000)
    wC = Weight.new(0.5750)

    a = Neuron.new([i0, i1, i2], [wA, w1A, w2A])
    b = Neuron.new([i0, i1, i2], [wB, w1B, w2B])
    c = Neuron.new([i0, a, b], [wC, wAC, wBC])

    i1.set_value(0)
    i2.set_value(1)
    a.reset_cache
    b.reset_cache
    c.reset_cache
    result = c.output()
    assert((result - 1).abs < 0.1)

    i1.set_value(1)
    i2.set_value(0)
    a.reset_cache
    b.reset_cache
    c.reset_cache
    result = c.output()
    assert((result - 1).abs < 0.1)

    i1.set_value(1)
    i2.set_value(1)
    a.reset_cache
    b.reset_cache
    c.reset_cache
    result = c.output()
    assert((result - 1).abs < 0.1)

    i1.set_value(0)
    i2.set_value(0)
    a.reset_cache
    b.reset_cache
    c.reset_cache
    result = c.output()
    assert((result - 0).abs < 0.1)
  end

  def test_single_output_network_output
    i0 = Input.new
    i1 = Input.new(0)
    i2 = Input.new(1)

    # these weights are supposed to make the net an or-gate
    w1A = Weight.new(4.5273)
    w2A = Weight.new(4.5070)
    wA = Weight.new(2.6737)
    w1B = Weight.new(-4.1961)
    w2B = Weight.new(-4.2173)
    wB = Weight.new(-2.4721)
    wAC = Weight.new(7.8100)
    wBC = Weight.new(-7.3000)
    wC = Weight.new(0.5750)

    a = Neuron.new([i0, i1, i2], [wA, w1A, w2A])
    b = Neuron.new([i0, i1, i2], [wB, w1B, w2B])
    c = Neuron.new([i0, a, b], [wC, wAC, wBC])

    p = PerformanceTester.new(c)

    net = Network.new([p], [a, b, c])

    assert((net.output - 1.0).abs < 0.1)

    i1.set_value(1)
    i2.set_value(0)
    result = net.output
    assert((result - 1).abs < 0.1)

    i1.set_value(1)
    i2.set_value(1)
    result = net.output
    assert((result - 1).abs < 0.1)

    i1.set_value(0)
    i2.set_value(0)
    result = net.output
    assert((result - 0).abs < 0.1)
  end

  def test_simplest_network_learning
    g = Random.new
    wN = Weight.new(g.rand(-1..1))
    wI = Weight.new(g.rand(-1..1))

    i0 = Input.new(-1, true)
    i1 = Input.new(1)

    n = Neuron.new([i0, i1], [wN, wI])
    p = PerformanceTester.new(n)

    net = Network.new([p], [n])

    # 0 for negative values, 1 for positive values
    inputs = [[-3], [-2], [-1], [1], [2], [3]]
    desired_outputs = [[0], [0], [0], [1], [1], [1]]

    perf = net.train(inputs, desired_outputs, 10000)
    
    #puts perf

    # see if it learned the data
    inputs.zip(desired_outputs).each do |input, desired_output|
      i1.set_value(input.first)
      o = net.output
      # puts "next:"
      # puts "input: #{input.first}"
      # puts "output: #{o}"
      # puts "desired: #{desired_output.first}"
      assert((o - desired_output.first).abs < 0.1)
    end
  end

  def test_learn_or_data

    g = Random.new
    
    i0 = Input.new(-1, true)
    i1 = Input.new(0)
    i2 = Input.new(1)

    w1A = Weight.new(g.rand(-1..1))
    w2A = Weight.new(g.rand(-1..1))
    wA = Weight.new(g.rand(-1..1))
    w1B = Weight.new(g.rand(-1..1))
    w2B = Weight.new(g.rand(-1..1))
    wB = Weight.new(g.rand(-1..1))
    wAC = Weight.new(g.rand(-1..1))
    wBC = Weight.new(g.rand(-1..1))
    wC = Weight.new(g.rand(-1..1))

    a = Neuron.new([i0, i1, i2], [wA, w1A, w2A])
    b = Neuron.new([i0, i1, i2], [wB, w1B, w2B])
    c = Neuron.new([i0, a, b], [wC, wAC, wBC])

    p = PerformanceTester.new(c)

    net = Network.new([p], [a, b, c])

    inputs = [[0, 1], [1, 0], [1, 1], [0, 0]]
    desired_outputs = [[1], [1], [1], [0]]

    perf = net.train(inputs, desired_outputs, 10000)

    #puts perf <- this is currently not very informative

    i1.set_value(0)
    i2.set_value(1)
    o = net.output
    #puts "output: #{o}"
    assert((o - 1).abs < 0.1)

    i1.set_value(1)
    i2.set_value(0)
    o = net.output
    #puts "output: #{o}"
    assert((o - 1).abs < 0.1)

    i1.set_value(1)
    i2.set_value(1)
    o = net.output
    #puts "output: #{o}"
    assert((o - 1).abs < 0.1)

    i1.set_value(0)
    i2.set_value(0)
    o = net.output
    #puts "output: #{o}"
    assert((o - 0).abs < 0.1)
  end

  def test_learn_double_output_identity
    i0 = Input.new(-1, true)
    i1 = Input.new(0)
    i2 = Input.new(1)

    g = Random.new
    
    w1A = Weight.new(g.rand(-1..1))
    w2A = Weight.new(g.rand(-1..1))
    wA = Weight.new(g.rand(-1..1))
    w1B = Weight.new(g.rand(-1..1))
    w2B = Weight.new(g.rand(-1..1))
    wB = Weight.new(g.rand(-1..1))
    wAC = Weight.new(g.rand(-1..1))
    wBC = Weight.new(g.rand(-1..1))
    wC = Weight.new(g.rand(-1..1))
    wAD = Weight.new(g.rand(-1..1))
    wBD = Weight.new(g.rand(-1..1))
    wD = Weight.new(g.rand(-1..1))

    a = Neuron.new([i0, i1, i2], [wA, w1A, w2A])
    b = Neuron.new([i0, i1, i2], [wB, w1B, w2B])
    c = Neuron.new([i0, a, b], [wC, wAC, wBC])
    d = Neuron.new([i0, a, b], [wD, wAD, wBD])

    p1 = PerformanceTester.new(c)
    p2 = PerformanceTester.new(d)

    net = Network.new([p1, p2], [a, b, c, d])

    inputs = [[1, 0], [0, 1], [1, 1], [0, 0]]
    desired_outputs = inputs

    net.train(inputs, desired_outputs, 1000)

    i1.set_value(0)
    i2.set_value(1)
    o1, o2 = net.output
    #puts "output: #{o1}, #{o2}"
    assert((o1 - 0).abs < 0.1)
    assert((o2 - 1).abs < 0.1)

    i1.set_value(1)
    i2.set_value(0)
    o1, o2 = net.output
    #puts "output: #{o1}, #{o2}"
    assert((o1 - 1).abs < 0.1)
    assert((o2 - 0).abs < 0.1)

    i1.set_value(1)
    i2.set_value(1)
    o1, o2 = net.output
    #puts "output: #{o1}, #{o2}"
    assert((o1 - 1).abs < 0.1)
    assert((o2 - 1).abs < 0.1)

    i1.set_value(0)
    i2.set_value(0)
    o1, o2 = net.output
    #puts "output: #{o1}, #{o2}"
    assert((o1 - 0).abs < 0.1)
    assert((o2 - 0).abs < 0.1)
    
  end
end
