require_relative 'bayes'
require 'test/unit'

class BayesTest < Test::Unit::TestCase


  def setup
    @training_data = {
      "class1" => [[1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1]],
      "class2" => [[2, 2, 2, 2, 2]]
    }

    @classifier1 = NaiveBayes.new(@training_data, 5)
  end

  def test_num_classes
    assert_equal @classifier1.num_classes, 2
  end

  def test_classes
    assert(@classifier1.classes == ["class1", "class2"] ||
      @classifier1.classes == ["class2", "class1"])
  end

  def test_feature_set
    assert_equal [1, 1], @classifier1.feature_set(1, 'class1')
    assert_equal [2], @classifier1.feature_set(2, 'class2')
  end

  def test_feature_probability
    assert_equal 0, @classifier1.feature_probability(0, 0, 'class1')
    assert_equal 1, @classifier1.feature_probability(0, 2, 'class2')
  end

  def test_classify
    assert_equal "class1", @classifier1.classify([1, 1, 1, 1, 1], )
  end
end