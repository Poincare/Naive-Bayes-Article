#used for mean, standard deviation, etc.
require 'descriptive_statistics'

class NaiveBayes

  #training data format:
  #{class-name: [[parameter1, parameter2, parameter3, ...],
  #              [parameter1, parameter2, parameter3,...]}
  def initialize(training_data, dim)
    @training_data = training_data
    @dimension = dim

  end

  def num_classes
    return @training_data.length
  end

  def classes
    return @training_data.keys
  end

  #returns that the values that a certain feature (determined by index)
  #takes on in a class.
  def feature_set(index, class_name)
    feature_set = []
    training_set = @training_data[class_name]

    training_set.length.times do |i|
      feature_set << training_set[i][index]
    end

    return feature_set
  end

  #given a class, this method determines the probability
  #of a certain value ocurring for a given feature
  #index: index of the feature in consideration in the training data
  #value: the value of the feature for which we are finding the probability
  #class_name: name of the class in consideration
  def feature_probability(index, value, class_name)
    #get the feature value set
    fs = feature_set(index, class_name)

    #statistical properties of the feature set
    fs_std = fs.standard_deviation
    fs_mean = fs.mean
    fs_var = fs.variance

    #deal with the edge case of a 0 standard deviation
    if fs_std == 0
      return fs_mean == value ? 1.0 : 0.0
    end

    #calculate the gaussian probability
    pi = Math::PI
    e = Math::E
    
    exp = -((value - fs_mean)**2)/(2*fs_var)
    prob = (1.0/(Math.sqrt(2*pi*fs_var))) * (e**exp)

    return prob
  end

  #multiply together the feature probabilities for all of the 
  #features in a class for given values
  def feature_mult(feature_values, class_name)
    res = 1.0
    feature_values.length.times do |i|
      res *= feature_probability(i, feature_values[i], class_name)
    end

    return res
  end

  #this is where we compute the final naive Bayesian probability
  #for a given set of features being a part of a given class.
  def class_probability(feature_values, class_name)
    class_fraction = 1.0 / num_classes
    feature_bayes = feature_mult(feature_values, class_name)
    res = feature_bayes * class_fraction
    return res
  end

  #This the method we should be calling!
  #Given a set of feature values, it decides
  #what class to categorize them under
  def classify(feature_values)
    res = classes.sort_by do |class_name|
      class_probability(feature_values, class_name)
    end

    return res[-1]
  end
end

#test out our classifier with the Iris dataset
def iris
  training_data = {"Iris-setosa" => [], 
    "Iris-versicolor" => [], 
    "Iris-virginica" => []}

  training_file = File.open('iris-partial.csv', 'r')
  training_file.each_line do |t|
    features = t.split(',')
    class_name = features.pop.chomp
    features = features.map(&:to_f)

    training_data[class_name] << features
  end

  #the dimensionality is four since we have four features
  classifier = NaiveBayes.new training_data, 4

  #we can know quiz the classifier
  quiz_file = File.open('omitted.csv', 'r')
  quiz_file.each_line do |t|
    features = t.split(',')
    expected_class_name = features.pop.chomp
    features = features.map(&:to_f)

    class_name = classifier.classify(features)
    puts (class_name == expected_class_name ? "CORRECT" : "WRONG")
  end
end

iris
