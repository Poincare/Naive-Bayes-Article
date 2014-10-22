#Introduction

We've all heard of the "Big Data" trend in computer science: everyone and their
grandmother is trying to garner insights from large datasets. Often, the tool
for the job is machine learning. A large portion of the Ruby community revolves 
around web development; even here, some basic machine learning knowledge
can come in handy if applied creatively. This article will get you up to speed with a
simple (but often quite effective) machine learning technique: the Naive Bayes
Classifier.

We'll go through the underlying theory, which does require some math background
but not too much. Then, we'll build a reuseable implementation of the classifier
and work it through an example dataset.

##Classifiers

First of all, what is a "classifier", specifically? The concept is pretty simple.
Given some descriptions (called "features") about a given object, the classifier
classifies this object into a category. Obviously, this is not a perfect process
every now and then, we get an object that has characteristics that could fit
into both categories. Classifiers are generally rated on the accuracy of their
classifications.

A classic example of classification is spam detection. Given a piece of text
the classifier has to decide whether or not it is spam. A good question to ask
would be, "What characteristics about the piece of text determine whether
or not it is spam?". These characteristics or features can be things like the
length of the text, the presence of certain words or the number of links in
the email. The classifier then uses the values of these features (they are
either boolean or real numbers) to make the classification.

##The Naive Bayes Classifier and Bayes' Theorem

So far, we've considered the general ideas behind a classifier but not the actual
mechanism. There are lots of ways to actually go about classifying stuff, each
with its own strengths and weaknesses. We'll consider the Naive Bayes Classifier (NBC)
in this article.

Let's say we have classes (i.e. categories) \[y_1, y_2, ..., y_k\]. And we are
given the set of observations \[X = x_1, x_2, ..., x_n\]. The NBC is trying to 
find the value of:

\[P(y_k \vert X = X = x_1, x_2, ..., x_n)\]

In other words, given an object some set of feature values, the NBC tries to figure out
the probability of that object being part of a given class. We do this for
every class and then classify the object into the class that gave us the highest
probability. Of course, so far we've just specified *what* we want to find, not
exactly how to find it. That's where Bayes' theorem comes into play. It says:

\[P(A \vert B) = \frac{P(B \vert A) \cdot P(A)}{B} \]

In English, that says that to find out the probability
of event A happening (for example, rain) given that event B happens (for example,
clouds skies), we multiply the probabilty of B happening given A (i.e.
cloudy skies given that its raining) and the probability of A happening (i.e.
chance of raining), and finally divide by the probability of B (i.e. chance
of cloudy skies). That is a bit complicated; reading the explanation a couple
times until it "sets" might help (drop questions in the comments below!).

We can use Bayes' Theorem to say:

\[P(y_k \vert X = x_1, x_2, ..., x_n) = 
  \frac{P(X = x_1, x_2, .. x_n \vert y_k) P(y_k)}{P(x_1, x_2, ... x_n)} \]

That's a pretty complicated looking expression. But, it is just an 
application of Bayes' Theorem, but instead of rain and cloudy skies, we are
using the class and the feature values.

Now, we still don't know how to figure out $P(X = x_1, x_2, .. x_n \vert y_k)$
(which is the probability of the feature values occurring, given
they are part of a certain class). Here we make something called the "independence
assumption", which says that "given a certain class, the values of the features
are independent of one another." For example, if we know that it is breakfast
time, the chance of it being sunny and me being asleep are independent of 
one another. There are cases where this assumption is catastrophically wrong,
but there are lots of cases where it is reasonably accurate. As it happens,
we can write this down as our expression:

\[P(y_k \vert X = X = x_1, x_2, ..., x_n) = 
  \frac{P(x_1 \vert y_k) P(x_2 \vert y_k)...\cdot P(x_k \vert y_k) P(y_k)}{P(x_1, x_2, ... x_n)} \]

We can think of this probability as a score. Once our NBC looks at the
feature values, it assigns each of the classes a score and the highest one
is the right classification for the feature values. But the denominator
of the expression has nothing to do with the class (y_k) and is therefore
constant for all the classes so we can just get rid of it. 
Our final score expression can be written as:

\[\text{Class score for } y_k = P(x_1 \vert y_k) P(x_2 \vert y_k)...\cdot P(x_k \vert y_k) P(y_k)\]

So, we've broken down the problem to the point where we need to know only two
types of things: the chance that a given feature value is part of a class (\[P(x_i \vert y_k \]) and
the chance of a class occurring (\[P(y_k)\]).

The latter can be made pretty simple if the classes are supposed to be of roughly
the same size:

$\[P(y_k) = \frac{1}{k}\]

The second one is a bit more difficult. Fortunately, this 18th century guy
called Gauss figured it out for us (using the normal distribution, which doesn't
work everywhere but should for many cases):

$\[P(x_i | y_k) = \frac{1}{\sqrt{2\pi\sigma^2_c}}\,e^{ -\frac{(v-\mu_c)^2}{2\sigma^2_c} } \]

In the above epxression, $\sigma_c$ represents the standard deviation of the i'th feature
of the $y_k$ class, and the $\mu_c$ represents the mean of the same. If this equation seems
a bit like mumbo jumbo, the only thing you really need to understand from it is that
it uses the idea that values far away from the mean value have a lower chance of ocurring
(or, more accurately, it uses a normal distribution).

Enough math for the time-being; let's get to the Ruby.

##Implementation

First of all, we need to work out how to represent our training data.
We need to have class names and for each class name, a corresponding
set of training feature values. We can make it look like this:

```ruby
{
  "class1" => [
    [feature1, feature2, feature3, ..., featureN],
    [feature1, feature2, feature3, ..., featureN]
  ],

  "class2" => [
    [feature1, feature2, feature3, ..., featureN],
    [feature1, feature2, feature3, ..., featureN]
  ],
}
```

For some parts of the implementation, we some statistical information
about certain sets (such as when computing the Gaussian probabilities). To do
that, we'll be using the [descriptive_statistics gem](https://github.com/thirtysixthspan/descriptive_statistics).

We can start by sketching out some of the basic methods:

```ruby
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
```

Notice that `@dimension` refers to the number of features we have in every
feature set (denoted "N" in the training data format described above).
Now that we have some utility methods, we can actually get cracking on the meat of
the NBC algorithm. In order to compute $P(x_i \vert y_k)$, we have to compute
the mean and standard deviation of the occurrences of a feature. Here's what I mean.
If we have the following training set:

```ruby
{
  ...
  "class1" => [
  [1, 1, 2],
  [1, 2, 2],
  [1, 16, 2],
  [1, 4, 2]
  ]
}
```

Then the standard deviation for the first and last feature (which don't change)
would be 0, whereas the same measure for the second feature is a positive number.
In order to compute this information, we need a method that can put all
the features of a certain index in a given class into one set. Ask and you
shall receive:

```ruby
def feature_set(index, class_name)
  feature_set = []
  training_set = @training_data[class_name]

  training_set.length.times do |i|
    feature_set << training_set[i][index]
  end

  return feature_set
end
```

This also outlines something that differentiates machine learning code from 
web code. On the web, you're able to see errors much more quickly than you
would in many ML applications. Since there's always possibility of your
classifier's classification being wrong, it is difficult to detect whether the source
of error is a coding mistake or an algorithm deficiency. So, in my experience, ML code is generally more 
explicit than the stuff we write for the web.

Now, we can put together the code for $P(x_i \vert y_k)$:

```ruby
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
```
Going back to our final expression for the class score:

\[\text{Class score for } y_k = P(x_1 \vert y_k) P(x_2 \vert y_k)...\cdot P(x_k \vert y_k) P(y_k)\]

So, we need to multiply together the $P(x_i \vert y_k)$ for all the indices finally multiplying that by $P(y_k)$ to get the result. Ruby makes it easy:

```ruby
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
``` 

Finally, we just have to sort the classes by their score and then return
the one with the highest score (that is the classification for a given
set of feature values):

```ruby
#This the method we should be calling!
#Given a set of feature values, it decides
#what class to categorize them under
def classify(feature_values)
  res = classes.sort_by do |class_name|
    class_probability(feature_values, class_name)
  end

  return res[-1]
end
```

The code for all of these pieces is very straightforward (except possibly
the feature_probability function, but it is essentially the translation
of a formula into Ruby) and that's in large part thanks to Ruby. Blocks,
nice loop syntax, easy sorting with a key block, etc. all make Ruby a very
pleasant language for machine learning. 

##Test Drive
Now that we've put together the NaiveBayes class, we can test it out. The Iris
dataset can be considered something of a standard in the machine learning community.
It provides data about petal and sepal dimensions for certain Iris flowers and a species
label for each entry.

An example of an entry would be:

```
4.8,3.0,1.4,0.3,Iris-setosa
```

Our goal is to train our Naive Bayes Classifier to take feature values (the numbers)
and tell us what kind of species the numbers are likely to be. Essentially,
we just have to put together the training data set and quiz the classifier
with a separate file of entries which are not part of the training set. You
can find both the code and the data in this article's repository. As it happens,
the NBC is able to correctly identify 11 out of 12 feature values, which is
pretty good considering the small training set.

##Wrapping It Up

The go to language for MLers these days seems to be Python. In part, this might
be due to the fact that Python has extensive library support in the form of
scikit-learn and numpy and also because Ruby's web-centered community
is not extremely active in machine learning. I think that Ruby is a nice language
to implement machine learning algorithms in (particularly simple ones) and also I 
feel that Web developers would also do well to pick up some ML tricks!





