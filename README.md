# OCR---Devanagari-Lipi-Recognition
This is an optical character recognition project made using python. The recognition model is trained using the Convolutional Neural Network.

CONVOLUTIONAL NEURAL NETWORKS
When it comes to Machine Learning, Artificial Neural Networks perform really well. Artificial Neural
Networks are used in various classification tasks like images, audio, and words. Different types of Neural
Networks are used for different purposes. For image classification, we use Convolution Neural Networks.
A Convolutional Neural Network (ConvNet/CNN) is a Deep Learning algorithm that can take in an input
image, assign importance (learnable weights and biases) to various aspects/objects in the image, and be able to
differentiate one from the other. The pre-processing required in a ConvNet is much lower as compared to other
classification algorithms. While in primitive methods filters are hand-engineered, with enough training,
ConvNets have the ability to learn these filters/characteristics.

Libraries Used:-
● Numpy- NumPy is a general-purpose array-processing package. It provides a high-performance
multidimensional array object, and tools for working with these arrays. It is the fundamental package for
scientific computing with Python. It is open-source software.
15
● Pandas- Pandas is an open-source library that is made mainly for working with relational or labeled data
both easily and intuitively. It provides various data structures and operations for manipulating numerical data
and time series. This library is built on top of the NumPy library. Pandas is fast and it has high performance &
productivity for users.
● Matplotlib- Matplotlib is an amazing visualization library in Python for 2D plots of arrays. Matplotlib is
a multi-platform data visualization library built on NumPy arrays and designed to work with the broader SciPy
stack. It was introduced by John Hunter in the year 2002.One of the greatest benefits of visualization is that it
allows us visual access to huge amounts of data in easily digestible visuals. Matplotlib consists of several plots
like line, bar, scatter, histogram etc
● Tensorflow- TensorFlow is an open-source software library. TensorFlow was originally developed by
researchers and engineers working on the Google Brain Team within Google’s Machine Intelligence research
organization for the purposes of conducting machine learning and deep neural networks research, but the system
is general enough to be applicable in a wide variety of other domains as well.TensorFlow is basically a software
library for numerical computation using data flow graphs where:
o nodes in the graph represent mathematical operations.
o edges in the graph represent the multidimensional data arrays (called tensors) communicated between
them. (Please note that tensor is the central unit of data in TensorFlow).
● Keras- Keras is an open source library for implementing neural network written in Python. It is capable
of running on top of TensorFlow, Theano and Microsoft Cognitive tool. It focuses on being user-friendly,
modular, and extensible and is designed to enable fast experimentation with deep neural networks.
A Keras model is realized as a sequence or a stand-alone, fully-configurable modules' graph that can be
plugged together with as little as restrictions as possible. New models can be created by combining various
modules like layers, loss functions, optimization algorithms, initialization techniques, activation functions, etc..
Keras is suitable for advanced research in neural networks as new modules are simple to add just as new classes
and functions,
● Sklearn- Scikit-learn is an open source data analysis library, and the gold standard for Machine
Learning (ML) in the Python ecosystem. Key concepts and features include:
Algorithmic decision-making methods, including:
o Classification: identifying and categorizing data based on patterns.
Regression: predicting or projecting data values based on the average mean of existing and planned data.
o Clustering: automatic grouping of similar data into datasets.
Algorithms that support predictive analysis ranging from simple linear regression to neural network pattern
recognition. Interoperability with NumPy, pandas, and matplotlib libraries.
● SciPy- SciPy in Python is an open-source library used for solving mathematical, scientific, engineering,
and technical problems. It allows users to manipulate the data and visualize the data using a wide range of
high-level Python commands. SciPy is built on the Python NumPy extention. SciPy is also pronounced as “Sigh
Pi.”
● Cv2- OpenCV is a huge open-source library for computer vision, machine learning, and image
processing. OpenCV supports a wide variety of programming languages like Python, C++, Java, etc. It can
process images and videos to identify objects, faces, or even the handwriting of a human. When it is integrated
with various libraries, such as Numpy which is a highly optimized library for numerical operations, then the
number of weapons increases in your Arsenal i.e whatever operations one can do in Numpy can be combined
with OpenCV.

Devanagari Lipi Detection Dataset
Dataset used for this project consisted of 46 Devanagari characters each having 2000 images. Each image
was 32 x 32 pixels in size. 46 classes consisting of 36 devanagari characters and 10 devanagari numbers from 0
to 9.
