# How does the model make predictions? A systematic literature review on the explainability power of machine learning in healthcare.

This ReadMe contains supplementary material for our research paper about explainable AI in medicine.
It is divided into two parts: **PRISMA Literature Review** and **Explainable Artificial Intelligence Methods**.
The first part shows which search terms we used and how the questionnaire looks like that we filled out for each paper during the full screening process.
The second parts gives a deeper introduction into each XAI method that we considered for our review, which is beyond the scope of the main paper.
For any questions, please contact the corresponding author [johannes.allgaier@uni-wuerzburg.de](johannes.allgaier@uni-wuerzburg.de). 

## PRISMA Literature Review

**PubMed Search Term Results (2568 Papers as by 2022-03-07)**
```
(((explainab*[Title/Abstract] OR interpretab*[Title/Abstract] OR understandab*[Title/Abstract] OR comprehensib*[Title/Abstract] OR intelligib*[Title/Abstract]) 
AND 
("machine learning"[Title/Abstract] OR ml[Title/Abstract] OR model[Title/Abstract] OR "deep learning"[Title/Abstract] OR neural network[Title/Abstract] OR ann[Title/Abstract] OR "artificial intelligence"[Title/Abstract])) 
AND 
(medic*[Title/Abstract] OR health*[Title/Abstract] OR health[Title/Abstract] OR radiology[Title/Abstract] OR patient[Title/Abstract] OR doctor[Title/Abstract] OR physician[Title/Abstract] OR surgeon[Title/Abstract])) 
NOT 
("literature review"[Title/Abstract] OR "systematic review"[Title/Abstract] OR Prisma[Title/Abstract])
Filters: Abstract, from 2002 - 2022
```

### PRISMA Flowchart
![](/plots/prisma-result_alt2.png)

### Research questions
![](/readme_img/ms_forms.png)

## Explainable Artificial Intelligence Methods

| Method / Taxonomy                                                                                                                                                               | Specific (S)<br> or <br>Agnostic (A) | Local (L)<br>or<br>Global (G) | Intrinsic (I)<br>or<br>Post-Hoc (P) | Adresses<br>Neural Networks | Adresses<br>Computer Vision Tasks | Addresses <br>Tabular Data | Year <br>of publication | \# citations in<br>Google Scholar<br>as of 21-12-03 | Regr. (R)<br>or<br>Classif. (C) | Source Code<br>Freely Available                                                                                  |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------- | ----------------------------- | ----------------------------------- | --------------------------- | --------------------------------- | ---------------------- | ------------------- | ----------------------------------- | ------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| [Partial Dependence Plots (PDP)](https://www.jstor.org/stable/2699986?seq=1#metadata_info_tab_contents)                                                                          | A                            | G                             | P                                   | 🛇                          | 🛇                                | ✓                      | 2001                | 15545                               | R and C                         | [✓](https://scikit-learn.org/stable/modules/partial_dependence.html)                                             |
| [Permutation Importance](https://arxiv.org/abs/1801.01489)                                                                                                                      | A                            | G                             | P                                   | 🛇                          | 🛇                                | ✓                      | 2010                | 15545                               | R and C                         | [✓](https://scikit-learn.org/stable/modules/permutation_importance.html)                                         |
| [Mean Decrease Impurity](https://proceedings.neurips.cc/paper/4928-understanding-variable-importances-in-forests-of-randomized-trees.pdf)                                       | S                            | G                             | I                                   | 🛇                          | 🛇                                | ✓                      | 2013                | 823                                 | R and C                         | 🛇                                                                                                               |
| [Individual Conditional Expectation](https://www.tandfonline.com/doi/abs/10.1080/10618600.2014.907095)                                                                          | A                            | L                             | P                                   | ✓                           | 🛇                                | ✓                      | 2013                | 571                                 | R and C                         | [✓](https://scikit-learn.org/stable/modules/partial_dependence.html#individual-conditional-expectation-ice-plot) |
| [Deep Learning Important FeaTures (DeepLift)](http://proceedings.mlr.press/v70/shrikumar17a)                                                                                    | S                            | L                             | P                                   | ✓                           | ✓                                 | 🛇                     | 2016                | 1629                                | C                               | [✓](https://github.com/kundajelab/deeplift)                                                                      |
| [Layer-Wise Relevane Propagation](https://arxiv.org/abs/1604.00825)                                                                                                             | S                            | L                             | P                                   | ✓                           | ✓                                 | 🛇                     | 2016                | 2160                                | C                               | [✓](https://github.com/sebastian-lapuschkin/lrp_toolbox)                                                         |
| [Maximum Mean Discrepancy - Critic](http://papers.nips.cc/paper/6300-examples-are-not-enough-learn-tocriticize-criticism-for-interpretability.pdf)                              | A                            | G                             | P                                   | ✓                           | ✓                                 | 🛇                     | 2016                | 445                                 | C                               | [✓](https://github.com/BeenKim/MMD-critic)                                                                       |
| [Gradient-weighted Class Activation Mapping](https://arxiv.org/abs/1610.02391)                                                                                                  | S                            | L                             | P                                   | ✓                           | ✓                                 | 🛇                     | 2016                | 6758                                | C[^1]                             | [✓](https://github.com/ramprs/grad-cam)                                                                          |
| [Integrated Gradients](https://arxiv.org/abs/1703.01365)                                                                                                                        | S                            | L                             | P                                   | ✓                           | ✓                                 | 🛇                     | 2017                | 2017                                | C                               | [✓](https://github.com/ankurtaly/Integrated-Gradients)                                                           |
| [Local Interpretable Model-agnostic Explanation (LIME)](http://papers.nips.cc/paper/2017/file/8a20a8621978632d76c43dfd28b67767-Paper.pdf)                                                                                        | A                            | L                             | P                                   | ✓                           | ✓                                 | ✓                      | 2017                | 5020                                | R and C                         | [✓](https://github.com/marcotcr/lime)                                                                            |
| [SHapely Additive exPlanations(SHAP)](https://arxiv.org/abs/1705.07874)                                                                                                               | A                            | Both                          | P                                   | ✓                           | ✓                                 | ✓                      | 2017                | 5020                                | R and C                         | [✓](https://github.com/slundberg/shap)                                                                     |
| [Deep Lattice Networks<br>and Partial Monotonic Functions](https://arxiv.org/abs/1709.06680)                                                                                    | S                            | G                             | P                                   | ✓                           | 🛇                                | ✓                      | 2017                | 88                                  | R and C                         | 🛇                                                                                                                |
| [Leave One Covariate Out](https://arxiv.org/abs/1604.04173)                                                                                                                     | A                            | L                             | P                                   | 🛇                          | 🛇                                | ✓                      | 2017                | 274                                 | R                               | [✓](https://github.com/ryantibs/conformal)                                                                       |
| [Influence Functions](https://arxiv.org/abs/1703.04730)                                                                                                                         | A                            | L                             | P                                   | ✓                           | ✓                                 | 🛇                     | 2017                | 1377                                | C                               | [✓](https://github.com/kohpangwei/influence-release)                                                             |
| [Soft Decision Trees](https://arxiv.org/abs/1711.09784)                                                                                                                         | S                            | G                             | P                                   | ✓                           | 🛇                                | 🛇                     | 2017                | 357                                 | C                               | [✓](https://github.com/xuyxu/Soft-Decision-Tree)                                                                 |
| [SmoothGrad](https://arxiv.org/abs/1706.03825)                                                                                                                                  | S                            | L                             | P                                   | ✓                           | ✓                                 | 🛇                     | 2017                | 867                                 | C                               | [✓](https://github.com/PAIR-code/saliency)                                                                       |
| [Testing Concept Activation Vectors](https://arxiv.org/abs/1711.11279)                                                                                                          | S                            | Both                          | P                                   | ✓                           | ✓                                 | 🛇                     | 2018                | 583                                 | C                               | [✓](https://github.com/tensorflow/tcav)                                                                          |
| [Anchors](https://homes.cs.washington.edu/~marcotcr/aaai18.pdf)                                                                                                                 | A                            | L                             | P                                   | ✓                           | ✓                                 | ✓                      | 2018                | 922                                 | R and C                         | [✓](https://github.com/marcotcr/anchor)                                                                          |
| [Representer Point Selection](https://arxiv.org/abs/1811.09720)                                                                                                                 | S                            | L                             | P                                   | ✓                           | ✓                                 | 🛇                     | 2018                | 105                                 | C                               | [✓](https://github.com/chihkuanyeh/Representer_Point_Selection)                                                  |
| [eXplanation with Ranked Area Integrals (XRAI)](https://openaccess.thecvf.com/content_ICCV_2019/html/Kapishnikov_XRAI_Better_Attributions_Through_Regions_ICCV_2019_paper.html) | S                            | L                             | P                                   | ✓                           | ✓                                 | 🛇                     | 2019                | 47                                  | C                               | [✓](https://github.com/PAIR-code/saliency)                                                                       |
| [Automatic Concept-based Explanations](https://arxiv.org/abs/1902.03129)                                                                                                        | S                            | G                             | P                                   | ✓                           | ✓                                 | 🛇                     | 2019                | 157                                 | C                               | [✓](https://github.com/amiratag/ACE)                                                                             |
| [Attention-Based Prototypical Learning](https://arxiv.org/abs/1902.06292)                                                                                                           | S                            | L                             | P                                   | ✓                           | ✓                                 | ✓                      | 2019                | 8                                   | C                               | 🛇                                                                                                               |
| [Quantifiable Feature Importance Technique (Q-FIT)](https://arxiv.org/abs/2010.13872)                                                                                                                                       | A                            | L                             | P                                   | 🛇                          | 🛇                                | ✓                      | 2020                | n. A.                               | R and C                         | 🛇                                                                                                               |
| [TabNet](https://www.aaai.org/AAAI21Papers/AAAI-1063.ArikS.pdf)                                                                                                                 | S                            | Both                          | P                                   | ✓                           | 🛇                                | ✓                      | 2020                | 100                                 | R and C                         | 🛇                                                                                                                |
| [Wilks feature importance](https://hess.copernicus.org/articles/25/4947/2021/hess-25-4947-2021.html)                                                                            | S                            | G                             | I                                   | 🛇                          | 🛇                                | ✓                      | 2021                | 1                                   | R and C                         | 🛇                                                                                                               |

## Explanation methods in more detail

#### Partial Dependence Plots
**Explanation**
Partial dependence plots (PDP) is a global, model-agnostic method that shows the marginal effect of a feature on the target. In principal, PDP answers the question \textit{"What is the relation of these features to the target given that other features are held constant?"}. 
**Application**
The plots are applicable on any type of features (categorical and continuous) as well as vor classification and regression problems.
**Advantages**
Since this method is detached from the ML model, one is independent of the algorithm and primarily looks at the data itself. An implementation can be seen [here](https://scikit-learn.org/stable/auto_examples/inspection/plot_partial_dependence.html).
**Limitation**
However, PDP makes the naive assumption that features have no correlation with each other. Averaging many data points further results in loss of information regarding the heterogeneity of features.

#### Permutation importance
**Explanation**
Permutation importance is a model-agnostic and global method to estimate how important a feature for a given trained model is(altmann2010permutation). It can be used for both regression and classification tasks. Permutation Importance is defined as the absolute difference in the performance score when a feature is replaced by a dummy feature. The more the performance drops, the more important this feature is for the model. 
A little-noticed variance of permutation importance is the \textbf{perturbation rank}, in which the values within a feature are shuffled(du2008perturbationrank). The advantage of this method is that statistical properties of feature and dummy feature remain identical.
**Application**
For Python users, there is an implementation of the method in [sklearn](https://scikit-learn.org/stable/modules/permutation_importance.html#id2). The method can be used whenever tabular data is used, regardless of the model.
**Advantages**
An advantage of the method is its intuitive comprehensibility to stakeholders and the open-source implementation by sklearn.
**Limitation**
A disadvantage is that the Permutation Importance depends on the model and the selected performance score. A modification of the performance scores can mean a change in the feature rankings. In addition, this method cannot take into account co-variances between features.

#### Mean Decrease Impurity
**Explanation**
Mean Decrease Impurity (MDI) is a global and model-specific method for explaining feature importances of ensembles of trees(louppe2013understanding}. It aims to identify irrelevant features for the target and attributes a relevance to each feature. For each feature, it calculates the importance (Gini or Shannon\cite{shannon1948mathematical), i.e.) as the sum over the number of splits for all trees with that feature, proportionally proto the total number of samples it splits.
**Application**
Conceivable applications are classification tasks with categorical variables as input, although according to the authors, regression tasks are also conceivable if one varies the method used to measure impurity.
**Advantages**
The methodology is mathematically sound within the paper.
**Limitation**
However within the paper, MDI is shown using categorical input and output variables, which limits potential usecases. At the same time, on the [documentation side](https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance.html) of permutation importance, sklearn criticizes a bias of MDI towards categorical variables. Perturbation Ranking, introduced by [Jeaff Heaton](https://www.youtube.com/watch?v=htNL1iiBnDE) shuffles the values within one feature, however, the statistical property of the feature (min, max, mean, std) remains the same.

#### Individual Conditional Expectation
**Explanation**
Individual Conditional Expectations (ICE) are a refinement of Partial Dependence Plots and address the heterogeneity of individual data points(goldstein2015peeking). It is a local, model-agnostic and post-hoc method which simply disaggregates PDPs to shed light on individual conditional expectations.
**Application**
It can be applied to supervised applications and is essentially used to illustrate up to 3 variables. There is an implementation in [R](https://cran.r-project.org/web/packages/ICEbox/index.html) and [Python](https://scikit-learn.org/stable/modules/partial_dependence.html#individual-conditional-expectation-ice-plot).
**Advantages**
In classical PDPs, averaging results in a loss of information. Disaggregation can reverse this loss of information.
**Limitation**
ICEs can become confusing if there are too many heterogeneous data points. Although this shows the heterogeneity of the problem, it also makes it difficult to derive concise statements.

#### DeepLIFT
**Explanation**
Deep Learning Important FeaTures(shrikumar2016not) is a local and model-specific method using to explain individual predictions of neural networks. It can also be used for computer vision applications. DeepLIFT looks at how much each neuron in a neural network is activated relative to a reference input for an individual input. The reference input is neutral (*foil*), whereas the individual input can be described as *fact*.
% Appplication
In the paper, DeepLIFT is applied to MNIST dataset and for classification of DNA sequences.
**Advantages** to other methods
Other methods like (zeiler2014visualizing, zhou2015predicting, zintgraf2017visualizing) also need a forward propagation to for each perturbation and might therefore be computationally inefficient.
**Limitation**
DeepLift itself has the limitation that it is difficult to generate a suitable reference input (foil) from the data to explain the individual input relative to the reference input.

#### Maximum Mean Discrepancy - Critic
**Explanation**
Maximum Mean Discrepancy (*MMD-Critic*) is a global model-agnostic method that distinguishes representative samples of a class from outliers(kim2016examples). Typical representative samples are called prototypes, the outliers are called criticism. Samples in a distribution with high sample density are seen as good prototypes. By detecting the criticisms, a higher interpretability of black-box models should be achieved.
**Application**
Typical applications are examples of computer vision models, but tabular applications are also conceivable. The source code is freely available on [GitHub](https://github.com/BeenKim/MMD-critic).
**Advantages**
MMD-Critic works for any data type and any model. It is therefore maximally flexible. It can help people when labeling images to recognize untypical images of a class more reliably.
**Limitation**
A critisicm is not necessarily harder to classify from the model. As an alternative, a classical error analysis of misclassified samples can help to detect difficult samples systematically by examining these images for concepts.

#### Local Interpretable Model-agnostic Explanation
**Explanation**
Local Interpretable Model-agnostic Explanation ([LIME](https://github.com/marcotcr/lime)) is a popular, open-source, and post-hoc method that learns an interpretable model around a single prediction.
Using data points close to the individual predictions, LIME trains an interpretable model to approximate the predictions of the real model. The new interpretable model is then used to interpret the result, which is also called local fidelity. 
**Application**
LIME can locally explain text-models from tree-based algorithms as well as computer-vision models, such as deep neural networks.
**Advantages**
LIME breaks the complexity of a global model by taking samples that are locally close to a prediction.
**Limitation**
It has been shown that a random generation of noise results in an instability of the generated explanations by LIME(zhang2019should, zafar2019dlime). This results in modifications of the originally posted LIME approach, i.e. S-LIME(zhou2021s) or DLIME(zafar2019dlime).

#### Shapley Additive explanations
**Explanation**
SHapley Additive exPlanations (SHAP) is a model-agnostic method that allows both global and local explanations and also addresses structured as well as unstructured data(NIPS2017_7062). SHAP is the contribution of a feature value to the difference between the actual prediction and the mean prediction. The popularity of SHAP is not least explained by the freely available source code on [GitHub](https://github.com/slundberg/shap).
SHAP builds on Shapley values from game theory(shapley1951notes}, propagation activation features(shrikumar2016not), and model-intrinsic approaches from [tree-based methods](http://blog.datadive.net/interpreting-random-forests/), among others.
**Application**
SHAP is written in Python and can be applied to models from Tensorflow, Keras, Pytorch and Sci-Kit learn. The built-in visualization functions facilitate the interpretation of the methods.
**Advantages**
By combining different methods on a high-level API, SHAP is also available to a wider audience. This is a decisive advantage over other methods.
**Limitation**
However, there is also the danger that SHAP is applied without questioning the limitations of the underlying methods.

#### Deep Lattice Networks
**Explanation**
Deep Lattice Networks (DLN) is a model-specific method that provides global interpretability using a nine-layers network(you2017deep}. The concept is based on look-up tables(garcia2009lattice). The model itself is interpretable because of monotonicity constraints (gupta2016monotonic). This means that one feature is varied and all other parameters are kept constant. It then looks at the marginal change in the outcome for a marginal change in the input.
**Application**
Deep Lattice Networks have the advantage in the application that domain knowledge can be injected into the model, thus limiting unpredictable behavior in real applications. An overview of this concept is given [here](https://www.tensorflow.org/lattice/overview). The source code is freely available on [here](https://github.com/tensorflow/lattice).
**Advantages**
The fact that subject matter expert knowledge can be incorporated into the model is an advantage over other methods. In this way, confidence in the model is generated even before it is implemented.
**Limitation**
For high-dimensional input spaces with more than 20 features, the authors do not recommend the use of DLNs.

#### Leave One Covariate Out
**Explanation**
Leave One Covariate Out (LOCO) is a model-agnostic, global and local feature importance method similar to feature importances in random forests(lei2018distribution). In contrast to feature importance in random forests, however, the feature under consideration is not replaced by a dummy variable, but simply dropped. 
**Application**
The authors themselves describe regressions as a use case, although classification is also conceivable. There is also a [GitHub](https://github.com/ryantibs/conformal) repository freely available in R.
**Advantages**
One advantage of this method is its simple implementation. Although there is a GitHub repository for R, you can also implement LOCO yourself with a for loop.
**Limitation**
It remains unclear whether LOCO offers a real advantage over Breimann's older feature importance.

#### Influence Functions
 **Explanation**
Influence Functions is a local, model-agnostic explainability method for providing training points most responsible for a given test sample(koh2017understanding).
An Influence Functions treats the model as a function of the training data. It gives more weight to a single sample and examines the change in output when that sample is changed.
**Application**
In the paper, the Influence Functions are applied to animal images. The authors also show the outlier sensitivity of a model when noise is applied to important images and added to the training set. The source code is available on [GitHub](https://github.com/kohpangwei/influence-release).
**Advantages**
The method can be applied to all machine learning models whose 2nd degree derivative exists.
**Limitation**
However, the method is very computationally expensive because the model must be re-trained when the training data changes. In addition, the boundary of an influencing or non-influencing training example is unclear.

#### Soft Decision Trees
**Explanation**
Soft Decision Trees is a model-specific and global interpretability method which uses a decision tree to mimic the input-output function of a neural network(frosst2017distilling}. In a soft decision tree, all the leave nodes contribute to the final decision with different probabilities\cite{irsoy2012soft).
**Application**
The authors demonstrate the soft decision tree using the MNIST dataset. Inner nodes of a soft decision tree represent learned filters of the neural network. The code was re-implemented by third parties on [GitHub](https://github.com/xuyxu/Soft-Decision-Tree).
**Advantages**
For some leave nodes, the soft decision tree allows the visual interpretation of the neural network. The simplification of the complex network architecture results in a leaner model with relatively low performance loss.
**Limitation**
Not all learned filters are interpretable to the human eye. The explainability of this method is therefore limited.

#### Testing Concept Activation Vectors
**Explanation**
Testing Concept Activation Vectors (TCAV) is a model-specific, global and local explainability method for computer vision models and tabular, discrete data(kim2018interpretability). TCAV gives an explanation (i.e., a concept) that generally applies to a class which is beyond one image. It learns the concept from examples. The concepts are learned through delineation examples. For example, to learn the concept feminine, some images of feminine must be shown in differentiation from non-feminine.
**Application**
The source code is freely available on [GitHub](https://github.com/tensorflow/tcav). In order to apply TCAV, two data sets must be provided. One representing the concept and a random dataset for delineation. We then train a binary classifier to distinguish between the concept and the random data. The coefficient vector of the classifier is then called a concept vector.
**Advantages**
Since people think in concepts and not in numbers, this method is also applicable for non machine learning experts.
**Limitation**
The concept datasets need additional labels and therefore could be expensive to create. Also, abstract (e.g. sadness) or too general concepts are difficult to learn.

#### Anchors
**Explanation**
Anchors is a model-agnostic local explanation method developed by the LIME authors(ribeiro2018anchors}. Based on a prediction, relevant features are determined. If a marginal change in other features does not change the prediction, then the rule is \textit{anchored}. The outputs of the anchors approeach are \texttt{IF-THEN) rules. 
**Application**
Anchors can be applied to structured predictions, tabular classification, image classification, and visual question answering. The source code is freely available in [Python](https://github.com/marcotcr/anchor) and [Java](https://github.com/viadee/javaAnchorExplainer).
**Advantages**
By generating if-then rules, the output of this explanation method is easy to understand even for non machine learning experts. In addition, Anchor offers a very wide range of applications.
**Limitation**
Rules for rare classes or near the boundary of decision functions can become complex and sometimes ambiguous. With complex output, different rules can also become the same prediction. In high dimensional spaces also every small change can lead to a change of the prediction, which makes the coverage of the rule very low.

#### Representer Point Selection
**Explanation**
Representer Point Selection is a model-specific, local explainbility method for computer vision applications(yeh2018representer}. For a given test image, \textit{representer points) are similar images from the training set and are close to the decision boundary. Positive representer points belong to the same class as the test image, negative ones to a different class. 
**Application**
The method can be applied to any image classification task. The source code is available on [GitHub](https://github.com/chihkuanyeh/Representer_Point_Selection).
**Advantages**
By showing these images, the method helps in error analysis and model understanding. Within the paper, examples are also shown which demonstrate an improvement to the Influence Functions method.
**Limitation**
It is questionable whether the method has advantages over a classical error analysis. By displaying the misclassified images, one can look for systematic errors by clustering them. This is in essence also what Influence Functions and Representer Point Selections do.

#### Explanation Ranked Area Integrals
**Explanation**
EXplanation Ranked Area Integrals (XRAI) is a saliency and region-based attribution based method which builds upon integrated gradients(recio2021case). It is model-specific for local interpretability of computer vision tasks. Further, it assesses overlapping regions of the image to create a saliency map and ultimately combines a segmented image with the saliency map to highlight the pixels of the input image that contributed to the classification at most. 
**Application**
General UseCases are the classification of image data. Google recommends to use this method for natural images with multiple objects in it. It is used for X-rays as well. 
**Advantages** to other methods
The paper highlights the superiority over previous methods such as Integrated Gradients(sundararajan2017axiomatic} or GradCAM\cite{selvaraju2017grad). By combining integrated gradients with image segmentations, a higher discriminatory power from activated to non-activated pixels is achieved.
**Limitation**
Within the task of image classification, the input image should be well segmentable. If no selectivity can be achieved by segmentation, the output of this method becomes noisy.

#### Automatic Concept-Based Explanations
**Explanation**
Automatic Concept-Based Explanations (ACE) is a global, model-specific interpretability method to cluster and visualize segments of an image that are important for a particular class(ghorbani2019towards). Multiple images for one class are segmented using the activation space of a layer of a pre-trained neural network as a simliarity score. Similar segments of the images are then pooled together. Finally, each pool is assigned a TCAV importance score.
**Application**
The method can be applied for any computer vision classification model. The source code is written in Python and available on [GitHub](https://github.com/amiratag/ACE).
**Advantages**
By pooling multiple images, this method can also be considered as a global explanation method for computer vision usecases.
**Limitation**s
The explanation method only works if the concepts are present in the form of groups of pixels. Abstract concepts do not work. 

#### Attention-Based Prototypical Learning
**Explanation**
Attention-Based Prototypical Learning (ProtoAttend) is, similar to Representer Point Selection and MMD-Critic, a sample-based interpretability method(arik2019protoattend}. It is model-specific for neural networks, local and addresses computer vision tasks as well as text classification. The prediction of a model is essentially explained by attaching similar examples with the same prediction. The similar examples are referred to as \textit{prototypes).
**Application**
The authors demonstrate the functionality using classic image classifications from [DBPedia](http://yann.lecun.com/exdb/mnist) and [MNIST](https://github.com/zalandoresearch/fashion-mnist}{Fashion-MNIST). In addition, texts from [Wikipedia](http://wikidata.dbpedia.org/develop/datasets) are classified and also a binary classification from tabular data is provided.
**Advantages**
If the prototypes shown have a different label than the one predicted, misclasification of the model can be better understood. In this case, more samples can be purposefully collected.
**Limitation**
However, the quality of the prototypes may decrease if there are few related examples or there is a high intra-class variance of examples, or the encoder of the relevant features acts differently from the human.

#### TabNet
**Explanation**
TabNet is a model-specific global feature selection method whose explainability plays a role at the margin(arik2020tabnet). Designed for tabular data, it follows the logic of a decision tree, but instead of taking the Gini importance for the decision boundary, sequential attention of neural networks is used. The code if freely available on [GitHub](https://github.com/dreamquark-ai/tabnet).
**Application**
TabNet can be applied to all tabular data, furthermore also for semi-supervised applications where the labels are partially missing.
**Advantages**
The authors argue that models with TabNet sometimes achieve higher Accuracies. The feature importance, which can be retrieved e.g. also from Random Forests, gives an interpretation approach for global explanations.
For local explanations, a heatmap is provided that highlights the features used per sample. 
**Limitation**
However, the instance-wise feature selection can lead to confusion in the application if local and global feature importances contradict each other.


%Rachel is working on this section

### Gradient-Based Explanation Methods

Gradient-based explanation methods leverage a model's gradient to produce an explanation (ancona2019XAIBook). They are most commonly applied to image classification neural networks, for which they yield a heatmap for a particular input image and output class. The heatmap is intended to indicate which regions of the input image were used to predict the output class.

#### Input-Level Approaches

Input-level approaches involve gradient or gradient-like computations that proceed from the output layer of the network all the way back to the input layer to produce a heatmap explanation with the same number of pixels as the input image. Input-level approaches include Saliency Mapping, Guided Backpropagation, Deconvolutional Networks,
Gradient $\times$ Input, and Layer-Wise Relevance Propagation, detailed further in the next sections.

%Advantages/Limitations
All these approaches are computationally efficient, but suffer from a white noise appearance caused by shattered gradients (balduzzi2017shattered) that sometimes prevents the resulting heatmaps from appearing class-specific in practice.

#### Saliency Mapping, Guided Backpropagation, and Deconvolutional Networks
%Explanation
Saliency Mapping is the original gradient-based explanation method for neural networks. Saliency Mapping computes the gradient of the class score with respect to the input image (simonyan2014deep}. DeconvNets (zeiler2014visualizing} and Guided Backpropagation \cite{springenberg2014striving) are explanation methods developed independently that happen to be mathematically identical to saliency mapping except for handling of the nonlinearities \cite{nie2018theoretical).
%Application/Advantages
All three techniques can be applied to any classification neural network.
%Limitation
Saliency mapping passes explanation method sanity checks, while Guided Backpropagation does not and may in fact function more like an edge detector than a model explanation (adebayo2018sanity). 

#### Gradient $\times$ Input

The Gradient $\times$ Input method is the same as saliency mapping, except that the saliency map is multiplied element-wise against the input image to create the final visualization. Gradient $\times$ Input fails sanity checks (adebayo2018sanity).

#### Layer-Wise Relevance Propagation
**Explanation**
Layer-Wise Relevance Propagation (LRP)(bach2015pixel} produces relevance scores for the input pixels by iteratively distributing the final score across the neural network's layers, starting from the output layer and proceeding backwards to the input layer. Values greater than zero indicate that a particular pixel is relevant for the chosen class. There are several variants of LRP; while LRP was not originally described as a gradient-based explanation method, it was later shown \cite{ancona2019XAIBook) that $\epsilon$-LRP is a variant of the Gradient $*$ Input method in which the gradient calculation is modified based on the ratio between the output and input at each nonlinearity.
**Application**
LRP has been applied to image classification models, bag-of-words models(csurka2004visual}, and Fisher Vectors\cite{lapuschkin2016analyzing).

#### Output-Level Approaches
Output-level gradient-based explanations involve computation of a gradient that proceeds from the output layer backwards for only one or a few layers. The computation does not proceed all the way back to the input layer. Thus, in a convolutional neural network, the raw explanation is of smaller dimension than the input image and must be upsampled before being overlaid with the input image to create the final explanation. Such an upsampling step is permitted because in a typical CNN there will not be any internal rotation or flipping steps between convolutions, and so the spatial relationship between the later convolutional representation and the input image is preserved. Output level approaches include Class Activation Mapping (CAM), Grad-CAM, Guided Grad-CAM, and HiResCAM.

#### Class Activation Mapping (CAM)


#### {Gradient-weighted Class Activation Mapping (Grad-CAM) and Guided Grad-CAM}
**Explanation**
Gradient-weighted Class Activation Mapping (Grad-CAM) is a model-specific, local and post-hoc explainability method for computer vision tasks and reinforcement learning(selvaraju2017grad). It calculates a linear combination of neuron importance weights and feature map activations for the last convolutional layer as this layer has the best compromise between spatial information and high-level semantics. Grad-CAM basically answers the question: Which part of an image is important for a specific classification?
**Application**
Grad-CAM can be used to explain computer vision models solving object detection, image classification and visual question answering tasks. The model has been evaluated on datasets like [ImageNet](https://paperswithcode.com/dataset/imagenet), [COCO](https://paperswithcode.com/dataset/coco), [Visual Question Answering](https://paperswithcode.com/dataset/visual-question-answering) and [Places](https://paperswithcode.com/dataset/places). The source code is freely available on [GitHub](https://github.com/ramprs/grad-cam).
**Advantages**
Although we categorize the method as model-specific, it is applicable to a variety of CNN model families such as fully connected ones, multi-modal inputs for visual question answering or structured outputs such as captioning.
Unlike the older CAM algorithm(zhou2016learning), the Grad-CAM method is a generalization in that it does not require a specific model architecture.
**Limitation**
A potential limitation could be the "Guided Grad-Cam" variation presented in the paper. Guided backpropagation acts more like an edge detector than providing insights into the model behavior(adebayo2018sanity, nie2018theoretical}. Solutions for this could lie in further developments of CAM, such as Grad-CAM++\cite{chattopadhay2018grad).

#### Integrated Gradients
**Explanation**
Integrated gradients calculates step by step the difference of a neutral input (a baseline, i.e. a black image) to a given input(sayres2019using). The gradient provides an estimator of which value weights most strongly for prediction.
**Application**
Integrated gradients was demonstrated by the authors on image models, text models, and a chemistry model. The method has even the ability to debug a model.
**Advantages**
The method does not require any modification to the model of interest and can directly be applied to the standard gradient operator. The paper presents two axioms that, according to the authors, should be fulfilled for an attribution method, namely \textit{sensitivity} and \textit{implementation invariance}. Sensitivity is given when a different feature between input and baseline is non-zero. Implementation invariance is given if a neural network always outputs the same prediction for a given input, regardless of its architecture. According to the authors, DeepLIFT, i.e., breaks both of these axioms.
Integrated gradients can be applied to any differentiable model.
**Limitation**
The limitation of this (and other image attribution methods) is that interactions between features as well as the logic of the network are not addressed.

#### SmoothGrad
**Explanation**
SmoothGrad is a model-specific, local and post-hoc explainability method that tries to reduce noise in saliency maps (also called sensitivity maps or pixel attribution maps) for model explanation(smilkov2017smoothgrad). In the neighborhood of an input image x, random examples are generated and blended with the sensitivity map by averaging.
**Application**
The authors apply the method to their own input images and parts of the MNIST dataset. The source code is freely available on [Github](https://github.com/PAIR-code/saliency).
**Advantages**
For some input images this method works better than comparable ones like Integrated Gradients(sundararajan2017axiomatic} or Guided Backpropagation\cite{springenberg2014striving). The method can also be combined with other methods.
**Limitation**
It remains unclear for which type of images the method works better than others. There is no discernible pattern for the examples shown in the paper.