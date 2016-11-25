
from [http://ufldl.stanford.edu/tutorial/supervised/FeatureExtractionUsingConvolution/](http://ufldl.stanford.edu/tutorial/supervised/FeatureExtractionUsingConvolution/)

Convolutions
===========

Natural images have the property of being ”‘stationary”’, meaning that
the statistics of one part of the image are the same as any other
part. This suggests that the features that we learn at one part of the
image can also be applied to other parts of the image, and we can use the
same features at all locations.

More precisely, having learned features over small (say 8x8) patches
sampled randomly from the larger image, we can then apply this learned 8x8
feature detector anywhere in the image. Specifically, we can take the
learned 8x8 features and ”‘convolve”’ them with the larger image, thus
obtaining a different feature activation value at each location in the
image.

To give a concrete example, suppose you have learned features on 8x8
patches sampled from a 96x96 image. Suppose further this was done with an
autoencoder that has 100 hidden units. To get the convolved features, for
every 8x8 region of the 96x96 image, that is, the 8x8 regions starting at
(1,1),(1,2),...(89,89)(1,1),(1,2)...…(89,89), you would extract the 8x8
patch, and run it through your trained sparse autoencoder to get the
feature activations. This would result in 100 sets 89x89 convolved
features.

![Convolution_semantic.gif](Convolution_semantic.gif)

Formally, given some large $r\times c$ iamges $x_{large}$, we first train a
sparse autoencoder on small $a\times b$ patches $x_{ small}$ sampled from
these images, learning $k$ features $f=\sigma(W_^{(1)} x_{small}+b^{(1)})$
)) (wher$\sigma$σσ is the sigmoid function), given by the weights
$W^{(1)}W^{(1)}$ and biases $b^{(1)}$ from the visible units to the hidden
units. For every a×ba×b patc$h_ $xs in the large image, we compute
$f_s=\sigma(W^{(1)}x_s+b^{(1)})$ , giving u$s_{ fconvolv}$, a
$k\times((r−a+1\times(×(c−b+1$ array of cray of convolved features.

