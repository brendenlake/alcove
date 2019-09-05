# ALCOVE model of category learning in PyTorch

Kruschke, J. K. (1992). ALCOVE: an exemplar-based connectionist model of category learning. Psychological Review, 99(1), 22.

The main script runs ALCOVE on the 6 category learning problems of  Shephard, Hovland, and Jenkins (1961) and plots their learning curves.

There are a few modifications to Kruschke's original ALCOVE:

The original ALCOVE is modified as follows:
- With just two classes, we use only a single sigmoid output unit
- The original loss function has been replaced with either a maximum likelihood objective ("ll") or a hinge loss, which is a variant of the humble teacher used by Kruschke ("hinge")