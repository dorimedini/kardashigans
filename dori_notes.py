# The winning-lotto-ticket hypothesis basically says:
# "If you initialize parameters to P, there is a subset
# of these parameters P' s.t. training only the graph
# on the nodes corresponding to the parameters in P',
# initialized to the weights in P', does no worse than
# training the entire network initialized on via P"
#
# The all-layers-aren't-equal observation shows us that
# if we initialize the network to parameters P, there
# exist layers that are 'ambient' in the sense that if
# you reset the layer's parameters to their pre-training
# initial values, the resulting network (without re-
# training!) is still accurate.
#
# If G is the winning ticket (sub-graph of our FCN F) and
# L is some layer of F, is there any correlation between
# the ambience of L and the size of (G\cap L)?