from keras import backend as K



def reset_to_checkpoint(model, checkpoint_weights):
    for idx in range(len(checkpoint_weights)):
        model.layers[idx].set_weights(checkpoint_weights[idx])
