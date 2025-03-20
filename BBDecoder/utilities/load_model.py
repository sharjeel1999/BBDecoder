

def load_matching_weights(model, model_dict):
    '''
    model: The model to load the weights on to.
    model_dict: State Dict of the pretrained model.
    '''
    # Filter out unnecessary keys
    same_size = 0
    all_size = 0
    for k, v in pretrained_dict.items():
        all_size += 1
        if k in model_dict and model_dict[k].shape == v.shape:
            same_size += 1
            
    print('usage percentage: ', (same_size*100)/all_size)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    return model