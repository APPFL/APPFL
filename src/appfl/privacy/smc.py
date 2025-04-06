import crypten

crypten.init()


# Uses https://github.com/facebookresearch/CrypTen for SMC


# Pass a pytorch model from the client to encrypt it
def model_encryption(model):
    s_dict = model.state_dict()
    tensor_dict = {}
    for n, tensor in s_dict.items():
        e_t = crypten.cryptensor(tensor)
        tensor_dict[n] = e_t
    return tensor_dict


# Pass a pytorch model from the client to decrypt it
def model_decryption(tensor_dict, model):
    for k in tensor_dict:
        tensor_dict[k] = tensor_dict[k].get_plain_text()
    model.load_state_dict(tensor_dict)
    return model


# A simple weighted avg in the server to avg models from the clients
def smc_encrypted_model_aggregation(list_of_e_models, weights=None):
    avg_model = {}
    for i, model in enumerate(list_of_e_models):
        for k in model:
            if avg_model.get(k) is None:
                avg_model[k] = model[k] * weights[i]
            else:
                avg_model[k] += model[k] * weights[i]
    for k in avg_model:
        avg_model[k] /= sum(weights)
    return avg_model
