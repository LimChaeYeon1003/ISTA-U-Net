import torch


# FGSM attack code
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image

def fgsm_attack_single_image(data, target, model, epsilon = 0.01):
    # Send the data and label to the device
    model.eval()
    
    # Set requires_grad attribute of tensor. Important for Attack
    data.requires_grad = True

    # Forward pass the data through the model
    # output = model(data)

    output = model(data)

    # Calculate the loss
    loss = torch.nn.MSELoss()(output, target)


    # Zero all existing gradients
    model.zero_grad()


    # Calculate gradients of model in backward pass
    loss.backward()

    # Collect datagrad
    data_grad = data.grad.data

    # Call FGSM Attack
    perturbed_data = fgsm_attack(data, epsilon, data_grad)

    orig_input_numpy = data.cpu().data.squeeze().numpy()
    orig_prediction_numpy = output.cpu().data.squeeze().numpy()

    adv_input_numpy = perturbed_data.cpu().data.squeeze().numpy()
    adv_output_numpy = torch.clamp(model(perturbed_data), 0, 1).cpu().data.squeeze().numpy()
    
    return orig_input_numpy, orig_prediction_numpy, adv_input_numpy, adv_output_numpy